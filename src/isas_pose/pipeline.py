"""Subject-isolated evaluation, final training, and frame-level prediction."""

import json
import random
from collections import Counter, deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .audit import data_fingerprints
from .config import Config
from .data import Windows, activity_type, load_subjects, make_bags, make_windows
from .export import format_predictions
from .models import MILTransformer, PoseEncoder


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(name)


def _flat(subjects: dict[int, Windows]):
    poses = np.concatenate([value.poses for value in subjects.values()])
    labels = [label for value in subjects.values() for label in value.labels]
    return poses, labels


def _sample_triplets(labels: list[str], count: int, rng: random.Random):
    by_label: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        by_label.setdefault(label, []).append(index)
    eligible = [label for label, indices in by_label.items() if len(indices) >= 2]
    if not eligible or len(by_label) < 2:
        raise ValueError("Triplet training needs two classes and two windows in one class")
    other = {label: [x for x in by_label if x != label] for label in eligible}
    return np.asarray([
        (*rng.sample(by_label[label], 2), rng.choice(by_label[rng.choice(other[label])]))
        for label in (rng.choice(eligible) for _ in range(count))
    ], dtype=np.int64)


def train_encoder(subjects: dict[int, Windows], config: Config, device: torch.device) -> PoseEncoder:
    poses, labels = _flat(subjects)
    triplets = _sample_triplets(labels, config.triplets, random.Random(config.seed))
    # Keep original sequence length configurable; flatten only joint coordinates.
    data = torch.from_numpy(poses.reshape(len(poses), config.window_size, -1))
    ids = torch.from_numpy(triplets)
    loader = DataLoader(TensorDataset(ids), batch_size=config.encoder_batch_size, shuffle=True)
    model = PoseEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for epoch in range(config.encoder_epochs):
        model.train()
        losses = []
        for (batch,) in loader:
            anchor, positive, negative = (data[batch[:, i]].to(device) for i in range(3))
            loss = F.triplet_margin_loss(model(anchor), model(positive), model(negative), margin=1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"encoder epoch {epoch + 1}/{config.encoder_epochs}: loss={np.mean(losses):.4f}")
    return model


def embed(windows: Windows, model: PoseEncoder, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    poses = torch.from_numpy(windows.poses.reshape(len(windows.poses), windows.poses.shape[1], -1))
    loader = DataLoader(TensorDataset(poses), batch_size=batch_size)
    with torch.inference_mode():
        parts = [model(batch.to(device)).cpu().numpy() for (batch,) in loader]
    return np.concatenate(parts)


def _bagged(subjects: dict[int, Windows], encoder: PoseEncoder, config: Config, device: torch.device):
    bags, labels = [], []
    for uid, windows in subjects.items():
        embs = embed(windows, encoder, device, config.encoder_batch_size)
        try:
            subject_bags, subject_labels = make_bags(
                embs, windows.labels, config.bag_size,
                frame_ids=windows.frame_ids, expected_step=config.stride)
        except ValueError as exc:
            raise ValueError(f"Subject {uid}: {exc}") from exc
        bags.append(subject_bags)
        labels.extend(subject_labels)
    return np.concatenate(bags), labels


def train_mil(bags: np.ndarray, labels: list[str], label_names: list[str], config: Config,
              device: torch.device) -> MILTransformer:
    label_to_index = {name: index for index, name in enumerate(label_names)}
    x = torch.from_numpy(bags)
    y = torch.tensor([label_to_index[label] for label in labels], dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=config.mil_batch_size, shuffle=True)
    model = MILTransformer(num_classes=len(label_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for epoch in range(config.mil_epochs):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            loss = F.cross_entropy(model(batch_x.to(device)), batch_y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"MIL epoch {epoch + 1}/{config.mil_epochs}: loss={np.mean(losses):.4f}")
    return model


def predict_bags(model: MILTransformer, bags: np.ndarray, device: torch.device,
                 batch_size: int) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(bags)), batch_size=batch_size)
    with torch.inference_mode():
        return np.concatenate([model(x.to(device)).argmax(1).cpu().numpy() for (x,) in loader])


def evaluate(data_dir: Path, config: Config, device: torch.device, output: Path) -> dict:
    subjects = load_subjects(data_dir, config.window_size, config.stride)
    label_names = sorted({label for windows in subjects.values() for label in windows.labels})
    label_to_index = {label: i for i, label in enumerate(label_names)}
    results = []
    for held_out in sorted(subjects):
        print(f"LOSO held-out subject: {held_out}")
        seed_everything(config.seed)
        train = {uid: windows for uid, windows in subjects.items() if uid != held_out}
        encoder = train_encoder(train, config, device)
        x_train, y_train = _bagged(train, encoder, config, device)
        model = train_mil(x_train, y_train, label_names, config, device)
        x_test, y_test = _bagged({held_out: subjects[held_out]}, encoder, config, device)
        y_true = [label_to_index[label] for label in y_test]
        y_pred = predict_bags(model, x_test, device, config.mil_batch_size).tolist()
        unusual_true = [int(activity_type(label_names[i]) == "unusual") for i in y_true]
        unusual_pred = [int(activity_type(label_names[i]) == "unusual") for i in y_pred]
        results.append({
            "subject_id": held_out,
            "bags": len(y_true),
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, labels=list(range(len(label_names))),
                                 average="macro", zero_division=0),
            "unusual_f1": f1_score(unusual_true, unusual_pred, zero_division=0),
            "unusual_precision": precision_score(unusual_true, unusual_pred, zero_division=0),
            "unusual_recall": recall_score(unusual_true, unusual_pred, zero_division=0),
            "binary_confusion_matrix": confusion_matrix(unusual_true, unusual_pred,
                                                          labels=[0, 1]).tolist(),
            "report": classification_report(y_true, y_pred, labels=list(range(len(label_names))),
                                            target_names=label_names, output_dict=True,
                                            zero_division=0),
        })
    report = {"config": asdict(config), "data_sha256": data_fingerprints(data_dir),
              "labels": label_names, "folds": results,
              "mean_accuracy": float(np.mean([r["accuracy"] for r in results])),
              "mean_macro_f1": float(np.mean([r["macro_f1"] for r in results])),
              "mean_unusual_f1": float(np.mean([r["unusual_f1"] for r in results]))}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def fit_final(data_dir: Path, config: Config, device: torch.device, output: Path) -> None:
    subjects = load_subjects(data_dir, config.window_size, config.stride)
    label_names = sorted({label for windows in subjects.values() for label in windows.labels})
    seed_everything(config.seed)
    encoder = train_encoder(subjects, config, device)
    bags, labels = _bagged(subjects, encoder, config, device)
    model = train_mil(bags, labels, label_names, config, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "config": asdict(config),
                "data_sha256": data_fingerprints(data_dir), "labels": label_names,
                "encoder": encoder.cpu().state_dict(), "mil": model.cpu().state_dict()}, output)


def _smooth(labels: list[str], size: int) -> list[str]:
    window: deque[str] = deque(maxlen=size)
    result = []
    for label in labels:
        window.append(label)
        result.append(Counter(window).most_common(1)[0][0])
    return result


def predict(input_csv: Path, checkpoint: Path, output_csv: Path, device: torch.device,
            *, output_format: str = "test-file", participant_id: str | None = None,
            timestamp_column: str = "timestamp") -> None:
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("PyTorch with weights_only support is required") from exc
    if state.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    config = Config(**state["config"])
    label_names = state["labels"]
    encoder = PoseEncoder().to(device)
    encoder.load_state_dict(state["encoder"])
    model = MILTransformer(len(label_names)).to(device)
    model.load_state_dict(state["mil"])
    frame = pd.read_csv(input_csv)
    if not np.all(np.diff(frame["frame_id"].to_numpy()) == 1):
        raise ValueError("Test CSV has frame gaps; predict each continuous session separately")
    windows = make_windows(frame, config.window_size, config.stride, labeled=False)
    embeddings = embed(windows, encoder, device, config.encoder_batch_size)
    bags, _ = make_bags(embeddings, None, config.bag_size)
    indices = predict_bags(model, bags, device, config.mil_batch_size)
    labels = _smooth([label_names[i] for i in indices], config.smoothing_window)
    # The notebook assigned each bag's label to its three window starts, with later
    # bags overwriting earlier ones. Preserve that mapping and fill remaining frames.
    frame_to_label = {}
    for i, label in enumerate(labels):
        for frame_id in windows.frame_ids[i:i + config.bag_size]:
            frame_to_label[frame_id] = label
    predictions = frame["frame_id"].map(frame_to_label).ffill().bfill()
    if predictions.isna().any():
        raise RuntimeError("Prediction coverage is incomplete")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result = format_predictions(frame, predictions, output_format=output_format,
                                participant_id=participant_id,
                                timestamp_column=timestamp_column)
    result.to_csv(output_csv, index=False)
