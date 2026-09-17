# ISAS 2025 · Pose activity recognition

Code for **“Recognizing Normal and Unusual Human Activities from Pose Sequences Using Triplet Embeddings and Multiple Instance Learning”** ([DOI: 10.1088/1742-6596/3180/1/012001](https://doi.org/10.1088/1742-6596/3180/1/012001)). This repository turns the original Colab experiment into a command line research pipeline. The original [notebook](ISAS25.ipynb) remains as a historical record; use `src/isas_pose/` for new runs.

**Tóm tắt cho nhóm:** Repo chia luồng xử lý thành `data.py` (đọc và kiểm tra CSV), `models.py` (hai Transformer), `pipeline.py` (đánh giá LOSO, huấn luyện cuối và dự đoán) và `cli.py` (lệnh chạy). Chạy `evaluate` trước để đo khả năng tổng quát hóa theo người, rồi `fit` để tạo checkpoint cuối và `predict` để xuất nhãn từng frame. Lệnh `predict` hỗ trợ hai định dạng: file test được điền cột `Action Label` theo tutorial, hoặc CSV ba cột theo ảnh trang cuộc thi. Xem [đối chiếu yêu cầu cuộc thi](docs/challenge.md). Dữ liệu và checkpoint để trong `data/`, `artifacts/`; hai thư mục này không được đưa lên Git. Phần dưới mô tả chi tiết bằng tiếng Anh để tiện cộng tác và trích dẫn.

> **Status:** The source code has been reorganized and checked for syntax and small data transformations. The competition CSVs and trained weights are not included, so paper results have **not** been reproduced by this repository yet. See [reproducibility notes](docs/reproducibility.md).

## Documentation map

| If you need to… | Read |
| --- | --- |
| Understand the models and data flow | [Architecture](docs/architecture.md) |
| Prepare data and check the CSV/output schema | [Data contract](docs/data.md) |
| Compare a run with the paper's historical numbers | [Historical experiments](docs/experiments.md) |
| Run, troubleshoot and hand over an experiment | [Operations runbook](docs/operations.md) |
| Check what the organizer's tutorial and supplied screenshot say | [Challenge notes](docs/challenge.md) |
| Review scientific assumptions and notebook differences | [Reproducibility notes](docs/reproducibility.md) |
| Contribute code or documentation | [Contributing](CONTRIBUTING.md) |

## Pipeline

```mermaid
flowchart LR
  A[Subject CSVs: 17 joints + labels] --> B[Hip centering + shoulder scaling]
  B --> C[90-frame windows, stride 15]
  C --> D[Triplet-trained pose Transformer]
  D --> E[128-dimensional embeddings]
  E --> F[3 adjacent windows per bag]
  F --> G[MIL Transformer + attention]
  G --> H[Bag predictions]
  H --> I[Majority smoothing + frame mapping]
```

The paper's supplied [pipeline figure](docs/architecture.md) and two model diagrams provide a visual explanation of these stages. The source code remains the implementation reference.

Evaluation trains **both** networks afresh for each held-out subject. Final training uses all labeled subjects and creates one checkpoint for inference. These are separate operations; the LOSO fold models are not pooled into a final model.

## Quick start

Python 3.10+ is required. Install a PyTorch build suitable for your machine if the default package install does not work; see [PyTorch install instructions](https://pytorch.org/get-started/locally/).

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

Put authorized data under `data/raw/` using the names below. Data and weights are ignored by Git.

```text
data/raw/
├── keypoints_with_labels_1.csv
├── keypoints_with_labels_2.csv
├── ...
└── test_data_keypoint.csv
```

```bash
# 0. Check CSVs and record source hashes before training.
isas-pose inspect --data-dir data/raw --output artifacts/data-audit.json

# 1. Subject-isolated evaluation; writes metrics for every fold and their mean.
isas-pose evaluate --data-dir data/raw --output artifacts/loso.json

# 2. Train a model on every labeled subject for inference.
isas-pose fit --data-dir data/raw --output artifacts/final.pt

# 3a. Fill the test CSV with the Action Label column (tutorial format).
isas-pose predict --input data/raw/test_data_keypoint.csv --checkpoint artifacts/final.pt --output artifacts/test_filled.csv

# 3b. Or export the website's three-column submission format.
# Input must have a timestamp column; participant_id can come from input or --participant-id.
isas-pose predict --input data/raw/test_data_keypoint.csv --checkpoint artifacts/final.pt --format submission --participant-id P5 --output artifacts/submission.csv
```

For a short smoke run, use `--triplets 20 --encoder-epochs 1 --mil-epochs 1` on `evaluate` or `fit`. It checks the plumbing only; it does not measure model quality. `--device cpu` forces CPU. Full defaults match the notebook where possible: 90 frames, stride 15, three embeddings per bag, 10,000 triplets, 10 epochs for each network, and learning rate 0.001.

## Data contract

Each labeled file represents one subject. Its numeric suffix is the subject ID. Rows must have increasing unique `frame_id`, `Action Label`, and `x`/`y` columns for the 17 joints listed in [data.py](src/isas_pose/data.py). The test CSV uses the same columns except `Action Label`. Coordinates must be numeric and finite. Training windows containing missing, blank, or `None` labels are excluded because the tutorial says `None` is outside the eight target activities. The source alias `Throwing` is normalized to `Throwing things`; any other unknown label is reported. See the [full data contract](docs/data.md).

Files shorter than one window fail clearly. Each subject needs at least three adjacent windows to form a bag. Training windows and bags cannot span gaps in `frame_id` or excluded `None` rows. Test inference expects one continuous `frame_id` sequence per file. A training fold also needs at least two classes and two windows from one class for triplet sampling. These checks intentionally stop a run before a misleading result is produced.

**Privacy and access:** Raw data and participant-derived outputs are excluded from Git. Add only data you are permitted to use and publish. Do not commit checkpoints that may carry information from restricted data without checking the dataset terms.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/isas_pose/data.py` | CSV validation, normalization, window and bag construction |
| `src/isas_pose/models.py` | Transformer pose encoder and MIL classifier |
| `src/isas_pose/pipeline.py` | Triplet training, LOSO, final fit and prediction |
| `src/isas_pose/export.py` | Tutorial and website CSV output contracts |
| `src/isas_pose/audit.py` | Dataset preflight and SHA-256 fingerprints |
| `src/isas_pose/cli.py` | `isas-pose` commands |
| `tests/` | Data contract and boundary checks |
| `docs/reproducibility.md` | Method decisions, changes from notebook, limitations |
| `docs/challenge.md` | Verified competition requirements and format differences |
| `docs/architecture.md`, `docs/data.md`, `docs/experiments.md`, `docs/operations.md` | Visual architecture, schemas, historical evidence and runbook |
| `CONTRIBUTING.md` | Team workflow |
| `ISAS25.ipynb` | Original historical notebook, not the maintained entry point |

## Results and references

The published paper reports approximately 87% mean accuracy and F1 across eight classes. That number is a **paper claim**, not a verified outcome of this refactor. The paper is [Pham et al., Journal of Physics: Conference Series 3180 (2026) 012001](https://doi.org/10.1088/1742-6596/3180/1/012001). The [ISAS 2025 challenge website](https://isaschallenge2025.my.canva.site/) and its tutorial are summarized only where supplied materials make a point verifiable; see [challenge notes](docs/challenge.md). See the [challenge overview](https://arxiv.org/abs/2601.17049) for the general task and LOSO context.

No license has been assigned here. The paper's CC BY 4.0 notice applies to the paper, not automatically to this code or the competition data. Repository owners should choose a code license after confirming ownership and collaborator consent.
