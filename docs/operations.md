# Runbook for collaborators

## Before a run

1. Create a fresh Python 3.10+ virtual environment and install `python -m pip install -e ".[dev]"`. Create `artifacts/`, then record `python --version` and `python -m pip freeze > artifacts/environment.txt` for the run. Choose a PyTorch build supported by the machine.
2. Obtain the challenge CSVs through an authorized channel. Keep originals outside version control and place working copies under `data/raw/`.
3. Run `isas-pose inspect --data-dir data/raw --output artifacts/data-audit.json`. Review subject IDs, row counts, excluded `None` labels, class coverage and usable bags. Investigate any unexpected result before training.
4. Record the chosen `window-size`, `stride`, `bag-size`, seed, subject set and intended score. The notebook defaults and the paper's window study are different; see [experiments](experiments.md).

## Evaluation and final model

```bash
isas-pose evaluate --data-dir data/raw --output artifacts/loso.json --seed 42
isas-pose fit --data-dir data/raw --output artifacts/final.pt --seed 42
```

The JSON report contains each held-out subject's metrics and source-file SHA-256 hashes. The checkpoint stores its configuration, label order, source hashes and both model state dictionaries. Keep the report and checkpoint together with your environment record; do not overwrite a previous experiment's files. LOSO trains a fresh pair of models per fold and may be computationally expensive.

For a plumbing check, add `--triplets 20 --encoder-epochs 1 --mil-epochs 1` to `evaluate` or `fit`. Its metrics are not comparable to a full run. The tests use small synthetic CSVs and do not establish model quality.

## Prediction and delivery

```bash
isas-pose predict --input data/raw/test_data_keypoint.csv --checkpoint artifacts/final.pt --output artifacts/test_filled.csv

isas-pose predict --input data/raw/test_data_keypoint.csv --checkpoint artifacts/final.pt --format submission --participant-id P5 --output artifacts/submission.csv
```

Use the second command only when the input includes a real `timestamp` column. Replace `P5` with the authorized participant ID. If the column has a different name, pass `--timestamp-column NAME`. Inspect the output header, row count and several labels before delivery. The command requires a continuous `frame_id` sequence; run separate sessions separately.

## Failure guide

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Missing coordinate columns | CSV has a different keypoint schema | Map it explicitly to the [data contract](data.md); do not silently reorder joints. |
| Unknown activity label | Spelling differs from the eight accepted classes | Check source annotation and add a documented alias only if it means the same activity. |
| No usable windows or bags | Too few contiguous labeled frames or many `None` spans | Review the audit and selected window/stride; do not connect unrelated sessions. |
| Triplet training needs two classes | A training fold lacks class diversity | Check subject/class distribution; report the limitation rather than scoring an invalid fold. |
| CUDA unavailable | Requested device is not installed/visible | Use `--device cpu` or install the matching PyTorch/CUDA build. |
| Submission lacks timestamp | Input CSV has no real timestamp field | Obtain the source timestamp data; do not invent it from frame IDs. |
| Scores differ from paper | Different dataset, split, preprocessing, seed or parameters | Compare hashes and settings with [historical results](experiments.md) and [method notes](reproducibility.md). |

## Change control

Changes to label mapping, segmentation, model architecture, smoothing or metrics can alter scientific conclusions. Update the relevant docs, add a test that captures the new behavior, and describe the comparability impact in the pull request. Follow [CONTRIBUTING](../CONTRIBUTING.md).
