# Data contract and preparation

## Supported input

The maintained pipeline accepts **the notebook's labeled raw-keypoint CSV layout**. Name training files `keypoints_with_labels_<subject_id>.csv` with a numeric subject ID. Keep each subject in one file. The supplied tutorial also mentions a timetable plus raw keypoints and a pre-segmented alternative; neither alternative is automatically converted by this repository. First produce the explicit CSV schema below and record the transformation you used.

| Column | Requirement |
| --- | --- |
| `frame_id` | Numeric, increasing, unique in each CSV. Consecutive frame IDs define a continuous segment. |
| `nose_x`, `nose_y`, …, `right_ankle_x`, `right_ankle_y` | Finite numeric coordinates for the 17 ordered joints in [`JOINTS`](../src/isas_pose/data.py). |
| `Action Label` | Required in training files; absent or ignored at inference. The allowed classes are listed below. |
| `timestamp` | Required only for the website's three-column submission export. Retained exactly as supplied. |
| `participant_id` | Required for that export unless supplied with `--participant-id`. |

The eight activity names shown in the supplied challenge screenshot are:

| Normal | Unusual |
| --- | --- |
| Sitting quietly | Head banging |
| Using phone | Throwing things |
| Walking | Attacking |
| Eating snacks | Biting nails |

`Throwing` is an accepted input alias and becomes `Throwing things`. Blank, missing and `None` label rows belong to no target class; training windows touching them are excluded. Any other unknown label fails preflight so it cannot silently enter an evaluation. Coordinate gaps or session boundaries need explicit handling before training; do not forward-fill missing pose coordinates without documenting and validating that choice.

## Preflight

```bash
isas-pose inspect --data-dir data/raw --output artifacts/data-audit.json
```

The audit records, per subject: source filename, SHA-256 checksum, row count, excluded-label rows, usable windows, usable bags and class counts after window labeling. The checksum lets collaborators confirm they evaluated the same bytes without committing private data. If a subject has too few adjacent windows for a bag, inspect fails early.

**Safe handling:** `data/`, `artifacts/`, CSVs and checkpoints are ignored by Git. Keep a separate authorized copy of the original data. The audit contains aggregate counts and hashes; review it before sharing if dataset terms restrict even metadata.

## Outputs

| Command | Output | Purpose |
| --- | --- | --- |
| `inspect` | JSON audit | Validate and identify source data before training |
| `evaluate` | JSON fold reports | Compare generalization across held-out subjects |
| `fit` | PyTorch `.pt` checkpoint | Final model trained on available labeled subjects |
| `predict` default | Input test columns plus populated `Action Label` | Tutorial's filled test-file format |
| `predict --format submission` | `participant_id,timestamp,predicted_label` | Screenshot's submission format |

One prediction is emitted for each input row. The frame-level labels expand smoothed bag decisions; they are not separately classified at frame resolution. Confirm the receiving form's expected artifact before uploading, as the tutorial and website screenshot describe two different formats. See [challenge notes](challenge.md).
