# Reproducibility and method notes

## What the code implements

1. Parse each `keypoints_with_labels_<id>.csv` as a separate subject. Exclude windows containing missing, blank or `None` labels and rename `Throwing` to `Throwing things`.
2. Subtract each frame's mid hip point from all 17 joint coordinates and divide by shoulder distance plus `1e-6`.
3. Form overlapping 90-frame windows with stride 15. The most common frame label becomes the window label; a tie follows first occurrence, as in Python's `Counter.most_common` used by the notebook.
4. Sample 10,000 anchor-positive-negative triples. Train a three-layer Transformer pose encoder using triplet margin loss 1.0. Mean-pool its temporal outputs into 128-dimensional embeddings.
5. Group three consecutive embeddings **within each subject**. The middle window supplies the bag label. Train a two-layer Transformer with attention pooling and cross-entropy.
6. For inference, classify each bag, smooth bag labels with a three-prediction majority window, assign each result to the three contributing window starts, then forward/backward fill to provide one label for each input row. This preserves the notebook's frame mapping, including overwrite by later bags.

## Evaluation protocol

For each subject, train a fresh encoder and MIL model using only the other subjects. Encode the held-out subject **after** training the encoder. Build its bags separately, then compute multiclass accuracy and macro F1 plus binary unusual-class F1, precision, recall and a 2×2 confusion matrix. `artifacts/loso.json` contains each fold's report and the unweighted mean over folds. Classes absent from a test fold receive zero scores via `zero_division=0`; the complete label list remains fixed across folds. The unusual mapping follows the eight activities listed in the supplied challenge screenshot.

The notebook's early LOSO cell extracted embeddings of *all* subjects after each fold and appended them together. Downstream MIL splitting could therefore include embeddings made by an encoder trained on the held-out subject. The later final-training cells use a different subset of IDs and train on all of them. This repository separates strict evaluation from final training to avoid that ambiguity. Expect metrics to differ from the paper until the exact dataset version, subject list, seeds and experiment protocol are verified.

Each `evaluate` report and final checkpoint records SHA-256 hashes of the labeled source CSVs. The `inspect` command reports the same hashes with row, window and bag counts before training. The paper's supplied window study compares 90/45 and 120/60, while the archived notebook uses 90/15; the CLI defaults follow the notebook. See [historical experiments](experiments.md) before interpreting any comparison.

## Operational assumptions and limitations

- CSV order is chronological and `frame_id` is numeric, unique, and increasing within a file. Windows and bags never cross a frame-ID gap. The inference command requires one continuous test session; run it separately for each session if there are gaps. Recording boundaries with consecutive frame IDs cannot be detected automatically and should be split before use.
- The label of a training window is its majority frame label. This is a practical training target, not evidence of a separate bag-level annotation source. A window touching a `None` row is excluded; surrounding windows retain their original temporal positions.
- Missing joint coordinates are rejected. If the source data contains occlusions, define and document an imputation policy before training.
- A window is three seconds only if the data were sampled at 30 FPS. The code uses frame counts, not timestamps.
- Inference uses the notebook's start-frame assignment and fill strategy. Predictions near activity transitions can inherit neighboring bag labels. Output is a frame-level expansion of bag decisions, not independent frame inference.
- Random seeds are set for Python, NumPy and PyTorch. GPU kernels can still be nondeterministic; report the device and package versions for a publication-grade rerun.
- The full training run can be expensive, especially LOSO, and requires the actual subject CSVs. The code does not download or redistribute challenge data.
- Checkpoints contain a version, configuration, label order and both state dictionaries. Load checkpoints only from trusted sources.

## Paper and source relationship

The original notebook is preserved verbatim for provenance. The accompanying paper is *Recognizing Normal and Unusual Human Activities from Pose Sequences Using Triplet Embeddings and Multiple Instance Learning*, DOI [10.1088/1742-6596/3180/1/012001](https://doi.org/10.1088/1742-6596/3180/1/012001). Its method and reported results informed the pipeline. Statements inside the paper were treated as research content, not as repository instructions.
