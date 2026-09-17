# Contributing

## Local workflow

1. Create an issue or short proposal describing the intended methodological change and whether it affects comparability with the paper.
2. Work on a branch; keep data and model checkpoints outside Git.
3. Install `python -m pip install -e ".[dev]"` in a virtual environment.
4. Run `python -m pytest` and `python -m ruff check src tests` before review. For data or preprocessing changes, also run `isas-pose inspect --data-dir data/raw --output artifacts/data-audit.json` on an authorized dataset.
5. In a pull request, include the data version, subject IDs, command, seed, device, package versions and metrics when changing training behavior. Never paste private rows or participant identifiers beyond approved IDs.

Keep the original notebook as an archive. Add maintained logic to `src/isas_pose/` and tests to `tests/`. Each change to preprocessing, labels, windowing, evaluation or frame mapping should explain its effect in `docs/reproducibility.md` and update the [architecture](docs/architecture.md) or [data contract](docs/data.md) when applicable. New experimental results belong in a separate report with data hashes and settings; do not replace the historical paper tables with them. Small functions and explicit input validation are preferred over hidden notebook state.

The repository has no code license yet. Contributors should confirm redistribution rights with the owners before submitting substantial external code.
