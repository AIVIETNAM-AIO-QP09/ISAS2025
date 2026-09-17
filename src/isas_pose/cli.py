"""Command line entry point: evaluate, fit, predict."""

import argparse
from pathlib import Path

from .audit import inspect_data, write_audit
from .config import Config


def main() -> None:
    parser = argparse.ArgumentParser(description="ISAS 2025 pose recognition pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Validate CSVs and summarize windows and bags")
    inspect.add_argument("--data-dir", type=Path, required=True)
    inspect.add_argument("--output", type=Path)
    inspect.add_argument("--window-size", type=int, default=90)
    inspect.add_argument("--stride", type=int, default=15)
    inspect.add_argument("--bag-size", type=int, default=3)
    for name in ("evaluate", "fit"):
        command = sub.add_parser(name)
        command.add_argument("--data-dir", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--window-size", type=int, default=90)
        command.add_argument("--stride", type=int, default=15)
        command.add_argument("--bag-size", type=int, default=3)
        command.add_argument("--triplets", type=int, default=10000)
        command.add_argument("--encoder-epochs", type=int, default=10)
        command.add_argument("--mil-epochs", type=int, default=10)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    command = sub.add_parser("predict")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--checkpoint", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    command.add_argument("--format", default="test-file", choices=("test-file", "submission"))
    command.add_argument("--participant-id", help="Required for submission if absent from input CSV")
    command.add_argument("--timestamp-column", default="timestamp")
    args = parser.parse_args()
    if args.command == "inspect":
        write_audit(inspect_data(args.data_dir, args.window_size, args.stride, args.bag_size),
                    args.output)
        return
    from .pipeline import evaluate, fit_final, predict, resolve_device

    device = resolve_device(args.device)
    if args.command == "predict":
        predict(args.input, args.checkpoint, args.output, device, output_format=args.format,
                participant_id=args.participant_id, timestamp_column=args.timestamp_column)
        return
    config = Config(window_size=args.window_size, stride=args.stride, bag_size=args.bag_size,
                    triplets=args.triplets, encoder_epochs=args.encoder_epochs,
                    mil_epochs=args.mil_epochs, seed=args.seed)
    if args.command == "evaluate":
        report = evaluate(args.data_dir, config, device, args.output)
        print(f"mean macro F1: {report['mean_macro_f1']:.4f}")
        print(f"mean unusual F1: {report['mean_unusual_f1']:.4f}")
    else:
        fit_final(args.data_dir, config, device, args.output)


if __name__ == "__main__":
    main()
