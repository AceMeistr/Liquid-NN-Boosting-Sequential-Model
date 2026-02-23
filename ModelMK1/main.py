from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    parser = argparse.ArgumentParser(description="ModelMK1 launcher")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build", help="Build untrained model artifacts")

    pipeline_parser = subparsers.add_parser("pipeline", help="Build -> Optuna -> Train -> Hybrid")
    pipeline_parser.add_argument("--data-path", type=str, default=None)
    pipeline_parser.add_argument("--trials", type=int, default=20)
    pipeline_parser.add_argument("--build-only", action="store_true")

    lnn_parser = subparsers.add_parser("train-lnn", help="Train LNN")
    lnn_parser.add_argument("--data-path", type=str, default=None)

    optuna_parser = subparsers.add_parser("tune", help="Run Optuna")
    optuna_parser.add_argument("--data-path", type=str, default=None)
    optuna_parser.add_argument("--trials", type=int, default=20)

    hybrid_parser = subparsers.add_parser("train-hybrid", help="Train hybrid residual model")
    hybrid_parser.add_argument("--data-path", type=str, default=None)

    predict_parser = subparsers.add_parser("predict", help="Generate predictions")
    predict_parser.add_argument("--data-path", type=str, default=None)

    subparsers.add_parser("backtest", help="Run backtest on predictions")
    args = parser.parse_args()

    if args.command == "build":
        from modelmk1.train.build_models import run_build

        print(run_build(24, 128, 0.15, 64, 15, "1min"))
    elif args.command == "pipeline":
        from modelmk1.train.train_pipeline import run_pipeline

        print(run_pipeline(args.data_path, args.trials, args.build_only))
    elif args.command == "train-lnn":
        from modelmk1.train.train_lnn import build_parser, run_training

        lnn_args = build_parser().parse_args([])
        lnn_args.data_path = args.data_path
        print(run_training(lnn_args))
    elif args.command == "tune":
        from modelmk1.tuning.optuna_lnn import run_optuna

        print(run_optuna(args.trials, 0, args.data_path, "modelmk1_lnn_optuna"))
    elif args.command == "train-hybrid":
        from modelmk1.train.train_hybrid import build_parser, run_hybrid_training

        hybrid_args = build_parser().parse_args([])
        hybrid_args.data_path = args.data_path
        print(run_hybrid_training(hybrid_args))
    elif args.command == "predict":
        from modelmk1.predict.predict import main as predict_main

        sys.argv = ["predict.py"] + (["--data-path", args.data_path] if args.data_path else [])
        predict_main()
    elif args.command == "backtest":
        from modelmk1.eval.backtest import main as backtest_main

        backtest_main()


if __name__ == "__main__":
    main()
