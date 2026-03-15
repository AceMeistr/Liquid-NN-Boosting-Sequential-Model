from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    from modelmk1.common.runtime import configure_logging

    configure_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="ModelMK1 launcher")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build", help="Build untrained model artifacts")

    pipeline_parser = subparsers.add_parser("pipeline", help="Build -> Optuna -> Train -> Hybrid")
    pipeline_parser.add_argument("--data-path", type=str, default=None)
    pipeline_parser.add_argument("--trials", type=int, default=20)
    pipeline_parser.add_argument("--build-only", action="store_true")

    lnn_parser = subparsers.add_parser("train-lnn", help="Train LNN")
    lnn_parser.add_argument("--data-path", type=str, default=None)

    graph_parser = subparsers.add_parser("train-graph", help="Train Route-J T-GCN graph model")
    graph_parser.add_argument("--data-path", type=str, default=None)

    optuna_parser = subparsers.add_parser("tune", help="Run Optuna")
    optuna_parser.add_argument("--data-path", type=str, default=None)
    optuna_parser.add_argument("--trials", type=int, default=20)
    optuna_parser.add_argument("--timeout", type=int, default=0)
    optuna_parser.add_argument("--study-name", type=str, default="modelmk1_lnn_optuna")
    optuna_parser.add_argument("--workers", type=int, default=3)

    xgb_optuna_parser = subparsers.add_parser("tune-xgb-spectral", help="Run Optuna for classification spectral XGBoost")
    xgb_optuna_parser.add_argument("--data-path", type=str, default=None)
    xgb_optuna_parser.add_argument("--trials", type=int, default=30)
    xgb_optuna_parser.add_argument("--timeout", type=int, default=0)
    xgb_optuna_parser.add_argument("--study-name", type=str, default="modelmk1_xgb_optuna")
    xgb_optuna_parser.add_argument("--jobs", type=int, default=1)
    xgb_optuna_parser.add_argument("--tickers", type=str, default=None)

    hybrid_parser = subparsers.add_parser("train-hybrid", help="Train hybrid residual model")
    hybrid_parser.add_argument("--data-path", type=str, default=None)
    hybrid_parser.add_argument("--resample-freq", type=str, default="1min")
    hybrid_parser.add_argument("--seq-len", type=int, default=64)
    hybrid_parser.add_argument("--horizon", type=int, default=15)
    hybrid_parser.add_argument("--train-ratio", type=float, default=0.8)
    hybrid_parser.add_argument("--use-best-studies", action="store_true", default=True)
    hybrid_parser.add_argument("--no-best-studies", dest="use_best_studies", action="store_false")
    hybrid_parser.add_argument("--clean-state", action="store_true", default=True)
    hybrid_parser.add_argument("--no-clean-state", dest="clean_state", action="store_false")
    hybrid_parser.add_argument("--cpu", action="store_true")

    spectral_parser = subparsers.add_parser("train-xgb-spectral", help="Train classification XGBoost on LNN-feature spectral indicators")
    spectral_parser.add_argument("--data-path", type=str, default=None)
    spectral_parser.add_argument("--top-n", type=int, default=10)
    spectral_parser.add_argument("--corr-window", type=int, default=60)
    spectral_parser.add_argument("--horizon", type=int, default=1)
    spectral_parser.add_argument("--seq-len", type=int, default=64)
    spectral_parser.add_argument("--train-ratio", type=float, default=0.8)
    spectral_parser.add_argument("--num-boost-round", type=int, default=700)
    spectral_parser.add_argument("--early-stopping-rounds", type=int, default=50)
    spectral_parser.add_argument("--robust-quantile-low", type=float, default=15.0)
    spectral_parser.add_argument("--robust-quantile-high", type=float, default=85.0)
    spectral_parser.add_argument("--tickers", type=str, default=None)

    predict_parser = subparsers.add_parser("predict", help="Generate predictions")
    predict_parser.add_argument("--data-path", type=str, default=None)
    predict_parser.add_argument("--with-confidence", action="store_true")

    subparsers.add_parser("backtest", help="Run backtest on predictions")

    export_parser = subparsers.add_parser("export-onnx", help="Export LNN to ONNX format")
    export_parser.add_argument("--seq-len", type=int, default=64)

    speed_parser = subparsers.add_parser("inference-speed", help="Benchmark inference speed and budgets")
    speed_parser.add_argument("--seq-len", type=int, default=64)
    speed_parser.add_argument("--mc-samples", type=int, default=30)
    speed_parser.add_argument("--cpu", action="store_true")

    subparsers.add_parser("info", help="Show model and system info")

    args = parser.parse_args()

    try:
        if args.command == "build":
            from modelmk1.train.build_models import run_build

            print(run_build(None, 128, 0.15, 64, 15, "1min"))
        elif args.command == "pipeline":
            from modelmk1.train.train_pipeline import run_pipeline

            print(run_pipeline(args.data_path, args.trials, args.build_only))
        elif args.command == "train-lnn":
            from modelmk1.train.train_lnn import build_parser, run_training

            lnn_args = build_parser().parse_args([])
            lnn_args.data_path = args.data_path
            print(run_training(lnn_args))
        elif args.command == "train-graph":
            from modelmk1.graph.train_graph import build_parser, run_graph_training

            graph_args = build_parser().parse_args([])
            graph_args.data_path = args.data_path
            print(run_graph_training(graph_args))
        elif args.command == "tune":
            from modelmk1.tuning.optuna_lnn import run_optuna

            print(run_optuna(args.trials, args.timeout, args.data_path, args.study_name, workers=args.workers))
        elif args.command == "tune-xgb-spectral":
            from modelmk1.tuning.optuna_xgb_spectral import run_optuna_xgb_spectral

            print(
                run_optuna_xgb_spectral(
                    trials=args.trials,
                    timeout=args.timeout,
                    data_path=args.data_path,
                    study_name=args.study_name,
                    jobs=args.jobs,
                    tickers_raw=args.tickers,
                )
            )
        elif args.command == "train-hybrid":
            from modelmk1.train.train_hybrid import build_parser, run_hybrid_training

            hybrid_args = build_parser().parse_args([])
            hybrid_args.data_path = args.data_path
            hybrid_args.resample_freq = args.resample_freq
            hybrid_args.seq_len = args.seq_len
            hybrid_args.horizon = args.horizon
            hybrid_args.train_ratio = args.train_ratio
            hybrid_args.use_best_studies = args.use_best_studies
            hybrid_args.clean_state = args.clean_state
            hybrid_args.cpu = args.cpu
            print(run_hybrid_training(hybrid_args))
        elif args.command == "train-xgb-spectral":
            from modelmk1.train.train_xgb_spectral import build_parser, run_xgb_spectral_training

            spectral_args = build_parser().parse_args([])
            spectral_args.data_path = args.data_path
            spectral_args.top_n = args.top_n
            spectral_args.corr_window = args.corr_window
            spectral_args.horizon = args.horizon
            spectral_args.seq_len = args.seq_len
            spectral_args.train_ratio = args.train_ratio
            spectral_args.num_boost_round = args.num_boost_round
            spectral_args.early_stopping_rounds = args.early_stopping_rounds
            spectral_args.robust_quantile_low = args.robust_quantile_low
            spectral_args.robust_quantile_high = args.robust_quantile_high
            spectral_args.tickers = args.tickers
            print(run_xgb_spectral_training(spectral_args))
        elif args.command == "predict":
            from modelmk1.predict.predict import main as predict_main

            predict_argv = ["predict.py"]
            if args.data_path:
                predict_argv += ["--data-path", args.data_path]
            if args.with_confidence:
                predict_argv.append("--with-confidence")
            sys.argv = predict_argv
            predict_main()
        elif args.command == "backtest":
            from modelmk1.eval.backtest import main as backtest_main

            backtest_main()
        elif args.command == "export-onnx":
            from modelmk1.common.paths import get_app_paths
            from modelmk1.models.onnx_export import export_to_onnx, validate_onnx

            paths = get_app_paths()
            model_dir = paths.outputs / "model"
            ckpt = model_dir / "lnn_best.pt"
            onnx_path = model_dir / "lnn_model.onnx"
            export_to_onnx(ckpt, onnx_path, seq_len=args.seq_len)
            validate_onnx(onnx_path, ckpt, seq_len=args.seq_len)
            print(f"ONNX model exported to {onnx_path}")
        elif args.command == "inference-speed":
            import torch

            from modelmk1.common.paths import get_app_paths
            from modelmk1.common.runtime import pick_device, safe_torch_load, write_json
            from modelmk1.eval.inference_speed import benchmark_model_latency_ms, build_pipeline_speed_report
            from modelmk1.models.checkpoint_utils import checkpoint_model_kwargs, is_legacy_checkpoint
            from modelmk1.models.lnn_model import MarketLNN

            device = pick_device(force_cpu=args.cpu)
            paths = get_app_paths()
            checkpoint = safe_torch_load(paths.outputs / "model" / "lnn_best.pt", map_location=device)
            model = MarketLNN(
                **checkpoint_model_kwargs(checkpoint),
            ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=not is_legacy_checkpoint(checkpoint))

            avg_ms = benchmark_model_latency_ms(
                model,
                seq_len=int(checkpoint.get("seq_len", args.seq_len)),
                input_size=int(checkpoint["input_size"]),
                device=device,
                warmup_runs=10,
                timed_runs=120,
            )
            report = build_pipeline_speed_report(
                avg_model_latency_ms=avg_ms,
                with_confidence=True,
                mc_samples=args.mc_samples,
            )
            write_json(paths.outputs / "inference_speed_budget.json", report)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(report)
        elif args.command == "info":
            import torch

            from modelmk1.common.paths import get_app_paths
            from modelmk1.common.runtime import get_device_info, pick_device

            device = pick_device()
            info = get_device_info(device)
            paths = get_app_paths()
            info["python_version"] = sys.version.split()[0]
            info["torch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            info["model_dir"] = str(paths.outputs / "model")
            info["training_data_dir"] = str(paths.training_data)
            info["data_engine_raw"] = str(paths.data_engine_raw)
            info["data_engine_exists"] = paths.data_engine_root.exists()
            for k, v in info.items():
                print(f"  {k}: {v}")
    except Exception:
        logger.exception("Command failed: %s", args.command)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
