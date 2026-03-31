from __future__ import annotations

import argparse
import logging
from collections.abc import Callable

import joblib
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader

from modelmk1.eval.cpcv import CPCVConfig, evaluate_cpcv_distribution
from modelmk1.eval.backtest import directional_accuracy, sharpe_ratio
from modelmk1.common.paths import get_app_paths, resolve_data_file
from modelmk1.common.runtime import (
    ModelEMA,
    build_scheduler,
    pick_device,
    safe_torch_load,
    set_seed,
    write_json,
)
from modelmk1.data.loader import (
    TickDataset,
    build_supervised_data,
    load_tick_df,
    prepare_split,
    resample_ticks,
)
from modelmk1.models.lnn_model import MarketLNN


logger = logging.getLogger(__name__)

EpochProgressCallback = Callable[[int, float, float, float, float, float], None]


_ANNUALIZATION_1MIN = 252.0 * 375.0


class DirectionalHuberLoss(nn.Module):
    """Blends standard Huber loss with a sign-agreement penalty."""
    def __init__(self, delta: float = 1.0, penalty: float = 1.0):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.penalty_weight = penalty

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base_loss = self.huber(pred, target)
        # When signs disagree, (pred * target) is negative, so -pred*target is positive.
        directional_penalty = torch.relu(-pred * target).mean()
        return base_loss + self.penalty_weight * directional_penalty


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    grad_clip: float = 1.0,
    ema: ModelEMA | None = None,
) -> float:
    model.train()
    total = 0.0
    n_seen = 0
    for x_batch, y_batch in loader:
        try:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            batch_n = int(y_batch.size(0))
            total += float(loss.item()) * batch_n
            n_seen += batch_n
        finally:
            del x_batch, y_batch, pred, loss
    return float(total) / max(n_seen, 1)


@torch.no_grad()
def _evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total = 0.0
    n_seen = 0
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for x_batch, y_batch in loader:
        try:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            batch_n = int(y_batch.size(0))
            total += float(loss.item()) * batch_n
            n_seen += batch_n
            preds.append(pred.detach().cpu().numpy().reshape(-1))
            targets.append(y_batch.detach().cpu().numpy().reshape(-1))
        finally:
            del x_batch, y_batch, pred, loss
    return float(total) / max(n_seen, 1), np.concatenate(preds), np.concatenate(targets)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def run_training(args: argparse.Namespace, on_epoch_end: EpochProgressCallback | None = None) -> dict:
    set_seed(args.seed)
    device = pick_device(force_cpu=args.cpu)

    dataset_path = resolve_data_file(args.data_path)
    df = load_tick_df(str(dataset_path))
    sampled = resample_ticks(df, freq=args.resample_freq)
    bundle = build_supervised_data(sampled, seq_len=args.seq_len, horizon=args.horizon, include_stoch_rsi=True)

    split = prepare_split(bundle, train_ratio=0.8)

    train_loader = DataLoader(
        TickDataset(split.x_seq_train, split.y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(device.type == "cuda"),
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        TickDataset(split.x_seq_val, split.y_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(device.type == "cuda"),
        prefetch_factor=2,
    )

    input_size = split.x_seq_train.shape[-1]
    model = MarketLNN(
        input_size=input_size,
        hidden_size=args.hidden_size,
        output_size=1,
        dropout=args.dropout,
        num_heads=getattr(args, "num_heads", 4),
        num_layers=getattr(args, "num_layers", 2),
        use_attention=getattr(args, "use_attention", True),
        backbone_type=getattr(args, "backbone_type", "mamba"),
        cnn_frontend=getattr(args, "cnn_frontend", "inception"),
        mamba_d_state=getattr(args, "mamba_d_state", 16),
        mamba_d_conv=getattr(args, "mamba_d_conv", 4),
        mamba_expand=getattr(args, "mamba_expand", 2),
    ).to(device)

    criterion = DirectionalHuberLoss(delta=1.0, penalty=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler_type = getattr(args, "scheduler", "cosine")
    warmup = getattr(args, "warmup_epochs", 2)
    scheduler = build_scheduler(optimizer, scheduler_type, args.epochs, warmup)

    amp_scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    use_ema = getattr(args, "use_ema", True)
    ema = ModelEMA(model, decay=getattr(args, "ema_decay", 0.999)) if use_ema else None

    paths = get_app_paths()
    model_dir = paths.outputs / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    best_epoch = -1
    patience = 0
    val_pred = np.array([])
    val_target = np.array([])

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(
            model, train_loader, optimizer, criterion, device, amp_scaler,
            grad_clip=getattr(args, "grad_clip_norm", 1.0),
            ema=ema,
        )

        # Validate with EMA weights if available
        if ema is not None:
            with ema.average_parameters(model):
                val_loss, val_pred, val_target = _evaluate(model, val_loader, criterion, device)
        else:
            val_loss, val_pred, val_target = _evaluate(model, val_loader, criterion, device)

        # Step scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        dir_acc = directional_accuracy(val_target, val_pred)
        strategy_direction = np.where(val_pred >= 0.0, 1.0, -1.0)
        strategy_returns = strategy_direction * val_target
        epoch_sharpe = sharpe_ratio(strategy_returns, annualization=_ANNUALIZATION_1MIN)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            patience = 0

            save_state = model.state_dict()
            if ema is not None:
                ema.apply(model)
                save_state = model.state_dict()
                ema.restore(model)

            torch.save(
                {
                    "model_state_dict": save_state,
                    "feature_columns": split.feature_columns,
                    "input_size": input_size,
                    "hidden_size": args.hidden_size,
                    "dropout": args.dropout,
                    "num_heads": getattr(args, "num_heads", 4),
                    "num_layers": getattr(args, "num_layers", 2),
                    "use_attention": getattr(args, "use_attention", True),
                    "backbone_type": getattr(args, "backbone_type", "mamba"),
                    "cnn_frontend": getattr(args, "cnn_frontend", "inception"),
                    "mamba_d_state": getattr(args, "mamba_d_state", 16),
                    "mamba_d_conv": getattr(args, "mamba_d_conv", 4),
                    "mamba_expand": getattr(args, "mamba_expand", 2),
                    "seq_len": args.seq_len,
                    "horizon": args.horizon,
                    "resample_freq": args.resample_freq,
                },
                model_dir / "lnn_best.pt",
            )
            joblib.dump(split.scaler, model_dir / "lnn_scaler.joblib")
        else:
            patience += 1

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %s/%s | train=%.6f | val=%.6f | dir_acc=%.3f | sharpe=%.3f | lr=%.2e | device=%s",
            epoch, args.epochs, train_loss, val_loss, dir_acc, epoch_sharpe, current_lr, device.type,
        )
        if on_epoch_end is not None:
            on_epoch_end(
                epoch,
                float(train_loss),
                float(val_loss),
                float(dir_acc),
                float(epoch_sharpe),
                float(current_lr),
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if patience >= args.early_stopping_patience:
            logger.info("Early stopping at epoch %s", epoch)
            break

    best_checkpoint = safe_torch_load(model_dir / "lnn_best.pt", map_location=device)
    best_model = MarketLNN(
        input_size=int(best_checkpoint["input_size"]),
        hidden_size=int(best_checkpoint["hidden_size"]),
        output_size=1,
        dropout=float(best_checkpoint.get("dropout", 0.1)),
        num_heads=int(best_checkpoint.get("num_heads", 4)),
        num_layers=int(best_checkpoint.get("num_layers", 2)),
        use_attention=bool(best_checkpoint.get("use_attention", True)),
        backbone_type=str(best_checkpoint.get("backbone_type", "mamba")),
        cnn_frontend=str(best_checkpoint.get("cnn_frontend", "inception")),
        mamba_d_state=int(best_checkpoint.get("mamba_d_state", 16)),
        mamba_d_conv=int(best_checkpoint.get("mamba_d_conv", 4)),
        mamba_expand=int(best_checkpoint.get("mamba_expand", 2)),
    ).to(device)
    best_model.load_state_dict(best_checkpoint["model_state_dict"])
    _, val_pred, val_target = _evaluate(best_model, val_loader, criterion, device)

    cpcv_summary = evaluate_cpcv_distribution(
        val_target,
        val_pred,
        config=CPCVConfig(
            n_groups=16,
            test_group_size=8,
            max_combinations=200,
            purge_gap=max(2 * int(args.horizon), 2),
            embargo=max(int(args.horizon), 1),
            seed=int(args.seed),
        ),
    )

    metrics = {
        "best_val_loss": float(best_loss),
        "best_epoch": best_epoch,
        "val_mse": float(mean_squared_error(val_target, val_pred)),
        "val_mae": float(mean_absolute_error(val_target, val_pred)),
        "val_directional_accuracy": directional_accuracy(val_target, val_pred),
        "cpcv": cpcv_summary,
        "device": device.type,
        "dataset": str(dataset_path),
    }
    write_json(model_dir / "lnn_metrics.json", metrics)

    del best_model
    del model
    del train_loader
    del val_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LNN with real data from Training Data or --data-path")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--resample-freq", type=str, default="1min")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["plateau", "cosine", "cosine_warm"])
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--use-attention", action="store_true", default=False)
    parser.add_argument("--no-attention", dest="use_attention", action="store_false")
    parser.add_argument("--backbone-type", type=str, default="mamba", choices=["mamba", "attention_gru"])
    parser.add_argument("--cnn-frontend", type=str, default="inception", choices=["inception", "none"])
    parser.add_argument("--mamba-d-state", type=int, default=16)
    parser.add_argument("--mamba-d-conv", type=int, default=4)
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--no-ema", dest="use_ema", action="store_false")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    print(run_training(args))


if __name__ == "__main__":
    main()
