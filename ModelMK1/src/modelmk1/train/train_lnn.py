from __future__ import annotations

import argparse

import joblib
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from modelmk1.common.paths import get_app_paths, resolve_data_file
from modelmk1.common.runtime import pick_device, set_seed, write_json
from modelmk1.data.loader import TickDataset, build_supervised_data, load_tick_df, resample_ticks
from modelmk1.models.lnn_model import MarketLNN


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
) -> float:
    model.train()
    total = 0.0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(y_batch)
    return total / len(loader.dataset)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total = 0.0
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        total += loss.item() * len(y_batch)
        preds.append(pred.detach().cpu().numpy().reshape(-1))
        targets.append(y_batch.detach().cpu().numpy().reshape(-1))
    return total / len(loader.dataset), np.concatenate(preds), np.concatenate(targets)


def run_training(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = pick_device(force_cpu=args.cpu)

    dataset_path = resolve_data_file(args.data_path)
    df = load_tick_df(str(dataset_path))
    sampled = resample_ticks(df, freq=args.resample_freq)
    bundle = build_supervised_data(sampled, seq_len=args.seq_len, horizon=args.horizon, include_stoch_rsi=True)

    split = int(0.8 * len(bundle.targets))
    x_train, x_val = bundle.sequences[:split], bundle.sequences[split:]
    y_train, y_val = bundle.targets[:split], bundle.targets[split:]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
    x_val_scaled = scaler.transform(x_val.reshape(-1, x_val.shape[-1])).reshape(x_val.shape)

    train_loader = DataLoader(TickDataset(x_train_scaled, y_train), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(TickDataset(x_val_scaled, y_val), batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MarketLNN(input_size=x_train.shape[-1], hidden_size=args.hidden_size, output_size=1, dropout=args.dropout).to(device)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    amp_scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    paths = get_app_paths()
    model_dir = paths.outputs / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    best_epoch = -1
    patience = 0
    val_pred = np.array([])
    val_target = np.array([])

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device, amp_scaler)
        val_loss, val_pred, val_target = _evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_columns": bundle.feature_columns,
                    "input_size": x_train.shape[-1],
                    "hidden_size": args.hidden_size,
                    "dropout": args.dropout,
                    "seq_len": args.seq_len,
                    "horizon": args.horizon,
                    "resample_freq": args.resample_freq,
                },
                model_dir / "lnn_best.pt",
            )
            joblib.dump(scaler, model_dir / "lnn_scaler.joblib")
        else:
            patience += 1

        print(f"Epoch {epoch}/{args.epochs} | train={train_loss:.6f} | val={val_loss:.6f} | device={device.type}")
        if patience >= args.early_stopping_patience:
            break

    metrics = {
        "best_val_loss": float(best_loss),
        "best_epoch": best_epoch,
        "val_mse": float(mean_squared_error(val_target, val_pred)),
        "val_mae": float(mean_absolute_error(val_target, val_pred)),
        "device": device.type,
        "dataset": str(dataset_path),
    }
    write_json(model_dir / "lnn_metrics.json", metrics)
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    print(run_training(args))


if __name__ == "__main__":
    main()
