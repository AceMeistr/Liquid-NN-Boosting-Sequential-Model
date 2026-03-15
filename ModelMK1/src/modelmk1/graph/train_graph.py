from __future__ import annotations

import argparse
import logging

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader

from modelmk1.common.paths import resolve_data_file, get_app_paths
from modelmk1.common.runtime import pick_device, set_seed, write_json
from modelmk1.graph.graph_loader import (
	DynamicGraphDataset,
	build_dynamic_graph_bundle,
	split_graph_end_indices,
)
from modelmk1.graph.tgcn_model import TGCNRegressor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def _train_epoch(
	model: TGCNRegressor,
	loader: DataLoader,
	optimizer: torch.optim.Optimizer,
	criterion: nn.Module,
	device: torch.device,
	grad_clip: float = 1.0,
) -> float:
	model.train()
	total_loss = 0.0
	n_seen = 0
	for x_seq, adj_seq, y in loader:
		x_seq = x_seq.to(device, non_blocking=True)
		adj_seq = adj_seq.to(device, non_blocking=True)
		y = y.unsqueeze(-1).to(device, non_blocking=True)
		optimizer.zero_grad(set_to_none=True)
		pred = model(x_seq, adj_seq)
		loss = criterion(pred, y)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
		optimizer.step()
		batch_n = int(y.size(0))
		total_loss += loss.item() * batch_n
		n_seen += batch_n
	return total_loss / max(n_seen, 1)


@torch.no_grad()
def _eval_epoch(
	model: TGCNRegressor,
	loader: DataLoader,
	criterion: nn.Module,
	device: torch.device,
) -> tuple[float, float, float]:
	model.eval()
	total_loss = 0.0
	n_seen = 0
	all_preds: list[np.ndarray] = []
	all_targets: list[np.ndarray] = []
	for x_seq, adj_seq, y in loader:
		x_seq = x_seq.to(device, non_blocking=True)
		adj_seq = adj_seq.to(device, non_blocking=True)
		y = y.unsqueeze(-1).to(device, non_blocking=True)
		pred = model(x_seq, adj_seq)
		loss = criterion(pred, y)
		batch_n = int(y.size(0))
		total_loss += loss.item() * batch_n
		n_seen += batch_n
		all_preds.append(pred.cpu().numpy())
		all_targets.append(y.cpu().numpy())
	preds = np.concatenate(all_preds).flatten()
	targets = np.concatenate(all_targets).flatten()
	mae = float(mean_absolute_error(targets, preds))
	rmse = float(mean_squared_error(targets, preds) ** 0.5)
	return total_loss / max(n_seen, 1), mae, rmse


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Train Route-J T-GCN graph model")
	parser.add_argument("--data-path", type=str, default=None,
						help="Path to long-format constituent CSV")
	parser.add_argument("--seq-len", type=int, default=60)
	parser.add_argument("--horizon", type=int, default=1)
	parser.add_argument("--corr-window", type=int, default=20)
	parser.add_argument("--graph-hidden", type=int, default=32)
	parser.add_argument("--temporal-hidden", type=int, default=64)
	parser.add_argument("--temporal-layers", type=int, default=1)
	parser.add_argument("--dropout", type=float, default=0.15)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--epochs", type=int, default=30)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--train-ratio", type=float, default=0.8)
	parser.add_argument("--seed", type=int, default=42)
	return parser


def run_graph_training(args: argparse.Namespace) -> str:
	"""Train a T-GCN model on constituent graph data.

	Parameters
	----------
	args:
		Parsed namespace from :func:`build_parser`.

	Returns
	-------
	str
		Human-readable summary of the training result.
	"""
	set_seed(args.seed)
	device = pick_device()
	paths = get_app_paths()

	data_path = resolve_data_file(getattr(args, "data_path", None))
	logger.info("Building dynamic graph bundle from %s", data_path)

	bundle = build_dynamic_graph_bundle(
		data_path,
		seq_len=args.seq_len,
		horizon=args.horizon,
		corr_window=args.corr_window,
	)

	train_idx, val_idx = split_graph_end_indices(bundle.end_indices, args.train_ratio)

	train_ds = DynamicGraphDataset(bundle, train_idx, seq_len=args.seq_len, corr_window=args.corr_window)
	val_ds = DynamicGraphDataset(bundle, val_idx, seq_len=args.seq_len, corr_window=args.corr_window)

	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
	val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

	num_nodes = bundle.returns.shape[1]
	logger.info("Nodes: %d | Train samples: %d | Val samples: %d", num_nodes, len(train_ds), len(val_ds))

	model = TGCNRegressor(
		in_features=1,
		graph_hidden_size=args.graph_hidden,
		temporal_hidden_size=args.temporal_hidden,
		temporal_layers=args.temporal_layers,
		dropout=args.dropout,
	).to(device)

	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	criterion = nn.HuberLoss(delta=1.0)

	model_dir = paths.outputs / "model"
	model_dir.mkdir(parents=True, exist_ok=True)
	best_ckpt_path = model_dir / "tgcn_best.pt"

	best_val_loss = float("inf")
	for epoch in range(1, args.epochs + 1):
		train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
		val_loss, val_mae, val_rmse = _eval_epoch(model, val_loader, criterion, device)

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			torch.save(
				{
					"epoch": epoch,
					"model_state_dict": model.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"val_loss": best_val_loss,
					"num_nodes": num_nodes,
					"in_features": 1,
					"graph_hidden_size": args.graph_hidden,
					"temporal_hidden_size": args.temporal_hidden,
					"temporal_layers": args.temporal_layers,
					"dropout": args.dropout,
					"seq_len": args.seq_len,
					"corr_window": args.corr_window,
					"constituents": bundle.constituents,
				},
				best_ckpt_path,
			)

		if epoch % 5 == 0 or epoch == 1:
			logger.info(
				"Epoch %3d/%d | train_loss=%.5f | val_loss=%.5f | MAE=%.5f | RMSE=%.5f",
				epoch, args.epochs, train_loss, val_loss, val_mae, val_rmse,
			)

	summary = {
		"best_val_loss": best_val_loss,
		"epochs_trained": args.epochs,
		"num_nodes": num_nodes,
		"train_samples": len(train_ds),
		"val_samples": len(val_ds),
		"checkpoint": str(best_ckpt_path),
	}
	write_json(model_dir / "tgcn_training_summary.json", summary)

	return (
		f"T-GCN training complete | nodes={num_nodes} | "
		f"best_val_loss={best_val_loss:.6f} | checkpoint={best_ckpt_path}"
	)
