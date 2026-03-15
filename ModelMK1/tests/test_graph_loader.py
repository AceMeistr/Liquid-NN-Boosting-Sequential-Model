from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from modelmk1.graph.graph_loader import (
	DynamicGraphDataset,
	build_dynamic_graph_bundle,
	correlation_adjacency,
	split_graph_end_indices,
)


def _make_long_format_graph_csv(path: Path, num_nodes: int = 4, length: int = 120) -> None:
	rng = np.random.default_rng(11)
	base = 100.0
	timestamps = pd.date_range("2025-01-01", periods=length, freq="min")

	rows: list[dict] = []
	for node_idx in range(num_nodes):
		drift = 0.0002 * (node_idx + 1)
		noise = rng.normal(loc=0.0, scale=0.003, size=length)
		returns = drift + noise
		prices = base * np.cumprod(1.0 + returns)
		ticker = f"STK{node_idx+1}"
		for ts, price in zip(timestamps, prices):
			rows.append({"timestamp": ts, "ticker": ticker, "price": float(price)})

	pd.DataFrame(rows).to_csv(path, index=False)


def test_correlation_adjacency_shape_and_diagonal() -> None:
	rng = np.random.default_rng(3)
	window = rng.normal(size=(8, 5)).astype(np.float32)
	adj = correlation_adjacency(window)

	assert adj.shape == (5, 5)
	assert np.all(np.isfinite(adj))
	assert np.allclose(np.diag(adj), 1.0)


def test_build_dynamic_graph_bundle_and_dataset(tmp_path: Path) -> None:
	csv_path = tmp_path / "constituents.csv"
	_make_long_format_graph_csv(csv_path, num_nodes=4, length=150)

	bundle = build_dynamic_graph_bundle(
		csv_path,
		seq_len=12,
		horizon=4,
		corr_window=5,
		expected_nodes=4,
	)

	assert bundle.returns.ndim == 2
	assert bundle.returns.shape[1] == 4
	assert len(bundle.constituents) == 4
	assert bundle.end_indices.size > 0

	train_idx, val_idx = split_graph_end_indices(bundle.end_indices, train_ratio=0.8)
	assert train_idx.size > 0
	assert val_idx.size > 0

	ds = DynamicGraphDataset(bundle, train_idx[:2], seq_len=12, corr_window=5)
	x_seq, adj_seq, target = ds[0]

	assert x_seq.shape == (12, 4, 1)
	assert adj_seq.shape == (12, 4, 4)
	assert target.shape == ()
