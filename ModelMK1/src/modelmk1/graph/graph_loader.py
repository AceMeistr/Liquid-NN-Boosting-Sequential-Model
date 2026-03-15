from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_PRICE_CANDIDATE_COLUMNS = ("price", "close", "idx_close", "last", "ltp")

@dataclass(frozen=True)
class GraphDataBundle:
	"""Prepared data for dynamic correlation graph modeling."""
	returns: np.ndarray
	targets: np.ndarray
	end_indices: np.ndarray
	constituents: list[str]
	timestamps: np.ndarray

class DynamicGraphDataset(Dataset):
	"""Dynamic graph sequences for T-GCN style training.
	x_seq shape: (seq_len, num_nodes, 1)
	adj_seq shape: (seq_len, num_nodes, num_nodes)
	target shape: scalar
	"""
	def __init__(self, bundle: GraphDataBundle, sample_end_indices: np.ndarray, seq_len: int, corr_window: int) -> None:
		if seq_len <= 0:
			raise ValueError("seq_len must be > 0")
		if corr_window <= 1:
			raise ValueError("corr_window must be > 1")
		self.bundle = bundle
		self.sample_end_indices = np.asarray(sample_end_indices, dtype=np.int64)
		self.seq_len = int(seq_len)
		self.corr_window = int(corr_window)
		if self.sample_end_indices.size == 0:
			raise ValueError("sample_end_indices cannot be empty")
	def __len__(self) -> int:
		return int(self.sample_end_indices.size)
	def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		end_idx = int(self.sample_end_indices[idx])
		start_idx = end_idx - self.seq_len + 1
		if start_idx < 0:
			raise IndexError("Invalid sample window; start index is negative")
		returns_seq = self.bundle.returns[start_idx : end_idx + 1]
		num_nodes = returns_seq.shape[1]
		x_seq = returns_seq[..., None].astype(np.float32)
		adj_seq = np.empty((self.seq_len, num_nodes, num_nodes), dtype=np.float32)
		for t_offset, t in enumerate(range(start_idx, end_idx + 1)):
			corr_start = t - self.corr_window + 1
			window = self.bundle.returns[corr_start : t + 1]
			adj_seq[t_offset] = correlation_adjacency(window)
		y = np.float32(self.bundle.targets[end_idx])
		return (
			torch.from_numpy(x_seq),
			torch.from_numpy(adj_seq),
			torch.tensor(y, dtype=torch.float32),
		)


def correlation_adjacency(window: np.ndarray, eps: float = 1e-6) -> np.ndarray:
	"""Compute a correlation-based adjacency matrix from a returns window.

	Parameters
	----------
	window:
		Array of shape (T, N) containing return observations.
	eps:
		Small constant added to denominator to prevent division by zero.

	Returns
	-------
	adj : np.ndarray of shape (N, N), values clipped to [-1, 1].
	"""
	if window.shape[0] < 2:
		num_nodes = window.shape[1]
		return np.eye(num_nodes, dtype=np.float32)
	corr = np.corrcoef(window.T)
	if not np.all(np.isfinite(corr)):
		num_nodes = window.shape[1]
		corr = np.eye(num_nodes, dtype=np.float32)
	return np.clip(corr, -1.0, 1.0).astype(np.float32)


def split_graph_end_indices(
	end_indices: np.ndarray,
	train_ratio: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
	"""Split sample end indices chronologically into train and validation sets.

	Parameters
	----------
	end_indices:
		Sorted array of valid sample end indices from :func:`build_dynamic_graph_bundle`.
	train_ratio:
		Fraction of samples to use for training.

	Returns
	-------
	train_indices, val_indices : pair of np.ndarray
	"""
	if not (0.0 < train_ratio < 1.0):
		raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
	n = len(end_indices)
	split_point = max(1, int(n * train_ratio))
	return end_indices[:split_point], end_indices[split_point:]


def _resolve_price_col(df: pd.DataFrame) -> str:
	"""Return the first matching price-like column name in *df*."""
	for col in _PRICE_CANDIDATE_COLUMNS:
		if col in df.columns:
			return col
	raise KeyError(
		f"No price column found in CSV. Expected one of {_PRICE_CANDIDATE_COLUMNS}. "
		f"Got columns: {list(df.columns)}"
	)


def build_dynamic_graph_bundle(
	csv_path: Path | str,
	seq_len: int = 60,
	horizon: int = 1,
	corr_window: int = 20,
	expected_nodes: int | None = None,
) -> GraphDataBundle:
	"""Load a long-format constituent CSV and build a :class:`GraphDataBundle`.

	The CSV must contain columns ``timestamp``, ``ticker``, and a price column
	(one of ``price``, ``close``, ``idx_close``, ``last``, ``ltp``).  Each row
	is one (timestamp, ticker) observation.

	Parameters
	----------
	csv_path:
		Path to the long-format CSV file.
	seq_len:
		Number of time-steps in each input sequence.
	horizon:
		Forward horizon (in bars) for the return target.
	corr_window:
		Minimum look-back window required for computing valid correlations;
		samples whose start index would fall before this window are excluded.
	expected_nodes:
		If provided, assert that the number of unique tickers matches this value.

	Returns
	-------
	GraphDataBundle
		Ready-to-use data bundle including returns matrix, targets, and valid end indices.
	"""
	csv_path = Path(csv_path)
	logger.info("Loading graph CSV: %s", csv_path)

	df = pd.read_csv(csv_path)

	required_cols = {"timestamp", "ticker"}
	missing = required_cols - set(df.columns)
	if missing:
		raise ValueError(f"CSV is missing required columns: {missing}")

	price_col = _resolve_price_col(df)

	df["timestamp"] = pd.to_datetime(df["timestamp"])
	df = df.sort_values(["timestamp", "ticker"]).reset_index(drop=True)

	constituents: list[str] = sorted(df["ticker"].unique().tolist())
	if expected_nodes is not None and len(constituents) != expected_nodes:
		raise ValueError(
			f"Expected {expected_nodes} unique tickers, found {len(constituents)}"
		)

	# Pivot to wide format: rows = timestamps, cols = tickers
	pivot = (
		df.pivot_table(index="timestamp", columns="ticker", values=price_col, aggfunc="last")
		.sort_index()
	)
	pivot = pivot[constituents]  # canonical column order

	# Forward-fill then backward-fill missing prices
	pivot = pivot.ffill().bfill()

	timestamps = pivot.index.to_numpy()
	prices: np.ndarray = pivot.values.astype(np.float64)

	# Log returns: r_t = log(p_t / p_{t-1})
	returns = np.log(prices[1:] / np.clip(prices[:-1], 1e-8, None)).astype(np.float32)
	timestamps = timestamps[1:]  # align with returns

	num_rows = returns.shape[0]

	# Targets: forward log-return of the mean constituent index
	# target[t] = log(mean_price[t+horizon] / mean_price[t])
	raw_index = prices.mean(axis=1)
	log_index = np.log(raw_index)
	targets = np.empty(num_rows, dtype=np.float32)
	for t in range(num_rows):
		future_t = min(t + 1 + horizon, len(log_index) - 1)
		targets[t] = float(log_index[future_t] - log_index[t + 1])

	# Valid end indices: need seq_len bars behind and corr_window bars for adjacency
	min_start = max(seq_len - 1, corr_window - 1)
	end_indices = np.arange(min_start, num_rows - horizon, dtype=np.int64)

	if end_indices.size == 0:
		raise ValueError(
			f"No valid samples: data has {num_rows} return rows but "
			f"seq_len={seq_len}, corr_window={corr_window}, horizon={horizon} "
			"require more rows."
		)

	logger.info(
		"Graph bundle ready: %d nodes, %d return rows, %d valid samples",
		len(constituents),
		num_rows,
		end_indices.size,
	)

	return GraphDataBundle(
		returns=returns,
		targets=targets,
		end_indices=end_indices,
		constituents=constituents,
		timestamps=timestamps,
	)
