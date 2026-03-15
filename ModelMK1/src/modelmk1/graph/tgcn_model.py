from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
	"""Simple graph convolution with normalized dynamic adjacency."""

	def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
		super().__init__()
		self.weight = nn.Parameter(torch.empty(in_features, out_features))
		if bias:
			self.bias = nn.Parameter(torch.zeros(out_features))
		else:
			self.register_parameter("bias", None)
		nn.init.xavier_uniform_(self.weight)

	@staticmethod
	def _normalize_adjacency(adj: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
		# adj: (B, N, N)
		bsz, num_nodes, _ = adj.shape
		eye = torch.eye(num_nodes, device=adj.device, dtype=adj.dtype).unsqueeze(0).expand(bsz, -1, -1)
		a_hat = adj + eye
		deg = a_hat.sum(dim=-1).clamp_min(eps)
		deg_inv_sqrt = deg.pow(-0.5)
		return deg_inv_sqrt.unsqueeze(-1) * a_hat * deg_inv_sqrt.unsqueeze(-2)

	def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
		# x: (B, N, Fin), adj: (B, N, N)
		norm_adj = self._normalize_adjacency(adj)
		support = torch.matmul(x, self.weight)
		out = torch.bmm(norm_adj, support)
		if self.bias is not None:
			out = out + self.bias
		return out


class TGCNRegressor(nn.Module):
	"""Temporal GCN regressor for dynamic constituent-correlation graphs."""

	def __init__(
		self,
		in_features: int,
		graph_hidden_size: int = 32,
		temporal_hidden_size: int = 64,
		temporal_layers: int = 1,
		dropout: float = 0.15,
	) -> None:
		super().__init__()
		self.gconv1 = GraphConv(in_features, graph_hidden_size)
		self.gconv2 = GraphConv(graph_hidden_size, graph_hidden_size)
		self.dropout = nn.Dropout(dropout)

		self.temporal = nn.GRU(
			input_size=graph_hidden_size * 2,
			hidden_size=temporal_hidden_size,
			num_layers=temporal_layers,
			batch_first=True,
			dropout=dropout if temporal_layers > 1 else 0.0,
		)
		self.head = nn.Sequential(
			nn.Linear(temporal_hidden_size, temporal_hidden_size // 2),
			nn.SiLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(temporal_hidden_size // 2, 1),
		)

	def forward(self, x_seq: torch.Tensor, adj_seq: torch.Tensor) -> torch.Tensor:
		# x_seq: (B, T, N, F), adj_seq: (B, T, N, N)
		if x_seq.dim() != 4:
			raise ValueError("x_seq must have shape (B, T, N, F)")
		if adj_seq.dim() != 4:
			raise ValueError("adj_seq must have shape (B, T, N, N)")

		seq_len = x_seq.size(1)
		pooled_steps: list[torch.Tensor] = []

		for t in range(seq_len):
			x_t = x_seq[:, t]
			adj_t = adj_seq[:, t]

			h = self.gconv1(x_t, adj_t)
			h = F.gelu(h)
			h = self.dropout(h)
			h = self.gconv2(h, adj_t)
			h = F.gelu(h)

			# Pool node states while preserving global topological signal.
			pooled = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=-1)
			pooled_steps.append(pooled)

		temporal_input = torch.stack(pooled_steps, dim=1)
		temporal_out, _ = self.temporal(temporal_input)
		last = self.dropout(temporal_out[:, -1, :])
		return self.head(last)
