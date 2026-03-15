from __future__ import annotations

import torch

from modelmk1.graph.tgcn_model import TGCNRegressor


def test_tgcn_forward_shape() -> None:
	model = TGCNRegressor(
		in_features=1,
		graph_hidden_size=16,
		temporal_hidden_size=32,
		temporal_layers=1,
		dropout=0.1,
	)

	x_seq = torch.randn(3, 10, 6, 1)
	adj_seq = torch.randn(3, 10, 6, 6)
	adj_seq = 0.5 * (adj_seq + adj_seq.transpose(-1, -2))

	out = model(x_seq, adj_seq)
	assert out.shape == (3, 1)


def test_tgcn_backprop() -> None:
	model = TGCNRegressor(
		in_features=1,
		graph_hidden_size=8,
		temporal_hidden_size=16,
		temporal_layers=1,
		dropout=0.1,
	)

	x_seq = torch.randn(2, 8, 5, 1, requires_grad=True)
	adj_seq = torch.eye(5).repeat(2, 8, 1, 1)
	target = torch.randn(2, 1)

	out = model(x_seq, adj_seq)
	loss = torch.nn.functional.mse_loss(out, target)
	loss.backward()
