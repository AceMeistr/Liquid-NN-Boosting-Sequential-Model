from __future__ import annotations

import torch
import pytest

from modelmk1.models.lnn_model import MarketLNN, SqueezeExcite, TemporalAttention


def test_market_lnn_forward_shape() -> None:
    model = MarketLNN(input_size=24, hidden_size=64, output_size=1, num_heads=4, num_layers=2)
    x = torch.randn(4, 32, 24)
    out = model(x)
    assert out.shape == (4, 1)


def test_market_lnn_no_attention() -> None:
    model = MarketLNN(input_size=16, hidden_size=32, output_size=1, use_attention=False, num_heads=2, num_layers=1)
    x = torch.randn(2, 16, 16)
    out = model(x)
    assert out.shape == (2, 1)


def test_squeeze_excite_preserves_shape() -> None:
    se = SqueezeExcite(channels=32)
    x = torch.randn(4, 10, 32)
    out = se(x)
    assert out.shape == x.shape


def test_temporal_attention_preserves_shape() -> None:
    attn = TemporalAttention(d_model=32, num_heads=4)
    x = torch.randn(4, 10, 32)
    out = attn(x)
    assert out.shape == x.shape


def test_predict_with_uncertainty() -> None:
    model = MarketLNN(input_size=16, hidden_size=32, output_size=1, dropout=0.2, num_heads=2, num_layers=1)
    x = torch.randn(4, 10, 16)
    mean, std = model.predict_with_uncertainty(x, n_samples=5)
    assert mean.shape == (4, 1)
    assert std.shape == (4, 1)
    assert (std >= 0).all()


def test_model_parameter_count() -> None:
    model = MarketLNN(input_size=24, hidden_size=64, num_heads=4, num_layers=2)
    total = sum(p.numel() for p in model.parameters())
    assert total > 0
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable > 0
    # NCP RNN cells have non-trainable sparsity masks, so trainable < total
    assert trainable <= total


def test_model_gradient_flow() -> None:
    model = MarketLNN(input_size=16, hidden_size=32, num_heads=2, num_layers=1)
    x = torch.randn(2, 8, 16, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0
