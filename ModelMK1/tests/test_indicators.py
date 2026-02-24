from __future__ import annotations

from modelmk1.features.indicators import get_feature_schema


def test_feature_schema_sizes() -> None:
    full = get_feature_schema(include_stoch_rsi=True)
    base = get_feature_schema(include_stoch_rsi=False)

    assert len(full) == len(base) + 3
    assert "stoch_rsi" in full
    assert "stoch_rsi" not in base
