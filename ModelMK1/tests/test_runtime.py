from __future__ import annotations

from pathlib import Path

import torch

from modelmk1.common.runtime import retry, safe_torch_load


def test_retry_succeeds_after_transient_failures() -> None:
    calls = {"count": 0}

    def flaky() -> int:
        calls["count"] += 1
        if calls["count"] < 3:
            raise ValueError("transient")
        return 7

    out = retry(flaky, retries=3, delay_seconds=0)
    assert out == 7
    assert calls["count"] == 3


def test_safe_torch_load_roundtrip(tmp_path: Path) -> None:
    payload = {"model_state_dict": {"w": torch.tensor([1.0, 2.0])}, "input_size": 2}
    ckpt = tmp_path / "ckpt.pt"
    torch.save(payload, ckpt)

    loaded = safe_torch_load(ckpt, map_location=torch.device("cpu"))
    assert loaded["input_size"] == 2
    assert "model_state_dict" in loaded
