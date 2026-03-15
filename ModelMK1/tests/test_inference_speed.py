from __future__ import annotations

from modelmk1.eval.inference_speed import build_pipeline_speed_report


def test_pipeline_speed_report_has_1m_and_5m_windows() -> None:
    report = build_pipeline_speed_report(
        avg_model_latency_ms=0.25,
        with_confidence=True,
        mc_samples=30,
    )

    assert "windows" in report
    assert "1m" in report["windows"]
    assert "5m" in report["windows"]
    assert report["windows"]["1m"]["budget_ms"] == 60000.0
    assert report["windows"]["5m"]["budget_ms"] == 300000.0
    assert report["windows"]["1m"]["budget_used_pct_high"] >= 0.0
