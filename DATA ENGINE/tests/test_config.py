from __future__ import annotations


def test_pipeline_config_defaults(pipeline_config):
    assert pipeline_config.BARS_PER_SESSION == 375
    assert pipeline_config.ZSCORE_CLIP == 3.5
    assert pipeline_config.RFR_PATH.exists()
    assert pipeline_config.WEIGHTS_PATH.exists()
