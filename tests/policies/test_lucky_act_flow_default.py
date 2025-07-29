import pytest

from lerobot.policies.lucky_act import LuckyACTConfig


def test_default_flow_fusion_enabled():
    """LuckyACTConfig should have flow fusion enabled out-of-the-box.

    This guards against silent regressions where duplicate field definitions
    or other refactors accidentally turn the feature off.
    """
    cfg = LuckyACTConfig()
    assert cfg.enable_flow_fusion is True, "Flow fusion should be enabled by default" 