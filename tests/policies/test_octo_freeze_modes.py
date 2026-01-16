import pytest


def test_octo_freeze_keys_match_recipe():
    from lerobot.policies.octo.modeling_octo import _freeze_keys

    assert _freeze_keys("full") is None
    assert _freeze_keys("head_only") == ["octo_transformer.*"]
    assert _freeze_keys("head_mlp_only") == [
        "octo_transformer.*",
        "heads_*.map_head.probe",
        "heads_*.map_head.MultiHeadDotProductAttention_0.*",
    ]
    with pytest.raises(ValueError):
        _freeze_keys("nope")


