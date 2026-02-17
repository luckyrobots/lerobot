from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LuckyEnginePolicyContract:
    action_dim: int
    state_dim: int
    camera_keys: tuple[str, ...]


DEFAULT_PIPER_CONTRACT = LuckyEnginePolicyContract(
    action_dim=7,
    state_dim=7,
    camera_keys=("CameraGripper", "CameraLeft", "CameraTop"),
)


def load_pretrained_config_json(pretrained_model_dir: str | Path) -> dict:
    pretrained_model_dir = Path(pretrained_model_dir)
    path = pretrained_model_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing config.json in pretrained_model dir: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _get_feature_shape(cfg: dict, key: str) -> list[int] | None:
    """
    Best-effort extraction of feature shape from a LeRobot config.json.
    """
    feats = cfg.get("input_features") or cfg.get("features") or {}
    if key not in feats:
        return None
    shape = feats[key].get("shape")
    if shape is None:
        return None
    return list(shape)


def _get_action_dim(cfg: dict) -> int | None:
    out = cfg.get("output_features") or {}
    act = out.get("action") or out.get("actions")
    if not isinstance(act, dict):
        return None
    shape = act.get("shape")
    if not shape:
        return None
    return int(shape[0])


def validate_checkpoint_contract(
    pretrained_model_dir: str | Path,
    *,
    contract: LuckyEnginePolicyContract = DEFAULT_PIPER_CONTRACT,
) -> None:
    """
    Validate that a checkpoint's `config.json` matches our expected LuckyEngine Piper setup.
    """
    cfg = load_pretrained_config_json(pretrained_model_dir)

    act_dim = _get_action_dim(cfg)
    if act_dim is not None and act_dim != contract.action_dim:
        raise ValueError(f"Action dim mismatch: expected {contract.action_dim}, got {act_dim}")

    state_shape = _get_feature_shape(cfg, "observation.state")
    if state_shape is not None:
        if len(state_shape) != 1 or int(state_shape[0]) != contract.state_dim:
            raise ValueError(f"observation.state shape mismatch: expected [{contract.state_dim}], got {state_shape}")

    for cam in contract.camera_keys:
        key = f"observation.images.{cam}"
        if _get_feature_shape(cfg, key) is None:
            # Some configs omit shapes; still require key presence if `input_features` is present.
            feats = cfg.get("input_features")
            if isinstance(feats, dict) and key not in feats:
                raise ValueError(f"Missing required input feature in config.json: {key}")


