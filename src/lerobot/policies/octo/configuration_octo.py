from __future__ import annotations

import math
from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("octo")
@dataclass
class OctoConfig(PreTrainedConfig):
    """Octo policy wrapper (OctoPt) for LeRobot training.

    This policy wraps OctoPt and follows its recommended finetuning/inference contract:
    - observations: (B, T, C, H, W) uint8 images + timestep/pad masks
    - actions: normalized for training, and unnormalized for inference using dataset statistics
    """

    base_checkpoint: str = "hf://rail-berkeley/octo-small-1.5"
    # NOTE: We intentionally keep this as `str` (not `Literal[...]`) because some draccus versions
    # can't decode Literal types from CLI args on Windows.
    finetune_mode: str = "head_only"

    primary_image_key: str = "observation.images.CameraLeft"
    wrist_image_key: str = "observation.images.CameraSide"

    # Octo expects specific spatial sizes.
    primary_image_size: tuple[int, int] = (256, 256)
    wrist_image_size: tuple[int, int] = (128, 128)

    # Temporal history window length (number of observation timesteps).
    window_size: int = 2

    action_horizon: int = 4

    # Target normalization / safety caps (matches Piper shim defaults).
    max_translation_m: float = 0.02
    max_rotation_deg: float = 5.0
    gripper_open_threshold_m: float = 0.0175
    dh_is_offset: int = 1

    # Training presets (mirrors upstream Octo finetune recipe).
    optimizer_lr: float = 3e-4
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 2000
    scheduler_decay_steps: int = 100_000

    @property
    def observation_delta_indices(self) -> list | None:
        # Request a history window of size `window_size` ending at current timestep.
        # Example: window_size=2 -> [-1, 0]
        w = int(self.window_size)
        if w <= 0:
            raise ValueError(f"window_size must be >= 1, got {w}")
        return list(range(-(w - 1), 1))

    @property
    def action_delta_indices(self) -> list | None:
        # Request the future action chunk starting at current timestep.
        h = int(self.action_horizon)
        if h <= 0:
            raise ValueError(f"action_horizon must be >= 1, got {h}")
        return list(range(0, h))

    @property
    def reward_delta_indices(self) -> list | None:
        return None

    def validate_features(self) -> None:
        # Dataset-driven feature parsing will populate `input_features`/`output_features` before policy init.
        # Here we ensure the expected camera keys exist.
        if self.primary_image_key not in self.input_features:
            raise ValueError(
                f"Missing primary image key {self.primary_image_key!r} in inputs: {list(self.input_features)}"
            )
        if self.wrist_image_key not in self.input_features:
            raise ValueError(
                f"Missing wrist image key {self.wrist_image_key!r} in inputs: {list(self.input_features)}"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.optimizer_lr * 0.1,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def max_rotation_rad(self) -> float:
        return math.radians(float(self.max_rotation_deg))

    def __post_init__(self):
        super().__post_init__()
        allowed = {"head_only", "head_mlp_only", "full"}
        if self.finetune_mode not in allowed:
            raise ValueError(f"Invalid finetune_mode={self.finetune_mode!r}. Expected one of {sorted(allowed)}.")
        # Keep base config consistent with requested temporal window.
        self.n_obs_steps = int(self.window_size)


