#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""IMLE Policy configuration.

Implicit Maximum Likelihood Estimation (IMLE) policy as per
"IMLE Policy: Fast and Sample Efficient Visuomotor Policy Learning via Implicit Maximum Likelihood Estimation"
(RSS 2025).

The architecture reuses the same vision encoder (ResNet18 + SpatialSoftmax) and 1D U-Net generator as the
Diffusion policy, but removes the diffusion timestep embedding and replaces the iterative denoising with a
single-shot generator trained via RS-IMLE loss.
"""
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("imle")
@dataclass
class IMLEConfig(PreTrainedConfig):
    """Configuration class for the IMLE (Implicit Maximum Likelihood Estimation) policy.

    Defaults are configured for training on visuomotor tasks following the IMLE paper (RSS 2025).
    The generator is a 1D U-Net (same architecture as Diffusion policy, minus the timestep embedding)
    that maps noise to action trajectories in a single forward pass.

    Args:
        n_obs_steps: Number of observation steps to condition on.
        horizon: Total action trajectory length predicted by the generator.
        n_action_steps: Number of actions to actually execute per policy call.
        drop_n_last_frames: Number of trailing frames to drop from the dataset.
        normalization_mapping: Feature type to normalization mode mapping.
        vision_backbone: Name of the torchvision ResNet backbone.
        crop_shape: Random crop shape for image augmentation, or None to skip.
        crop_is_random: Whether to use random crops (True) or center crops (False).
        pretrained_backbone_weights: Pretrained weights for the vision backbone.
        use_group_norm: Whether to replace BatchNorm with GroupNorm in the backbone.
        spatial_softmax_num_keypoints: Number of spatial softmax keypoints.
        use_separate_rgb_encoder_per_camera: Whether each camera gets its own encoder.
        down_dims: Channel dimensions for each U-Net encoder stage.
        kernel_size: Convolution kernel size in the U-Net.
        n_groups: Number of groups for GroupNorm in the U-Net.
        use_film_scale_modulation: Whether to use scale+bias FiLM (True) or bias-only (False).
        n_samples_per_condition: Number of candidate samples per ground truth during training (RS-IMLE).
        epsilon: Rejection sampling threshold for RS-IMLE loss.
        n_inference_samples: Number of noise samples at inference (1 for standard, >1 for ensembling).
        use_trajectory_consistency: Whether to use trajectory consistency filtering at inference.
        do_mask_loss_for_padding: Whether to mask loss for padded action frames.
    """

    # Input / output structure.
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # Vision backbone (same as Diffusion).
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False

    # Generator U-Net (IMLE paper defaults -- smaller than Diffusion).
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    use_film_scale_modulation: bool = True

    # RS-IMLE training.
    n_samples_per_condition: int = 20
    epsilon: float = 0.03

    # Inference.
    n_inference_samples: int = 1
    use_trajectory_consistency: bool = False

    # Loss computation.
    do_mask_loss_for_padding: bool = False

    # Training presets (from reference IMLE repo: AdamW, cosine LR with 500 warmup).
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    optimizer_grad_clip_norm: float = 1.0
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        # Check that the horizon size and U-Net downsampling are compatible.
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor (which is determined "
                f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for "
                        f"`{key}`."
                    )

        # Check that all input images have the same shape.
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                    )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
