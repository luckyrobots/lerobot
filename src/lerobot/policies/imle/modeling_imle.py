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
"""IMLE Policy: single-shot generative visuomotor policy via Implicit Maximum Likelihood Estimation.

Paper: "IMLE Policy: Fast and Sample Efficient Visuomotor Policy Learning via
Implicit Maximum Likelihood Estimation" (RSS 2025).

Architecture:
  - Vision encoder: ResNet18 + SpatialSoftmax (reused from Diffusion policy).
  - Generator: 1D conditional U-Net (same as Diffusion U-Net, minus the timestep embedding).
  - Training: RS-IMLE loss (rejection sampling IMLE).
  - Inference: single forward pass through the generator (no iterative denoising).
"""

from collections import deque

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalResidualBlock1d,
    DiffusionConv1dBlock,
    DiffusionRgbEncoder,
)
from lerobot.policies.imle.configuration_imle import IMLEConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


# ═══════════════════════════════════════════════════════════════════════════════
#  Top-level policy class (follows DiffusionPolicy pattern exactly)
# ═══════════════════════════════════════════════════════════════════════════════


class IMLEPolicy(PreTrainedPolicy):
    """IMLE Policy: single-shot generative visuomotor policy.

    Maps a random noise vector to an action trajectory conditioned on observations,
    in a single forward pass (no iterative denoising). Trained with RS-IMLE loss.
    """

    config_class = IMLEConfig
    name = "imle"

    def __init__(self, config: IMLEConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self._queues = None
        self.model = IMLEModel(config)
        self.reset()

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`."""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        actions = self.model.generate_actions(batch)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        Uses queue-based action chunking (identical to DiffusionPolicy).
        """
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        loss = self.model.compute_loss(batch)
        return loss, None


# ═══════════════════════════════════════════════════════════════════════════════
#  Core model (owns the generator and vision encoder)
# ═══════════════════════════════════════════════════════════════════════════════


class IMLEModel(nn.Module):
    """Inner model that owns the generator U-Net and vision encoders.

    Follows the same structure as DiffusionModel but without the noise scheduler.
    """

    def __init__(self, config: IMLEConfig):
        super().__init__()
        self.config = config

        # Build observation encoders (same logic as DiffusionModel).
        global_cond_dim = self.config.robot_state_feature.shape[0]
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        # Generator U-Net: conditioning is *only* global_cond (no timestep embedding).
        self.generator = IMLEGeneratorUnet1d(
            config, global_cond_dim=global_cond_dim * config.n_obs_steps
        )

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and concatenate with the state vector.

        Identical to DiffusionModel._prepare_global_conditioning.
        """
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]

        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                )
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                img_features = self.rgb_encoder(
                    einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                )
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """Generate actions via a single forward pass through the generator.

        Args:
            batch: Observation batch with OBS_STATE and optionally OBS_IMAGES/OBS_ENV_STATE.

        Returns:
            Action tensor of shape (B, n_action_steps, action_dim).
        """
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)

        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)
        action_dim = self.config.action_feature.shape[0]

        noise = torch.randn(
            batch_size, self.config.horizon, action_dim,
            dtype=dtype, device=device,
        )

        actions = self.generator(noise, global_cond=global_cond)

        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute the RS-IMLE training loss.

        For each ground-truth action trajectory, generates n_samples_per_condition candidate
        trajectories from noise and computes the RS-IMLE loss (nearest valid sample).

        Args:
            batch: Training batch with OBS_STATE, ACTION, action_is_pad, and optionally images.

        Returns:
            Scalar loss tensor.
        """
        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        batch_size = batch[ACTION].shape[0]
        n_samples = self.config.n_samples_per_condition
        action_dim = self.config.action_feature.shape[0]

        global_cond = self._prepare_global_conditioning(batch)

        # Repeat conditioning for all candidate samples: (B, D) -> (B*N, D)
        repeated_cond = global_cond.repeat_interleave(n_samples, dim=0)

        # Sample noise: (B*N, horizon, action_dim)
        noise = torch.randn(
            batch_size * n_samples, self.config.horizon, action_dim,
            dtype=global_cond.dtype, device=global_cond.device,
        )

        # Single forward pass for all candidates: (B*N, horizon, action_dim)
        pred = self.generator(noise, global_cond=repeated_cond)

        # Reshape: (B, N, horizon, action_dim)
        pred = pred.view(batch_size, n_samples, self.config.horizon, action_dim)

        # Compute RS-IMLE loss
        loss = rs_imle_loss(
            real_samples=batch[ACTION],
            fake_samples=pred,
            epsilon=self.config.epsilon,
        )

        # Apply padding mask if configured
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )

        return loss


# ═══════════════════════════════════════════════════════════════════════════════
#  Generator U-Net (Diffusion U-Net minus the timestep embedding)
# ═══════════════════════════════════════════════════════════════════════════════


class IMLEGeneratorUnet1d(nn.Module):
    """1D conditional U-Net generator for IMLE.

    Structurally identical to DiffusionConditionalUnet1d but without the sinusoidal
    diffusion-step encoder. The conditioning dimension is just global_cond_dim
    (not global_cond_dim + diffusion_step_embed_dim).

    Uses the same building blocks (DiffusionConditionalResidualBlock1d, DiffusionConv1dBlock)
    imported from the diffusion policy.
    """

    def __init__(self, config: IMLEConfig, global_cond_dim: int):
        super().__init__()
        self.config = config

        # No diffusion_step_encoder — cond_dim is just global_cond_dim.
        cond_dim = global_cond_dim

        in_out = [(config.action_feature.shape[0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }

        # Encoder (down path)
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Mid blocks
        self.mid_modules = nn.ModuleList(
            [
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
            ]
        )

        # Decoder (up path)
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Final projection
        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.action_feature.shape[0], 1),
        )

    def forward(self, x: Tensor, global_cond: Tensor) -> Tensor:
        """Forward pass through the generator.

        Args:
            x: Noise input of shape (B, T, action_dim).
            global_cond: Global conditioning vector of shape (B, cond_dim).

        Returns:
            Generated action trajectory of shape (B, T, action_dim).
        """
        # (B, T, D) -> (B, D, T) for 1D convolutions
        x = einops.rearrange(x, "b t d -> b d t")

        # No timestep encoding — global_feature is just the conditioning vector
        global_feature = global_cond

        # Encoder with skip connections
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        # Mid blocks
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Decoder with skip connections
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B, D, T) -> (B, T, D)
        x = einops.rearrange(x, "b d t -> b t d")
        return x


# ═══════════════════════════════════════════════════════════════════════════════
#  RS-IMLE Loss
# ═══════════════════════════════════════════════════════════════════════════════


def rs_imle_loss(real_samples: Tensor, fake_samples: Tensor, epsilon: float) -> Tensor:
    """Rejection Sampling IMLE loss.

    For each ground-truth trajectory, finds the nearest generated candidate that is farther
    than epsilon (rejection sampling to prevent mode collapse), and minimizes the distance.

    Args:
        real_samples: Ground-truth action trajectories, shape (B, T, D).
        fake_samples: Generated candidate trajectories, shape (B, N, T, D).
        epsilon: Minimum distance threshold for rejection sampling.

    Returns:
        Scalar loss (mean over valid real samples in the batch).
    """
    batch_size = real_samples.shape[0]

    # Flatten temporal and action dims: (B, T*D) and (B, N, T*D)
    real_flat = real_samples.reshape(batch_size, 1, -1)
    fake_flat = fake_samples.reshape(batch_size, fake_samples.shape[1], -1)

    # Pairwise L2 distances: (B, 1, N) -> squeeze -> (B, N)
    distances = torch.cdist(real_flat, fake_flat).squeeze(1)

    # Reject candidates closer than epsilon (prevents mode collapse)
    valid_mask = (distances > epsilon).float()

    # For invalid candidates, set distance to a large value so they won't be selected as minimum
    large_val = distances.max().detach() + 1.0
    masked_distances = distances + (1.0 - valid_mask) * large_val

    # Find the nearest valid candidate for each real sample
    min_distances, _ = masked_distances.min(dim=1)  # (B,)

    # If all candidates were rejected for a sample, exclude it from the loss
    has_valid = (min_distances < large_val).float()
    loss = (min_distances * has_valid).sum() / has_valid.sum().clamp(min=1.0)

    return loss
