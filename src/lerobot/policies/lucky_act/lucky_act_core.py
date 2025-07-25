from __future__ import annotations

from typing import List

import torch
from torch import nn, Tensor
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
import einops

from lerobot.policies.act.modeling_act import ACT
from .configuration_lucky_act import LuckyACTConfig


class LuckyACTCore(ACT):
    """ACT variant with dedicated flow backbones (2-channel input).

    This class inherits most of ACT’s implementation and only overrides the
    constructor and a *small* portion of the `forward` method dealing with
    visual feature extraction so that flow maps are encoded via separate light
    CNNs and concatenated with RGB camera tokens.
    """

    def __init__(self, config: LuckyACTConfig):
        self._flow_features = config.flow_features if config.enable_flow_fusion else []
        super().__init__(config)

        if self._flow_features:
            self._init_flow_backbones(config)

    # ------------------------------------------------------------------
    # Override parts dealing with backbones
    # ------------------------------------------------------------------
    def _init_flow_backbones(self, config: LuckyACTConfig):
        """Create one backbone per flow feature (2-channel conv stem)."""
        def create_flow_backbone():
            backbone = getattr(torchvision.models, config.flow_backbone_name)(
                weights=None if not config.flow_backbone_pretrained else "DEFAULT"
            )
            # Patch first conv to accept 2-channel input
            if hasattr(backbone, "conv1"):
                in_ch = 2
                orig = backbone.conv1
                backbone.conv1 = nn.Conv2d(
                    in_ch,
                    orig.out_channels,
                    kernel_size=orig.kernel_size,
                    stride=orig.stride,
                    padding=orig.padding,
                    bias=False,
                )
            # Remove final classifier/fc
            if hasattr(backbone, "fc"):
                backbone.fc = nn.Identity()
            return IntermediateLayerGetter(backbone, return_layers={"layer4": "feature_map"})

        self.flow_backbones = nn.ModuleList([create_flow_backbone() for _ in self._flow_features])

        # Determine backbone output channels via dummy
        with torch.no_grad():
            dummy = torch.zeros(1, 2, 64, 64)  # H,W arbitrary
            channels = self.flow_backbones[0](dummy)["feature_map"].shape[1]
        self.flow_feat_proj = nn.Conv2d(channels, config.dim_model, kernel_size=1)

        # Positional embedding for flow features
        from lerobot.policies.act.modeling_act import ACTSinusoidalPositionEmbedding2d

        self.flow_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Fusion transformer encoder (cross/self attention over combined tokens)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim_model,
            nhead=config.fusion_n_heads,
            dim_feedforward=config.dim_model * 4,
            dropout=config.fusion_dropout,
            batch_first=False,
        )
        self.fusion_module = nn.TransformerEncoder(encoder_layer, num_layers=config.fusion_n_layers)

    def _init_image_and_flow_backbones(self, config: LuckyACTConfig):
        """Called from overridden __init__ to set up backbones."""
        # original image backbones already built in ACT.__init__, so call parent then add flows
        pass  # placeholder

    # We intercept __init__ after super().__init__ to add flow backbones.
    def __post_init_backbones(self, config: LuckyACTConfig):
        if self._flow_features:
            self._init_flow_backbones(config)

    # Override forward to encode flows
    def forward(self, batch: dict[str, Tensor]):  # noqa: C901 (complexity inherited)
        """Forward with added flow token extraction.

        We replicate the parent ACT.forward logic but insert an extra block that
        encodes flow maps using the dedicated flow backbones and appends the
        resulting tokens/pos-embeddings to the encoder input.
        """
        from lerobot.constants import ACTION  # lazy import to avoid circular

        # ------------------------------------------------------------------
        # 0. Preparations identical to ACT.forward up to token creation
        # ------------------------------------------------------------------
        if self.config.use_vae and self.training:
            assert ACTION in batch, "actions must be provided when using VAE objective."

        device = next(self.parameters()).device

        # --------------------------------------------------------------
        # Determine batch size robustly (dataset may store images under
        # separate keys while ACT aggregates them under
        # `observation.images`).
        # --------------------------------------------------------------

        if "observation.images" in batch:
            # batch["observation.images"] is a list of (B,C,H,W) tensors – pick first camera
            batch_size = batch["observation.images"][0].shape[0]
        elif self.config.image_features:
            first_img_key = next(iter(self.config.image_features))
            batch_size = batch[first_img_key].shape[0]
        elif self.config.env_state_feature:
            batch_size = batch[self.config.env_state_feature].shape[0]
        else:
            batch_size = batch[self.config.robot_state_feature].shape[0]

        encoder_in_tokens = []  # list of (seq, B, D) tensors
        encoder_in_pos_embed = []

        #### 1. Latent token (always present)
        latent = torch.zeros(1, batch_size, self.config.dim_model, device=device)
        encoder_in_tokens.append(latent)
        encoder_in_pos_embed.append(torch.zeros_like(latent))  # positional enc 0

        #### 2. Robot & env state tokens (same as ACT)
        if self.config.robot_state_feature:
            robot_state = batch["observation.state"]
            if robot_state.ndim == 3:  # (B, T, D) where T==n_obs_steps
                robot_state = robot_state[:, -1]  # keep current frame
            robot_state_tok = self.encoder_robot_state_input_proj(robot_state).unsqueeze(0)
            encoder_in_tokens.append(robot_state_tok)
            encoder_in_pos_embed.append(torch.zeros_like(robot_state_tok))

        if self.config.env_state_feature:
            env_state = batch["observation.environment_state"]
            if env_state.ndim == 3:
                env_state = env_state[:, -1]
            env_state_tok = self.encoder_env_state_input_proj(env_state).unsqueeze(0)
            encoder_in_tokens.append(env_state_tok)
            encoder_in_pos_embed.append(torch.zeros_like(env_state_tok))

        #### 3. RGB camera tokens – reuse ACT helper
        if self.config.image_features:
            all_cam_features, all_cam_pos = self._encode_rgb_images(batch)
            rgb_tokens = all_cam_features
            rgb_pos = all_cam_pos
            # Safety: sometimes dataloader collates tensors into batch dimension inside channel; ensure B dim matches
            if rgb_pos.shape[1] == 1 and rgb_pos.shape[1] != batch_size:
                rgb_pos = rgb_pos.expand(-1, batch_size, -1)

        #### 4. Flow camera tokens (new)
        if self._flow_features:
            flow_tok_list = []
            flow_pos_list = []
            for flow_key, backbone in zip(self._flow_features, self.flow_backbones, strict=False):
                flow = batch[flow_key]
                flow = flow.to(device)
                if flow.ndim == 3:
                    flow = flow.unsqueeze(0).expand(batch_size, *flow.shape)
                # shape (B,2,H,W)
                feat_map = backbone(flow)["feature_map"]
                pos = self.flow_pos_embed(feat_map).to(dtype=feat_map.dtype)
                feat_map = self.flow_feat_proj(feat_map)

                flow_tokens = einops.rearrange(feat_map, "b c h w -> (h w) b c")
                flow_pos = einops.rearrange(pos, "b c h w -> (h w) b c")

                flow_tok_list.append(flow_tokens)
                flow_pos_list.append(flow_pos)

            flow_tokens = torch.cat(flow_tok_list, dim=0)
            flow_pos = torch.cat(flow_pos_list, dim=0)

            # Ensure batch dim matches
            if flow_pos.shape[1] == 1 and flow_pos.shape[1] != batch_size:
                flow_pos = flow_pos.expand(-1, batch_size, -1)

            # Cross/self attention fusion
            combined_tokens = torch.cat([rgb_tokens, flow_tokens], dim=0)
            combined_pos = torch.cat([rgb_pos, flow_pos], dim=0)
            combined_tokens = combined_tokens + combined_pos
            fused_tokens = self.fusion_module(combined_tokens)
            fused_rgb_tokens = fused_tokens[: rgb_tokens.shape[0]]  # discard flow-only tokens

            encoder_in_tokens.append(fused_rgb_tokens)
            encoder_in_pos_embed.append(rgb_pos)  # positional embedding unchanged
        else:
            encoder_in_tokens.append(rgb_tokens)
            encoder_in_pos_embed.append(rgb_pos)

        # ------------------------------------------------------------------
        # 5. Stack tokens & positions, run through encoder/decoder (same as ACT)
        # ------------------------------------------------------------------
        encoder_in_tokens = torch.cat(encoder_in_tokens, dim=0)
        encoder_in_pos_embed = torch.cat(encoder_in_pos_embed, dim=0)

        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_tokens.dtype,
            device=encoder_in_tokens.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        actions_hat = self.action_head(decoder_out.transpose(0, 1))  # (B, chunk, action_dim)

        if self.config.use_vae and self.training:
            # KLD parts from parent – reuse unchanged
            mu_hat, log_sigma_x2_hat = self._compute_latent_params(batch)
        else:
            mu_hat = log_sigma_x2_hat = None

        return actions_hat, (mu_hat, log_sigma_x2_hat)

    # ------------------------------------------------------------------
    # Helper: reuse original RGB camera encode logic via small wrapper
    # ------------------------------------------------------------------
    def _encode_rgb_images(self, batch):
        all_cam_features = []
        all_cam_pos_embeds = []
        if not self.config.image_features:
            return torch.empty(0), torch.empty(0)

        for img, backbone in zip(batch["observation.images"], self.backbones, strict=False):
            # img can be (B,C,H,W) or (B,T,C,H,W); keep current frame if temporal dim
            if img.ndim == 5 and img.shape[1] == 2:
                img = img[:, -1]  # select current frame
            cam_features = backbone(img)["feature_map"]
            cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
            cam_features = self.encoder_img_feat_input_proj(cam_features)

            cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
            cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

            all_cam_features.append(cam_features)
            all_cam_pos_embeds.append(cam_pos_embed)

        return torch.cat(all_cam_features, dim=0), torch.cat(all_cam_pos_embeds, dim=0)

    # ------------------------------------------------------------------
    # Latent parameter computation (copied & adapted from ACT.forward)
    # ------------------------------------------------------------------

    def _compute_latent_params(self, batch: dict[str, Tensor]):
        """Return (mu, log_sigma_x2) for variational objective.

        This mirrors the logic from `ACT.forward` so that Lucky-ACT keeps
        identical training behaviour when `use_vae=True`.
        """
        if not self.config.use_vae:
            return None, None

        from lerobot.constants import ACTION  # local import to avoid cycles

        assert ACTION in batch, "actions must be provided when using the variational objective."

        batch_size = batch[ACTION].shape[0]

        # Build VAE-encoder input: [CLS, (robot_state), action_seq]
        cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size)

        input_tokens = [cls_embed]

        if self.config.robot_state_feature:
            robot_state = batch["observation.state"]
            if robot_state.ndim == 3:
                robot_state = robot_state[:, -1]  # current frame
            robot_state_tok = self.vae_encoder_robot_state_input_proj(robot_state).unsqueeze(1)
            input_tokens.append(robot_state_tok)

        action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B,S,D)
        input_tokens.append(action_embed)

        vae_input = torch.cat(input_tokens, dim=1)  # (B, seq, D)

        # Positional embedding (buffer already registered)
        pos_embed = self.vae_encoder_pos_enc[:, : vae_input.shape[1]].detach()

        # Key padding mask: False for cls & robot_state, batch['action_is_pad'] for actions
        if "action_is_pad" in batch:
            pad_mask = batch["action_is_pad"]
        else:
            pad_mask = torch.zeros(batch[ACTION].shape[:2], dtype=torch.bool, device=vae_input.device)

        extra_tokens = 1 + (1 if self.config.robot_state_feature else 0)
        cls_joint_pad = torch.zeros(batch_size, extra_tokens, dtype=torch.bool, device=vae_input.device)
        key_padding_mask = torch.cat([cls_joint_pad, pad_mask], dim=1)

        # Forward through VAE encoder (full precision)
        with torch.cuda.amp.autocast(enabled=False):
            cls_token_out = self.vae_encoder(
                vae_input.permute(1, 0, 2).float(),
                pos_embed=pos_embed.permute(1, 0, 2).float(),
                key_padding_mask=key_padding_mask,
            )[0]

            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)

        latent_pdf_params = latent_pdf_params.to(vae_input.dtype)
        mu = latent_pdf_params[:, : self.config.latent_dim]
        log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

        return mu, log_sigma_x2 