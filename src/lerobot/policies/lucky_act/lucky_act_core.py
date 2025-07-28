from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn, Tensor
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
import einops

from lerobot.policies.act.modeling_act import ACT, ACTSinusoidalPositionEmbedding2d
from lerobot.policies.adaptive_layer_norm import AdaLayerNorm
from .configuration_lucky_act import LuckyACTConfig
from lerobot.constants import ACTION


class TaskConditionedTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with AdaLayerNorm for task conditioning."""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        task_embedding_dim: Optional[int] = None,
        use_adaln: bool = True,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        if use_adaln and task_embedding_dim is not None:
            self.norm1 = AdaLayerNorm(d_model, task_embedding_dim)
            self.norm2 = AdaLayerNorm(d_model, task_embedding_dim)
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation
        self.activation = getattr(torch.nn.functional, activation)
        
        self.norm_first = norm_first
        self.use_adaln = use_adaln and task_embedding_dim is not None
    
    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        task_embedding: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self._norm_forward(self.norm1, x, task_embedding), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self._norm_forward(self.norm2, x, task_embedding))
        else:
            x = self._norm_forward(self.norm1, x + self._sa_block(x, src_mask, src_key_padding_mask), task_embedding)
            x = self._norm_forward(self.norm2, x + self._ff_block(x), task_embedding)
        return x

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    
    def _norm_forward(self, norm_layer, x: Tensor, task_embedding: Optional[Tensor] = None) -> Tensor:
        """Apply normalization with optional task conditioning."""
        if self.use_adaln and task_embedding is not None:
            return norm_layer(x, task_embedding)
        else:
            return norm_layer(x)


class TaskConditionedTransformerEncoder(nn.Module):
    """Transformer encoder with task conditioning support."""
    
    def __init__(self, encoder_layer: TaskConditionedTransformerEncoderLayer, num_layers: int, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        task_embedding: Optional[Tensor] = None,
    ) -> Tensor:
        output = src
        
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                task_embedding=task_embedding,
            )
        
        if self.norm is not None:
            output = self.norm(output)
            
        return output


class LuckyACTCore(ACT):
    """ACT variant with dedicated flow backbones (2-channel input) and task conditioning.

    This class inherits most of ACT's implementation and overrides the
    constructor and forward method to support:
    1. Flow map encoding via separate CNNs
    2. Task conditioning via task tokens and AdaLayerNorm
    """

    def __init__(self, config: LuckyACTConfig):
        self._flow_features = config.flow_features if config.enable_flow_fusion else []
        self._use_task_conditioning = config.use_task_conditioning
        self._use_task_token = config.use_task_token if config.use_task_conditioning else False
        self._use_adaln = config.use_adaln_task_context if config.use_task_conditioning else False
        self._task_token_position = config.task_token_position
        self._task_embedding_dim = config.task_embedding_dim
        
        # Initialize parent ACT model
        super().__init__(config)
        
        # Replace the encoder with task-conditioned version if needed
        if self._use_adaln:
            self.encoder = self._create_task_conditioned_encoder(config)
        
        if self._flow_features:
            self._init_flow_backbones(config)
        
        # Task token projection
        if self._use_task_token:
            self.task_token_proj = nn.Linear(self._task_embedding_dim, config.dim_model)
            # Learnable positional embedding for task token
            self.task_token_pos_embed = nn.Parameter(torch.randn(1, 1, config.dim_model))
    
    def _create_task_conditioned_encoder(self, config: LuckyACTConfig):
        """Create a task-conditioned encoder replacing the standard ACT encoder."""
        encoder_layer = TaskConditionedTransformerEncoderLayer(
            d_model=config.dim_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.feedforward_activation,
            batch_first=False,  # ACT expects (seq, batch, dim)
            norm_first=config.pre_norm,
            task_embedding_dim=self._task_embedding_dim,
            use_adaln=True,
        )
        norm = nn.LayerNorm(config.dim_model) if config.pre_norm else None
        return TaskConditionedTransformerEncoder(encoder_layer, num_layers=config.n_encoder_layers, norm=norm)

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

        self.flow_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Fusion transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim_model,
            nhead=config.fusion_n_heads,
            dim_feedforward=config.dim_model * 4,
            dropout=config.fusion_dropout,
            batch_first=False,
        )
        self.fusion_module = nn.TransformerEncoder(encoder_layer, num_layers=config.fusion_n_layers)
    
    def forward(self, batch: dict[str, Tensor]):  # noqa: C901
        """Forward pass with task conditioning and optical flow.

        This method replicates the logic from ACT.forward and injects:
        - Task tokens into the transformer input sequence.
        - Task embeddings into AdaLayerNorm layers.
        - Optical flow features fused with RGB features.
        """
        from lerobot.constants import ACTION

        is_training = self.training
        
        # 1. Get latent variable for VAE
        if self.config.use_vae and is_training:
            latent_dist = self._compute_latent_dist(batch)
            latent_sample = latent_dist.rsample()
        else:
            batch_size = next(v for v in batch.values() if isinstance(v, Tensor) and v.ndim > 1).shape[0]
            latent_sample = torch.randn(batch_size, self.config.latent_dim, device=self.device)

        # 2. Prepare transformer encoder inputs
        encoder_in_tokens = []
        encoder_in_pos = []

        # Task token
        task_embedding = batch.get("task_embedding")
        if self._use_task_token and task_embedding is not None:
            task_token = self.task_token_proj(task_embedding).unsqueeze(0)
            task_pos = self.task_token_pos_embed.expand(1, task_token.shape[1], -1)
            if self._task_token_position == "prepend":
                encoder_in_tokens.append(task_token)
                encoder_in_pos.append(task_pos)

        # Latent token
        encoder_in_tokens.append(self.encoder_latent_input_proj(latent_sample).unsqueeze(0))
        encoder_in_pos.append(self.encoder_1d_feature_pos_embed.weight[0].unsqueeze(0).unsqueeze(1).expand(-1, latent_sample.shape[0], -1))

        # Robot state token
        if self.config.robot_state_feature:
            robot_state = batch["observation.state"][:, -1] if batch["observation.state"].ndim == 3 else batch["observation.state"]
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(robot_state).unsqueeze(0))
            encoder_in_pos.append(self.encoder_1d_feature_pos_embed.weight[1].unsqueeze(0).unsqueeze(1).expand(-1, robot_state.shape[0], -1))

        # RGB and Flow camera tokens
        rgb_tokens, rgb_pos = self._encode_rgb_images(batch)
        if self._flow_features:
            flow_tokens, flow_pos = self._encode_flow_images(batch)
            fused_tokens = self.fusion_module(torch.cat([rgb_tokens, flow_tokens], dim=0) + torch.cat([rgb_pos, flow_pos], dim=0))
            encoder_in_tokens.append(fused_tokens[:rgb_tokens.shape[0]])
            encoder_in_pos.append(rgb_pos)
        elif rgb_tokens is not None:
            encoder_in_tokens.append(rgb_tokens)
            encoder_in_pos.append(rgb_pos)

        # Task token (if appending)
        if self._use_task_token and task_embedding is not None and self._task_token_position == "append":
            encoder_in_tokens.append(task_token)
            encoder_in_pos.append(task_pos)

        # 3. Stack tokens and run encoder
        encoder_in_tokens = torch.cat(encoder_in_tokens, dim=0)
        encoder_in_pos = torch.cat(encoder_in_pos, dim=0)
        
        encoder_kwargs = {"task_embedding": task_embedding} if self._use_adaln else {}
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos, **encoder_kwargs)

        # 4. Run decoder
        decoder_in = self.decoder_pos_embed.weight.unsqueeze(1).repeat(1, encoder_out.shape[1], 1)
        decoder_out = self.decoder(decoder_in, encoder_out)
        
        actions_hat = self.action_head(decoder_out.transpose(0, 1))

        return actions_hat, (latent_dist if self.config.use_vae and is_training else None)

    def _encode_rgb_images(self, batch):
        if not self.config.image_features:
            return None, None
        
        all_cam_features, all_cam_pos = [], []
        for img_key, backbone in zip(self.config.image_features, self.backbones, strict=True):
            img = batch[img_key][:, -1] if batch[img_key].ndim == 5 else batch[img_key]
            features = backbone(img)["feature_map"]
            pos_embed = self.encoder_cam_feat_pos_embed(features)
            
            all_cam_features.append(einops.rearrange(self.encoder_img_feat_input_proj(features), "b c h w -> (h w) b c"))
            all_cam_pos.append(einops.rearrange(pos_embed, "b c h w -> (h w) b c"))
            
        return torch.cat(all_cam_features, dim=0), torch.cat(all_cam_pos, dim=0)

    def _encode_flow_images(self, batch):
        all_flow_features, all_flow_pos = [], []
        for flow_key, backbone in zip(self.flow_features, self.flow_backbones, strict=True):
            flow = batch[flow_key][:, -1] if batch[flow_key].ndim == 5 else batch[flow_key]
            features = backbone(flow)["feature_map"]
            pos_embed = self.flow_pos_embed(features)
            
            all_flow_features.append(einops.rearrange(self.flow_feat_proj(features), "b c h w -> (h w) b c"))
            all_flow_pos.append(einops.rearrange(pos_embed, "b c h w -> (h w) b c"))
            
        return torch.cat(all_flow_features, dim=0), torch.cat(all_flow_pos, dim=0)

    def _compute_latent_dist(self, batch: dict[str, Tensor]):
        """Return (mu, log_sigma_x2) for variational objective.
 
        This mirrors the logic from `ACT.forward` so that Lucky-ACT keeps
        identical training behaviour when `use_vae=True`.
        """
        if not self.config.use_vae:
            return None
 
        from lerobot.constants import ACTION
 
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
 
        return torch.distributions.Normal(mu, (0.5 * log_sigma_x2).exp()) 