from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.policies.act.configuration_act import ACTConfig
from .flow_on_the_fly import FlowOnTheFly


@PreTrainedConfig.register_subclass("lucky_act")
@dataclass
class LuckyACTConfig(ACTConfig):
    """Configuration for Lucky_ACT – ACT with optical-flow fusion.

    Changes w.r.t ACT:
    • Uses two observation steps (current & previous) so optical-flow can be
      computed on-the-fly.
    • Adds parameters controlling flow backbone and fusion.
    """

    # Add new field for auto-inference
    auto_infer_features: bool = True

    # Override defaults
    n_obs_steps: int = 2  # need previous & current frame

    # Flow-specific params
    enable_flow_fusion: bool = True
    flow_features: List[str] = field(default_factory=list)  # e.g., ["observation.image_flow_cam1"]
    flow_backbone_name: str = "resnet18"
    flow_backbone_pretrained: bool = False

    fusion_n_heads: int = 8
    fusion_n_layers: int = 2
    fusion_dropout: float = 0.1
    flow_feature_scale: float = 1.0  # scale factor for raw flow values

    # Update default normalization mapping – treat flow as visual modality
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # --- flow arguments ---
    # by default, no flow is used
    flow_features: list[str] = field(default_factory=list)

    # settings for fusion of flow and RGB
    enable_flow_fusion: bool = False
    # which layers to fuse flow into
    fusion_layers: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7])
    # whether to use a separate projection for flow
    use_separate_flow_proj: bool = False
    # whether to use a shared backbone for flow and RGB
    use_shared_backbone: bool = False

    # --- neuflow arguments ---
    # these are used only if flow is computed on the fly
    neuflow_repo_id: str = "hf-internal-testing/Optical-Flow-NEF"
    neuflow_version: str = "v5"
    flow_on_the_fly: FlowOnTheFly | None = None
    
    push_to_hub: bool = False

    # ------------------------------------------------------------------
    # Validation overrides
    # ------------------------------------------------------------------
    def validate_features(self) -> None:  # type: ignore[override]
        # When auto-inference is on, we skip the parent validation because features
        # might not be populated yet. The policy's __init__ will handle it later.
        if not self.auto_infer_features:
            super().validate_features()

        if self.enable_flow_fusion and not self.flow_features:
            raise ValueError("Flow features must be specified when enable_flow_fusion=True")
        if self.enable_flow_fusion:
            for key in self.flow_features:
                if not key.startswith("observation.image_flow"):
                    raise ValueError(
                        f"Flow feature '{key}' must start with 'observation.image_flow' to respect naming convention."
                    )

    # ------------------------------------------------------------------
    # Delta indices – ensure dataset loads 2 frames (prev, current)
    # ------------------------------------------------------------------
    @property
    def observation_delta_indices(self):  # noqa: D401
        # Want frames at t-1 and t (current). In loader, index 0 is current, -1 is previous.
        return [-1, 0]

    # action and reward deltas remain same as ACT (chunk_size etc.) 

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------
    def __post_init__(self):  # noqa: D401
        """Custom post-init that is identical to ACTConfig's except it *allows* n_obs_steps=2.

        ACTConfig forbids `n_obs_steps` other than 1. Lucky-ACT needs two
        frames (previous & current) so we reproduce the parent's validation
        logic but simply omit that specific check.
        """

        # Run generic PreTrainedConfig initialisation (device, AMP, …)
        PreTrainedConfig.__post_init__(self)

        # ---- Validation copied from ACTConfig.__post_init__, minus the n_obs_steps check ----
        valid_backbone_prefixes = ("resnet", "convnext")
        if not self.vision_backbone.startswith(valid_backbone_prefixes):
            raise ValueError(
                f"`vision_backbone` must start with one of {valid_backbone_prefixes}. Got {self.vision_backbone}."
            )

        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )

        # Skip the `n_obs_steps != 1` restriction from ACTConfig.

        # Finally, run feature validation (may raise) defined above.
        self.validate_features() 

    # ------------------------------------------------------------------
    # Compatibility helpers — expose attributes expected by the generic
    # `lerobot.policies.factory.make_policy` implementation.  The factory
    # (currently) assumes every policy config instance has nested
    # attributes `policy` (a self-reference) and `dataset.repo_id`.  While
    # this holds for the top-level training config, it is *not* true for
    # the per-policy `PreTrainedConfig` classes.  To avoid touching the
    # shared factory logic we provide lightweight fallbacks here so that
    # Lucky-ACT remains plug-and-play with the existing pipeline.
    # ------------------------------------------------------------------

    class _DatasetProxy:  # pylint: disable=too-few-public-methods
        """Minimal stand-in exposing only the `repo_id` attribute."""

        def __init__(self, repo_id: str | None = None):
            self.repo_id = repo_id

    # Store a single proxy instance so that attribute access is stable
    _dataset_proxy = _DatasetProxy()

    # NOTE: Using properties rather than assigning new dataclass fields
    # keeps the public dataclass schema unchanged (important for
    # serialization & CLI parsing) while still satisfying the factory.
    @property  # type: ignore[override]
    def policy(self) -> "LuckyACTConfig":  # noqa: D401
        """Self-reference expected by `make_policy()` logic."""
        return self

    @property  # type: ignore[override]
    def dataset(self) -> "LuckyACTConfig._DatasetProxy":  # noqa: D401
        """Return a dummy object carrying `.repo_id`.

        The actual *value* of `repo_id` is filled opportunistically by
        `LuckyACTPolicy.__init__` when a non-null `dataset_path` becomes
        available.  This is sufficient for the factory, which only needs
        the attribute to exist during kwargs construction.
        """
        return self._dataset_proxy 