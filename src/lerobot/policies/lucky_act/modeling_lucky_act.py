from __future__ import annotations

from typing import Dict

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from lerobot.policies.act.modeling_act import ACTPolicy
from .flow_on_the_fly import FlowOnTheFly
from .configuration_lucky_act import LuckyACTConfig


class LuckyACTPolicy(ACTPolicy):
    """Lucky_ACT policy – ACT enhanced with optical-flow fusion.

    For now, this class focuses on *injecting* optical-flow maps into the batch
    on-the-fly.  Architectural changes for fusing the flow tokens are left for a
    subsequent implementation; the RGB-only ACT backbone continues to operate
    (flow features are available should the model use them).
    """

    config_class = LuckyACTConfig
    name = "lucky_act"
    auto_infer_features = False

    def __init__(
        self,
        config: LuckyACTConfig,
        dataset_stats: Dict | None = None,
        dataset_path: str | None = None,
    ):
        """Construct a Lucky_ACT policy.

        The shared factory passes `dataset_path` only when it can obtain
        it from `cfg.dataset.repo_id`.  For compatibility with the
        existing training pipeline we make `dataset_path` optional:

        • If *both* `auto_infer_features` is **False** *or* dataset_path
          is provided, behaviour is identical to the previous
          implementation.
        • If `auto_infer_features` is **True** *and* `dataset_path` is
          *None* (e.g. because the factory could not locate
          `dataset.repo_id`), we disable auto-inference instead of
          raising.  This keeps training functional when the dataset
          metadata is already known (the factory has just filled
          `config.input_features` / `output_features`).
        """

        # Fill the proxy so downstream logic (incl. repr) sees the path.
        if dataset_path is not None:
            try:
                # Cache on the shared proxy created by the config helper.
                config._dataset_proxy.repo_id = dataset_path  # type: ignore[attr-defined]
            except AttributeError:
                pass  # Proxy missing – not critical.

        # ------------------------------------------------------------------
        # Automatic feature inference
        # ------------------------------------------------------------------
        if config.auto_infer_features and dataset_path is None:
            # Fall back to *manual* features – disable auto inference.
            config.auto_infer_features = False

        if config.auto_infer_features:
            meta = LeRobotDatasetMetadata(repo_id=dataset_path)

            # Infer image features (non-flow)
            image_keys = [
                k for k, v in meta.features.items() if v["dtype"] in ("video", "image") and "flow" not in k
            ]
            if image_keys:
                # Insert or update entries in `input_features` to reflect
                # newly discovered camera keys so that downstream property
                # `config.image_features` picks them up.
                from lerobot.configs.types import PolicyFeature, FeatureType  # local import to avoid circular
                for k in image_keys:
                    if k not in config.input_features:
                        # Metadata guarantees shape field
                        img_shape = meta.features[k]["shape"]
                        config.input_features[k] = PolicyFeature(type=FeatureType.VISUAL, shape=img_shape)

            # Infer flow features
            flow_keys = [k for k, v in meta.features.items() if "flow" in k]
            if flow_keys:
                config.flow_features = flow_keys
            else:
                # Disable flow fusion if no flow keys are found
                config.enable_flow_fusion = False

            # Infer environment state
            if "observation.environment_state" in meta.features:
                from lerobot.configs.types import PolicyFeature, FeatureType
                if "observation.environment_state" not in config.input_features:
                    cfg_shape = meta.features["observation.environment_state"]["shape"]
                    config.input_features["observation.environment_state"] = PolicyFeature(type=FeatureType.ENV, shape=cfg_shape)

            # Now that features are populated, run validation.
            config.validate_features()

        # Build flow transform if enabled
        self._flow_transform = None
        if config.enable_flow_fusion and config.flow_features:
            # ------------------------------------------------------------------
            # Build camera ➜ flow mapping robustly.
            # ------------------------------------------------------------------

            def _norm(s: str) -> str:
                """Normalize key for fuzzy comparison (lowercase, strip separators)."""
                return s.lower().replace(".", "").replace("_", "")

            image_keys = list(config.image_features)

            cam_to_flow: dict[str, str] = {}
            for idx, flow_key in enumerate(config.flow_features):
                best_cam = None

                # 1️⃣ 1-to-1 positional pairing when counts match.
                if idx < len(image_keys):
                    best_cam = image_keys[idx]

                # 2️⃣ Try fuzzy matching – remove "_flow" then compare normalized strings.
                if best_cam is None:
                    base = flow_key.replace("_flow", "")
                    base_n = _norm(base)
                    for cam_key in image_keys:
                        if _norm(cam_key).endswith(base_n) or base_n.endswith(_norm(cam_key)):
                            best_cam = cam_key
                            break

                # 3️⃣ Fall back to heuristic replacement (previous behaviour).
                if best_cam is None:
                    if "_flow" not in flow_key:
                        raise ValueError(
                            f"Flow feature '{flow_key}' does not contain '_flow' to map back to camera key."
                        )
                    best_cam = flow_key.replace("_flow", "")

                cam_to_flow[best_cam] = flow_key

            self._flow_transform = FlowOnTheFly(cam_to_flow)

        # Temporarily call ACTPolicy init but suppress model creation, then replace.
        super().__init__(config, dataset_stats=dataset_stats)

        # Replace ACT core with LuckyACTCore
        from .lucky_act_core import LuckyACTCore

        self.model = LuckyACTCore(config)

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------
    def _maybe_compute_flow(self, batch):
        # Aggregate per-camera images into a list consumed by ACT/Lucky-ACT cores.
        if self.config.image_features:
            cam_images = [batch[key] for key in self.config.image_features]
            batch["observation.images"] = cam_images

            # Provide alias keys with singular "image" for FlowOnTheFly & compatibility.
            for key in self.config.image_features:
                if ".images." in key:
                    # 1. Dot-notation alias (images ➜ image.)
                    alias_dot = key.replace(".images.", ".image.")
                    if alias_dot not in batch:
                        batch[alias_dot] = batch[key]

                    # 2. Underscore alias (images.cam1 ➜ image_cam1)
                    #    Required by FlowOnTheFly because flow keys typically
                    #    follow the underscore convention.
                    parts = key.split(".images.")
                    if len(parts) == 2:
                        alias_us = f"{parts[0]}.image_{parts[1]}"
                        if alias_us not in batch:
                            batch[alias_us] = batch[key]

        if self._flow_transform is not None:
            # Ensure all camera keys required by FlowOnTheFly exist – create fallbacks if necessary.
            for cam_key in self._flow_transform.camera_to_flow_key.keys():
                if cam_key in batch:
                    continue

                # Variant 1: change 'image_' ➜ 'images.'
                if cam_key.startswith("observation.image_"):
                    cand = cam_key.replace("observation.image_", "observation.images.")
                    if cand in batch:
                        batch[cam_key] = batch[cand]
                        continue

                # Variant 2: dot vs underscore after 'image'
                if ".image_" in cam_key:
                    cand = cam_key.replace(".image_", ".image.")
                    if cand in batch:
                        batch[cam_key] = batch[cand]
                        continue

                # Variant 3: plural 'images' to singular 'image' switch
                if cam_key.startswith("observation.image") and cam_key.count("_") >= 1:
                    # attempt to rebuild plural form
                    base, rest = cam_key.split(".image", 1)
                    cand = f"{base}.images{rest.replace('_', '.')}"
                    if cand in batch:
                        batch[cam_key] = batch[cand]

            batch = self._flow_transform(batch)
        return batch

    # ------------------------------------------------------------------
    # Overrides to inject flow computation before ACT logic
    # ------------------------------------------------------------------
    @torch.no_grad()
    def select_action(self, batch):  # type: ignore[override]
        batch = self._maybe_compute_flow(batch)
        return super().select_action(batch)

    @torch.no_grad()
    def predict_action_chunk(self, batch):  # type: ignore[override]
        batch = self._maybe_compute_flow(batch)
        return super().predict_action_chunk(batch)

    def forward(self, batch):  # type: ignore[override]
        batch = self._maybe_compute_flow(batch)
        return super().forward(batch) 