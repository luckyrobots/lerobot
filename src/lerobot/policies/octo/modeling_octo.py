from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.policies.octo.configuration_octo import OctoConfig
from lerobot.policies.pretrained import PreTrainedPolicy


def _ensure_octopt_importable() -> None:
    """Best-effort import path setup for the vendored `octo-pytorch-src`."""
    try:
        import octo  # noqa: F401

        return
    except Exception:
        pass

    # Add repo_root/octo-pytorch-src to sys.path
    here = Path(__file__).resolve()
    # .../lerobot/src/lerobot/policies/octo/modeling_octo.py -> repo root is parents[5]
    repo_root = here.parents[5]
    candidate = repo_root / "octo-pytorch-src"
    if candidate.exists():
        sys.path.insert(0, str(candidate))


def _freeze_keys(mode: str) -> list[str] | None:
    # Mirrors `piper_sdk/vla_shim/finetune_octo_pt_piper.py` (which mirrors upstream Octo recipe).
    if mode == "full":
        return None
    if mode == "head_only":
        return ["octo_transformer.*"]
    if mode == "head_mlp_only":
        return [
            "octo_transformer.*",
            "heads_*.map_head.probe",
            "heads_*.map_head.MultiHeadDotProductAttention_0.*",
        ]
    raise ValueError(f"Unknown finetune_mode {mode!r}")


class OctoPolicy(PreTrainedPolicy):
    config_class = OctoConfig
    name = "octo"

    def __init__(
        self,
        config: OctoConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.dataset_stats = dataset_stats or {}

        _ensure_octopt_importable()
        try:
            from octo.model.octo_model_pt import OctoModelPt
            from octo.utils.train_utils_pt import freeze_weights_pt
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "OctoPt dependencies are not importable. Ensure `octo-pytorch-src` is available "
                "and required deps are installed (see `piper_sdk/vla_shim/requirements-octo.txt`)."
            ) from e

        # Load OctoPt from official JAX checkpoint (text-conditioning initialized correctly).
        loaded: dict[str, Any] = OctoModelPt.load_pretrained_from_jax(
            str(config.base_checkpoint),
            skip_keys_regex=".*hf_model",
        )
        self.octo: OctoModelPt = loaded["octo_model"]

        # Freeze according to finetune_mode (and always keep HF text model frozen).
        keys = _freeze_keys(str(config.finetune_mode))
        freeze = [] if keys is None else list(keys)
        freeze = freeze + ["*hf_model*"]
        freeze_weights_pt(self.octo.module, freeze)

    def reset(self):
        return

    def get_optim_params(self) -> dict:
        return [{"params": [p for p in self.parameters() if p.requires_grad]}]

    def _as_cpu_tensor(self, x: Any, *, dtype: torch.dtype) -> Tensor:
        """Convert numpy / torch / list scalars to a CPU torch.Tensor."""
        if isinstance(x, torch.Tensor):
            return x.detach().to(device="cpu", dtype=dtype)
        if isinstance(x, np.ndarray):
            return torch.as_tensor(x, device="cpu", dtype=dtype)
        return torch.as_tensor(x, device="cpu", dtype=dtype)

    def _get_action_unnormalization_statistics(self) -> dict[str, Tensor] | None:
        """Build OctoPt unnormalization statistics from LeRobot dataset stats.

        OctoPt's `sample_actions` unnormalizes on CPU. We therefore keep these tensors on CPU.
        """
        stats = self.dataset_stats.get("action", None)
        if not isinstance(stats, dict):
            return None
        mean = stats.get("mean", None)
        std = stats.get("std", None)
        if mean is None or std is None:
            return None
        mean_t = self._as_cpu_tensor(mean, dtype=torch.float32)
        std_t = self._as_cpu_tensor(std, dtype=torch.float32)
        mask = stats.get("mask", None)
        if mask is None:
            mask_t = torch.ones_like(mean_t, dtype=torch.bool)
        else:
            mask_t = self._as_cpu_tensor(mask, dtype=torch.bool)
        return {"mean": mean_t, "std": std_t, "mask": mask_t}

    def _normalize_actions(self, actions: Tensor) -> Tensor:
        """Normalize raw actions to Octo's normalized space using dataset stats."""
        unnorm = self._get_action_unnormalization_statistics()
        if unnorm is None:
            raise ValueError(
                "Missing dataset action statistics for normalization. Ensure dataset metadata stats include "
                "`stats['action']['mean']` and `stats['action']['std']`."
            )
        mean = unnorm["mean"].to(device=actions.device)
        std = unnorm["std"].to(device=actions.device).clamp(min=1e-6)
        mask = unnorm["mask"].to(device=actions.device)
        # Broadcast over leading dims (e.g. B,H,A).
        while mean.ndim < actions.ndim:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
            mask = mask.unsqueeze(0)
        return torch.where(mask, (actions - mean) / std, actions)

    def _build_observations(self, batch: dict[str, Any]) -> tuple[dict[str, Tensor], list[str], Tensor, dict[str, Tensor]]:
        cfg = self.config
        x_primary: Tensor = batch[cfg.primary_image_key]
        x_wrist: Tensor = batch[cfg.wrist_image_key]

        # LeRobotDataset returns:
        # - single frame: (B,C,H,W)
        # - temporal stack via delta_timestamps: (B,T,C,H,W)
        def _ensure_btc_hw(x: Tensor, *, key: str) -> Tensor:
            if x.ndim == 4:
                return x[:, None, ...]
            if x.ndim == 5:
                return x
            raise ValueError(f"Expected {key} shape (B,C,H,W) or (B,T,C,H,W), got {tuple(x.shape)}")

        x_primary = _ensure_btc_hw(x_primary, key=cfg.primary_image_key)
        x_wrist = _ensure_btc_hw(x_wrist, key=cfg.wrist_image_key)

        if x_primary.shape[:2] != x_wrist.shape[:2]:
            raise ValueError(
                "Primary and wrist images must share (B,T) dimensions, got "
                f"{tuple(x_primary.shape[:2])} vs {tuple(x_wrist.shape[:2])}."
            )

        B, T = int(x_primary.shape[0]), int(x_primary.shape[1])

        # Resize per timestep (operate in float, clamp to [0,1]).
        def _resize(x: Tensor, *, size: tuple[int, int]) -> Tensor:
            # Datasets may provide either float32 in [0,1] (images stored as PIL) or uint8 in [0,255] (video decode).
            x = x.to(dtype=torch.float32)
            if not torch.is_floating_point(x) or (x.numel() > 0 and float(x.max().item()) > 1.5):
                x = x / 255.0
            x = x.clamp(0, 1)
            x = x.reshape(B * T, *x.shape[2:])  # (B*T,C,H,W)
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            return x.reshape(B, T, *x.shape[1:])  # (B,T,C,H,W)

        x_primary = _resize(x_primary, size=cfg.primary_image_size)
        x_wrist = _resize(x_wrist, size=cfg.wrist_image_size)

        # OctoPt (PyTorch) expects channel-first uint8 images: (B,T,C,H,W).
        def _to_octo_uint8_btc_hw(x: Tensor) -> Tensor:
            return (x * 255.0).round().to(torch.uint8)

        # Build pad masks from LeRobotDataset padding flags when available.
        # Convention: for a queried key K with delta_timestamps, dataset adds boolean tensor `f"{K}_is_pad"` of shape (B,T)
        def _get_timestep_pad_mask_for(key: str) -> Tensor:
            is_pad = batch.get(f"{key}_is_pad", None)
            if is_pad is None:
                return torch.ones((B, T), dtype=torch.bool, device=x_primary.device)
            if is_pad.ndim != 2 or is_pad.shape != (B, T):
                raise ValueError(f"Expected {key}_is_pad shape (B,T)={(B, T)}, got {tuple(is_pad.shape)}")
            return (~is_pad.to(device=x_primary.device)).to(dtype=torch.bool)

        primary_pad = _get_timestep_pad_mask_for(cfg.primary_image_key)
        wrist_pad = _get_timestep_pad_mask_for(cfg.wrist_image_key)
        timestep_pad_mask = primary_pad & wrist_pad
        pad_mask_dict = {
            "image_primary": primary_pad,
            "image_wrist": wrist_pad,
            # Some Octo checkpoints include a timestep tokenizer; provide its pad mask to avoid warnings.
            "timestep": timestep_pad_mask,
        }

        # Optional observation keys present in some Octo example batches; prefer dataset-provided values
        # (e.g. frame_index/timestamp for timestep, and next.done for episode end), else fall back to defaults.
        def _as_bt(x: Tensor, *, name: str) -> Tensor:
            if x.ndim == 1:
                x = x[:, None].repeat(1, T)
            if x.ndim != 2 or x.shape[:2] != (B, T):
                raise ValueError(f"Expected {name} shape (B,T)={(B, T)}, got {tuple(x.shape)}")
            return x

        timestep_src = None
        timestep_key = None
        for k in ("timestep", "frame_index", "timestamp"):
            v = batch.get(k, None)
            if isinstance(v, torch.Tensor):
                timestep_src = v
                timestep_key = k
                break
        if timestep_src is None:
            timestep = torch.arange(T, device=x_primary.device, dtype=torch.int64)[None, :].repeat(B, 1)
        else:
            assert timestep_key is not None
            timestep = _as_bt(timestep_src.to(device=x_primary.device), name=f"batch[{timestep_key!r}]")

        task_completed_src = None
        task_completed_key = None
        for k_done in ("task_completed", "next.done", "done", "terminal", "episode_done", "episode_end"):
            v = batch.get(k_done, None)
            if isinstance(v, torch.Tensor):
                task_completed_src = v
                task_completed_key = k_done
                break
        if task_completed_src is None:
            task_completed = torch.zeros((B, T), device=x_primary.device, dtype=torch.bool)
        else:
            assert task_completed_key is not None
            task_completed = _as_bt(
                task_completed_src.to(device=x_primary.device),
                name=f"batch[{task_completed_key!r}]",
            )
            task_completed = task_completed.to(dtype=torch.bool)

        # Some Octo checkpoints expect `task_completed` to have extra trailing dims (e.g. (B,T,H)).
        # We broadcast to match the checkpoint's example_batch shape when available.
        try:
            example_tc = getattr(self.octo, "example_batch", None)
            if isinstance(example_tc, dict):
                example_tc = example_tc.get("observation", {}).get("task_completed", None)
            if example_tc is not None and hasattr(example_tc, "shape"):
                tail = tuple(example_tc.shape[2:])
                if tail:
                    view_shape = (B, T) + (1,) * len(tail)
                    task_completed = task_completed.view(view_shape).expand((B, T) + tail)
        except Exception:
            # Best-effort only; keep (B,T) if we cannot introspect example_batch.
            pass

        obs = {
            "image_primary": _to_octo_uint8_btc_hw(x_primary),
            "image_wrist": _to_octo_uint8_btc_hw(x_wrist),
            "timestep_pad_mask": timestep_pad_mask,
            "pad_mask_dict": pad_mask_dict,
            "timestep": timestep,
            "task_completed": task_completed,
        }

        texts = batch.get("task", None)
        if texts is None:
            # Upstream Octo requires either goals or texts; LeRobotDataset usually injects `task`.
            texts = [""] * B
        if isinstance(texts, str):
            texts = [texts]
        if len(texts) != B:
            raise ValueError(f"Expected {B} task strings, got {len(texts)}.")
        return obs, list(texts), timestep_pad_mask, pad_mask_dict

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict | None]:
        cfg = self.config
        obs, texts, timestep_pad_mask, _pad_mask_dict = self._build_observations(batch)
        device = obs["image_primary"].device
        B = int(obs["image_primary"].shape[0])
        T = int(obs["image_primary"].shape[1])

        tasks = self.octo.create_tasks(texts=texts, device=device)

        # Supervise only the *last* timestep in the observation window with a future action chunk.
        # Dataset provides `action` stacked over cfg.action_delta_indices, i.e. shape (B, H, A).
        action_chunk = batch.get("action", None)
        if not isinstance(action_chunk, torch.Tensor):
            raise ValueError("Batch is missing 'action' tensor.")
        if action_chunk.ndim != 3:
            raise ValueError(f"Expected batch['action'] shape (B,H,A), got {tuple(action_chunk.shape)}")

        H = int(action_chunk.shape[1])
        A = int(action_chunk.shape[2])
        if H != int(cfg.action_horizon):
            raise ValueError(f"Expected action_horizon={cfg.action_horizon}, got chunk length {H}")

        action_is_pad = batch.get("action_is_pad", None)
        if action_is_pad is None:
            valid_steps = torch.ones((B, H), dtype=torch.bool, device=device)
        else:
            if action_is_pad.ndim != 2 or tuple(action_is_pad.shape) != (B, H):
                raise ValueError(f"Expected action_is_pad shape (B,H)={(B, H)}, got {tuple(action_is_pad.shape)}")
            valid_steps = (~action_is_pad.to(device=device)).to(dtype=torch.bool)

        action_chunk = action_chunk.to(device=device, dtype=torch.float32)
        norm_chunk = self._normalize_actions(action_chunk)

        gt_actions = torch.zeros((B, T, H, A), dtype=torch.float32, device=device)
        gt_actions[:, -1, :, :] = norm_chunk
        action_pad_mask = torch.zeros((B, T, H, A), dtype=torch.bool, device=device)
        action_pad_mask[:, -1, :, :] = valid_steps[:, :, None].expand(B, H, A)

        _, head_outputs = self.octo(
            observations=obs,
            tasks=tasks,
            timestep_pad_mask=timestep_pad_mask,
            action_pad_mask=action_pad_mask,
            gt_actions=gt_actions,
            train=True,
            verbose=False,
            save_attention_mask=False,
        )
        loss = head_outputs["action"][0]

        # Lightweight metric: MSE on the first predicted action in normalized space.
        with torch.no_grad():
            pred = self.octo.sample_actions(
                observations=obs,
                tasks=tasks,
                unnormalization_statistics=None,
                timestep_pad_mask=timestep_pad_mask,
                train=False,
            )
            pred0 = pred[:, 0, :]  # (B,A) on CPU
            gt0 = norm_chunk[:, 0, :].detach().to(device="cpu")
            valid0 = valid_steps[:, 0].detach().to(device="cpu")
            if bool(valid0.any()):
                mse0 = ((pred0 - gt0) ** 2).mean(dim=-1)
                action_mse0 = float(mse0[valid0].mean().item())
            else:
                action_mse0 = float("nan")

        metrics = {"action_mse0": action_mse0}
        return loss, metrics

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any]) -> Tensor:
        self.eval()
        obs, texts, timestep_pad_mask, _pad_mask_dict = self._build_observations(batch)
        device = obs["image_primary"].device
        tasks = self.octo.create_tasks(texts=texts, device=device)
        unnorm = self._get_action_unnormalization_statistics()
        if unnorm is None:
            raise ValueError(
                "Missing dataset action statistics for unnormalization. "
                "Instantiate policy with dataset metadata stats (e.g. via `make_policy`)."
            )
        actions = self.octo.sample_actions(
            observations=obs,
            tasks=tasks,
            unnormalization_statistics=unnorm,
            timestep_pad_mask=timestep_pad_mask,
            train=False,
        )
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any]) -> Tensor:
        actions = self.predict_action_chunk(batch)
        return actions[:, 0, :]


