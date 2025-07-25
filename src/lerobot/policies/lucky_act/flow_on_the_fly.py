from __future__ import annotations

"""On-the-fly optical-flow transform for LeRobot batches.

This small utility is intended for **run-time** use during training or
inference: given a batch dictionary returned by `LeRobotDataset`, it
computes dense optical-flow maps between two successive frames for each
specified camera and injects them back into the batch under new keys.

Usage
-----
>>> from lerobot.policies.lucky_act.flow_on_the_fly import FlowOnTheFly
>>> transform = FlowOnTheFly({
...     "observation.image_cam1": "observation.image_flow_cam1",
...     "observation.image_cam2": "observation.image_flow_cam2",
... })
>>> new_batch = transform(batch)  # now contains flow tensors (H,W,2)

Notes
-----
• Expects each image entry to be a tensor shaped (2, C, H, W) or (2, H,
  W, C): first slice is previous frame (t-1), second is current (t).
  This is compatible with `n_obs_steps = 2` and
  `observation_delta_indices = [-1, 0]` in the policy config.
• If only a single frame is found (shape (C,H,W)), a zero-flow map is
  generated as safeguard.
• Computation leverages the project-wide `OpticalFlowProcessor` (lazy
  singleton) for efficiency.
"""

from pathlib import Path
from typing import Dict, Mapping, Union

import torch
import numpy as np

from .optical_flow import OpticalFlowProcessor

__all__ = ["FlowOnTheFly"]


class FlowOnTheFly:
    def __init__(
        self,
        camera_to_flow_key: Mapping[str, str],
        *,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """Initialise the transform.

        Parameters
        ----------
        camera_to_flow_key: Mapping of image keys (source) to flow keys
            (destination) to be added into the batch.
        device: Where to run NeuFlow. Defaults to CUDA when available.
        """
        self.camera_to_flow_key: Dict[str, str] = dict(camera_to_flow_key)
        self._of_processor = OpticalFlowProcessor(device=device)

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = dict(batch)  # shallow copy – don't mutate caller's dict

        for cam_key, flow_key in self.camera_to_flow_key.items():
            if flow_key in batch:
                # Flow already provided (e.g., during inference); skip.
                continue
            if cam_key not in batch:
                raise KeyError(f"Camera key '{cam_key}' not found in batch.")

            img_tensor = batch[cam_key]
            # Accept different tensor layouts:
            # (C,H,W) single frame  → generate zero flow
            # (2,C,H,W)             → previous & current frames
            # (B,2,C,H,W)           → batched previous/current frames

            if img_tensor.ndim == 3:
                # Single frame – generate zero flow
                c, h, w = img_tensor.shape
                flow = torch.zeros(h, w, 2, dtype=torch.float32, device=img_tensor.device)
                batch[flow_key] = flow
                continue

            if img_tensor.ndim == 4 and img_tensor.shape[0] == 2:
                # Unbatched pair of frames
                prev, curr = img_tensor[0], img_tensor[1]
                batched = False
            elif img_tensor.ndim == 5 and img_tensor.shape[1] == 2:
                # Batched frames (B,2,C,H,W)
                batched = True
            else:
                raise ValueError(
                    f"Expected '{cam_key}' to have shape (2,C,H,W) or (C,H,W) or (B,2,C,H,W). Got {img_tensor.shape}."
                )

            if batched:
                bsz = img_tensor.shape[0]
                flows = []
                for b in range(bsz):
                    prev_b, curr_b = img_tensor[b, 0], img_tensor[b, 1]
                    prev_np = self._to_numpy(prev_b)
                    curr_np = self._to_numpy(curr_b)
                    flow_uv = self._of_processor.compute_flow(prev_np, curr_np, output_numpy=True)
                    flows.append(torch.from_numpy(flow_uv).permute(2, 0, 1))
                flow_tensor = torch.stack(flows, dim=0)  # (B,2,H,W)
            else:
                prev_np = self._to_numpy(prev)
                curr_np = self._to_numpy(curr)
                flow_uv = self._of_processor.compute_flow(prev_np, curr_np, output_numpy=True)
                flow_tensor = torch.from_numpy(flow_uv).permute(2, 0, 1)  # (2,H,W)

            batch[flow_key] = flow_tensor

        return batch

    # ---------------------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------------------

    @staticmethod
    def _to_numpy(img: torch.Tensor) -> np.ndarray:
        """Convert C×H×W tensor [0-1] or [0-255] to H×W×3 uint8 for NeuFlow."""
        if img.dtype == torch.float32 or img.dtype == torch.float16:
            img_clamped = torch.clamp(img * 255.0, 0, 255).to(torch.uint8)
        elif img.dtype == torch.uint8:
            img_clamped = img
        else:
            raise ValueError(f"Unsupported image dtype: {img.dtype}")
        img_np = img_clamped.permute(1, 2, 0).cpu().numpy()
        return img_np 