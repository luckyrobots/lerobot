from __future__ import annotations

"""Optical-flow wrapper built around the NeuFlow v2 model.

This utility offers a thin, production-ready API for computing dense optical
flow maps between two RGB frames.  It is **self-contained** and makes no
assumptions about how frames are provided – NumPy arrays, PIL images, or PyTorch
Tensors all work.  The first call lazily loads the NeuFlow model (either from a
local checkpoint path or from a HuggingFace repo) and re-uses it for the
lifetime of the current Python process.

Typical usage
-------------
>>> from lerobot.policies.lucky_act.optical_flow import OpticalFlowProcessor
>>> of_proc = OpticalFlowProcessor(device="cuda")
>>> flow_uv = of_proc.compute_flow(frame_prev, frame_curr)  # np.float32 [H, W, 2]
"""

from pathlib import Path
import logging
from typing import Optional, Union

import numpy as np
import torch

try:
    from .neuflow.neuflow import NeuFlow
    from .neuflow.backbone_v7 import ConvBlock
except Exception as exc:  # pragma: no cover – handled at runtime
    raise ImportError(
        "Failed to import NeuFlow. Ensure that the neuflow subdirectory is present."
    ) from exc

__all__ = ["OpticalFlowProcessor"]


def _fuse_conv_bn(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d) -> torch.nn.Conv2d:
    """Fuse a Conv2d + BatchNorm layer pair for faster inference (no-grad).

    This mirrors the helper used in NeuFlow demo scripts.
    """
    fusedconv = (
        torch.nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.running_var + bn.eps)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


class OpticalFlowProcessor:
    """Lazy-loading NeuFlow v2 optical-flow processor.

    Parameters
    ----------
    weight_path: str | Path | None
        Local `.pth` checkpoint.  If *None*, the model is fetched via
        `NeuFlow.from_pretrained(hf_repo)`.
    hf_repo: str | None
        HuggingFace repo ID containing the weights (ignored if *weight_path* is
        provided).
    device: str | torch.device
        Target device – defaults to *cuda* if available.
    iters_s16, iters_s8: int
        Refinement iterations at 1/16 and 1/8 resolutions.  Smaller numbers are
        faster (recommended: 1, 8 for real-time).
    amp: bool
        Whether to use half-precision (autocast) when running on CUDA.
    """

    _MODEL: Optional[NeuFlow] = None  # shared singleton
    _BHWD: Optional[tuple[int, int, int]] = None  # batch, height, width used to init kernels

    def __init__(
        self,
        *,
        weight_path: Union[str, Path, None] = None,
        hf_repo: Optional[str] = "Study-is-happy/neuflow-v2",
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        iters_s16: int = 1,
        iters_s8: int = 8,
        amp: bool | None = None,
    ) -> None:
        self.weight_path = Path(weight_path) if weight_path else None
        self.hf_repo = hf_repo if weight_path is None else None
        self.device = torch.device(device)
        self.iters_s16 = int(iters_s16)
        self.iters_s8 = int(iters_s8)
        self.amp = torch.cuda.is_available() if amp is None else bool(amp)

        # Lazy model loading – done per process the first time we need it
        if OpticalFlowProcessor._MODEL is None:
            logging.info("Loading NeuFlow model for optical-flow estimation …")
            OpticalFlowProcessor._MODEL = self._load_model()
            OpticalFlowProcessor._MODEL.eval()
            if self.device.type == "cuda":
                OpticalFlowProcessor._MODEL.half()  # use fp16 on GPU

        # Make sure the singleton is on the requested device (torch.nn.Module does not have a
        # `device` attribute – check the device of the first parameter instead).
        model_param_device = next(OpticalFlowProcessor._MODEL.parameters()).device
        if model_param_device != self.device:
            OpticalFlowProcessor._MODEL.to(self.device)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def compute_flow(
        self,
        prev_frame: Union[np.ndarray, torch.Tensor, "PIL.Image.Image"],
        curr_frame: Union[np.ndarray, torch.Tensor, "PIL.Image.Image"],
        *,
        output_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Compute a dense optical-flow map **curr → prev**.

        The direction follows NeuFlow: flow vectors point from *curr* to where
        the pixel came from in *prev*.
        """
        prev_t, curr_t = self._preprocess(prev_frame), self._preprocess(curr_frame)

        model = OpticalFlowProcessor._MODEL  # after lazy load guaranteed not None
        assert model is not None

        # Initialise kernels if spatial size changed
        b, _, h, w = prev_t.shape
        bhwd = (b, h, w)
        if OpticalFlowProcessor._BHWD != bhwd:
            model.init_bhwd(b, h, w, str(self.device), amp=self.amp)
            OpticalFlowProcessor._BHWD = bhwd

        flow_list = model(prev_t, curr_t, iters_s16=self.iters_s16, iters_s8=self.iters_s8)
        flow = flow_list[-1][0]  # [C=2, H, W]
        flow = flow.permute(1, 2, 0)  # -> [H, W, 2]

        if output_numpy:
            return flow.float().cpu().numpy()
        return flow.float()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> NeuFlow:
        """Instantiate NeuFlow and load weights (local or hf)."""
        model = NeuFlow().to(self.device)

        if self.weight_path and self.weight_path.is_file():
            ckpt = torch.load(self.weight_path, map_location=self.device)
            state_dict = ckpt["model"] if "model" in ckpt else ckpt
            model.load_state_dict(state_dict, strict=True)
        else:
            # Fallback to HF – this triggers internal download / cache
            if self.hf_repo is None:
                raise FileNotFoundError(
                    "Neither weight_path provided nor hf_repo specified for NeuFlow weights."
                )
            model = NeuFlow.from_pretrained(self.hf_repo).to(self.device)

        # Fuse Conv + BN for every ConvBlock – improves throughput
        for m in model.modules():
            if isinstance(m, ConvBlock):  # type: ignore
                m.conv1 = _fuse_conv_bn(m.conv1, m.norm1)  # type: ignore[attr-defined]
                m.conv2 = _fuse_conv_bn(m.conv2, m.norm2)  # type: ignore[attr-defined]
                # Remove BN and switch to fused forward
                delattr(m, "norm1")
                delattr(m, "norm2")
                m.forward = m.forward_fuse  # type: ignore[assignment]
        return model

    def _preprocess(self, frame: Union[np.ndarray, torch.Tensor, "PIL.Image.Image"]) -> torch.Tensor:
        """Convert various frame formats to a CUDA/CPU Tensor expected by NeuFlow."""
        import torchvision.transforms.functional as F  # lazy import

        if isinstance(frame, np.ndarray):
            if frame.ndim == 3 and frame.shape[2] == 3:
                tensor = torch.from_numpy(frame).permute(2, 0, 1)
            else:
                raise ValueError("NumPy frame must be HxWx3 RGB/BGR array.")
        elif torch.is_tensor(frame):
            if frame.ndim == 3:  # C x H x W or H x W x C
                if frame.shape[0] in {1, 2, 3}:  # assume C x H x W
                    tensor = frame.detach()
                else:  # assume H x W x C
                    tensor = frame.permute(2, 0, 1)
            elif frame.ndim == 4 and frame.shape[0] == 1:
                tensor = frame.squeeze(0)
            else:
                raise ValueError("Unsupported tensor shape for frame: %s" % str(frame.shape))
        else:  # PIL Image
            tensor = F.pil_to_tensor(frame)

        tensor = tensor.to(self.device, dtype=torch.half if self.amp and self.device.type == "cuda" else torch.float)
        tensor = tensor.unsqueeze(0)  # add batch dim
        return tensor 