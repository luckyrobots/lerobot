from __future__ import annotations

import torch
import torch.nn as nn


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization (AdaLN).

    This layer applies standard LayerNorm and then modulates the normalized
    activations with scale and shift parameters predicted from a context
    vector.  The context can encode any information (e.g., state statistics,
    timestep, task embedding) and has dimensionality ``context_dim``.

    The design keeps the default output close to the vanilla LayerNorm by
    initialising the modulation weights to zero (so that ``scale≈0`` and
    ``shift≈0`` at the start of training).
    """

    def __init__(self, hidden_size: int, context_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        # Predict additive bias and multiplicative scale from context
        self.scale = nn.Linear(context_dim, hidden_size, bias=False)
        self.shift = nn.Linear(context_dim, hidden_size, bias=False)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Apply AdaLN.

        Args:
            x: Tensor of shape ``(batch, seq_len, hidden_size)``.
            context: Tensor of shape ``(batch, context_dim)``.
        """
        x_norm = self.norm(x)
        # Compute per-sample modulation terms and unsqueeze to broadcast along seq_len.
        scale = self.scale(context).unsqueeze(1)  # (B, 1, H)
        shift = self.shift(context).unsqueeze(1)  # (B, 1, H)
        return x_norm * (1.0 + scale) + shift 