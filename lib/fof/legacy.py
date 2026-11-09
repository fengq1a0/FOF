"""Backward-compatible wrappers with the signatures of the original FOF code.

The original modules preallocated scratch buffers for a fixed batch size; the
new implementation allocates through PyTorch's caching allocator, so the batch
size argument is ignored and the modules hold no state.
"""

import torch

from .encode import mesh_to_fof, render_fof


class FOF(torch.nn.Module):
    """``FOF(B, H)(v_tensor [B, F, 3, 3], C) -> [B, C, H, H]``."""

    def __init__(self, B=None, H=512, fill="robust"):
        super().__init__()
        self.H = H
        self.fill = fill

    def forward(self, v_tensor, C):
        return mesh_to_fof(v_tensor, num_coeffs=C, resolution=self.H, fill=self.fill)


class FOF_Normal(torch.nn.Module):
    """``FOF_Normal(B, H)(v [B,F,3,3], vn [B,F,3,3], C) -> (fof, dF, dB, nF, nB)``.

    ``dF``/``dB`` are the largest/smallest surface depth (``-1``/``+1`` where
    nothing is hit), ``nF``/``nB`` the unit normals there (``[B, 3, H, W]``).
    """

    def __init__(self, B=None, H=512, fill="robust"):
        super().__init__()
        self.H = H
        self.fill = fill

    def forward(self, v_tensor, vn_tensor, C):
        out = render_fof(v_tensor, normals=vn_tensor, num_coeffs=C, resolution=self.H, fill=self.fill)
        return out.fof, out.depth_front, out.depth_back, out.normal_front, out.normal_back


def render(v, num, res, fill="robust"):
    """``render(v [B, F, 3, 3], num, res) -> [B, num, res, res]`` (original fill rule by default)."""
    return mesh_to_fof(v, num_coeffs=num, resolution=res, fill=fill)
