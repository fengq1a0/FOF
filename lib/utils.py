import numpy as np
import torch
import torch.nn.functional as F

from .fof import fof_to_mesh
from .fof.decode import _mc_skimage


def toDevice(sample, device):
    if torch.is_tensor(sample):
        return sample.to(device, non_blocking=True)
    elif isinstance(sample, dict):
        return {k: toDevice(v, device) for k, v in sample.items()}
    elif isinstance(sample, list):
        return [toDevice(s, device) for s in sample]
    elif isinstance(sample, tuple):
        return tuple([toDevice(s, device) for s in sample])
    else:
        return sample


class Recon:
    """FOF -> mesh as in the original code (open surfaces at the borders,
    degenerate faces kept); ``decode`` is bit-compatible with the original.  New code should call
    ``lib.fof.fof_to_mesh``, which evaluates the series at the requested depth
    resolution and closes the surface by default.
    """

    def __init__(self, device=None, dim=None, depth_res=512):
        self.device = device
        self.depth_res = depth_res

    def _basis(self, dim, device):  # the original float32 formula, bit for bit
        z = (torch.arange(self.depth_res, dtype=torch.float32, device=device) + 0.5) / self.depth_res
        z = torch.arange(dim, dtype=torch.float32, device=device).view(1, dim) * z.view(-1, 1) * np.pi
        basis = torch.cos(z)
        basis[:, 0] = 0.5
        return basis

    @torch.no_grad()
    def decode(self, ceof, resolution=512):
        # 512 depth samples, then trilinear resizing of the volume to resolution^3
        with torch.autocast(device_type=ceof.device.type, enabled=False):
            occ = torch.einsum("dc, chw -> dhw", self._basis(ceof.shape[0], ceof.device), ceof.float())
            if tuple(occ.shape) != (resolution,) * 3:
                occ = F.interpolate(occ[None, None], (resolution,) * 3, mode="trilinear", align_corners=False)[0, 0]
        mc = _mc_skimage(occ, 0.5, close=False, allow_degenerate=True)
        if mc is None:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64)
        v, f = mc
        v = v + 0.5  # the original float32 conversion, bit for bit
        v = v / (resolution / 2) - 1
        v[:, 1] *= -1
        return np.ascontiguousarray(v[:, [2, 1, 0]]), np.ascontiguousarray(f[:, [0, 2, 1]])

    def decode_lossless(self, ceof, refine_roots=False):
        # The original algorithm (Lewiner marching cubes, same vertices and triangles, same
        # least-squares energy) with an exact solve instead of Eigen's float32 incomplete-Cholesky
        # CG, so results match up to the solver tolerance, not bit for bit.  (For non-cubic
        # volumes the original also normalized x / y by the depth resolution; this does not.)
        return fof_to_mesh(ceof, depth_res=self.depth_res, backend="skimage", lossless=True, solver="direct",
                           refine_roots=refine_roots, close=False, allow_degenerate=True)
