"""Crack-free marching cubes on PyTorch tensors (CUDA kernels + a vectorized PyTorch fallback).

* Tables come from ``mc_tables.py`` (generated, not hand-typed).  Ambiguous faces are resolved
  with the asymptotic decider computed from the face's own four values, so neighbouring cells
  always agree, and loop triangulations never put a diagonal on a cube face: the output is a
  closed, consistently oriented 2-manifold wherever the volume border is outside.  (As with any
  marching cubes, samples exactly equal to the level can make several edge vertices coincide
  at a grid point, giving zero-area triangles; there is no interior disambiguation as in MC33,
  so ambiguous cells may get a different, still manifold, topology than the trilinear field.)
* ``pad`` adds a virtual one-sample border of that value around the volume (closes surfaces that
  touch the volume boundary): same mesh as padding the volume explicitly, without the copy.
* Every vertex is identified by an id on the grid ``[D, H, W]`` (border included when padded):
  grid-edge vertices by ``((d * H + h) * W + w) * 3 + axis`` (axis 0: w, 1: h, 2: d), the rare
  cell-interior vertices (loops that need a centre vertex, the case MC33 handles with its interior
  vertex) by ``3*D*H*W + cell * MAX_CENTRES + slot``.  Triangles are emitted as ids and
  deduplicated with ``torch.unique``, so shared vertices are exact and the output is deterministic.
* Edge vertices are linear interpolations along their edge; centre vertices are the mean of their
  loop's vertices.  Positions are (d, h, w) index coordinates of the input volume (the border, if
  any, lies at -1 and D, H, W).  Triangles are counter-clockwise around normals pointing from high
  to low values in the right-handed (w, h, d) frame (that is, clockwise in (d, h, w), the same
  convention as scikit-image).
"""

from typing import Optional

import numpy as np
import torch

from . import _ext
from . import mc_tables as T

_TABLES = {}


def _tables(device):
    key = str(device)
    if key not in _TABLES:
        edge = np.array(
            [list(T._corner_offset(T.EDGE_CORNERS[e][0]).astype(np.int64)) + [e // 4] for e in range(12)], np.int8
        )  # (dx, dy, dz, axis) of each local edge's start corner
        arrays = (
            T.AMB.astype(np.int32), T.BASE.astype(np.int32), T.NTRI.astype(np.int32), T.TRIS.astype(np.int8),
            T.NCEN.astype(np.int32), T.CLOOP.astype(np.int32), np.array(T.FACE_CORNERS, np.int8), edge,
        )
        _TABLES[key] = tuple(torch.from_numpy(np.ascontiguousarray(a)).to(device) for a in arrays)
    return _TABLES[key]


def vertex_axis(vertex_ids: torch.Tensor, shape, padded: bool = False) -> torch.Tensor:
    """Grid axis of each vertex's edge (0: w, 1: h, 2: d) or -1 for cell-interior vertices.
    ``shape``: the volume passed to :func:`marching_cubes`; ``padded``: whether ``pad`` was set."""
    D, H, W = (int(n) + (2 if padded else 0) for n in shape)
    return torch.where(vertex_ids < 3 * D * H * W, vertex_ids % 3, torch.full_like(vertex_ids, -1))


def _pad_f32(pad):
    """Border value as the float32 sample the kernels see (None: no border)."""
    if pad is None:
        return None
    with np.errstate(over="ignore"):
        f = float(np.float32(pad))
    if not np.isfinite(f):
        raise ValueError(f"pad must be finite in float32, got {pad!r}")
    return f


def _samples(vol, d, h, w, pad):
    """Samples of the grid with the optional border (``pad`` already float32-rounded)."""
    if pad is None:
        return vol[d, h, w]
    D, H, W = vol.shape
    d, h, w = d - 1, h - 1, w - 1
    inside = (d >= 0) & (d < D) & (h >= 0) & (h < H) & (w >= 0) & (w < W)
    v = vol[d.clamp(0, D - 1), h.clamp(0, H - 1), w.clamp(0, W - 1)]
    return torch.where(inside, v, torch.full_like(v, pad))


def edge_positions(vol: torch.Tensor, edge_ids: torch.Tensor, level: float, pad=None) -> torch.Tensor:
    """(d, h, w) positions, in ``vol``'s index coordinates, of the iso-crossings on the given edges
    of the grid (``vol`` plus the ``pad`` border when it is not None), float32."""
    pad = _pad_f32(pad)
    o = 0 if pad is None else 1
    D, H, W = (n + 2 * o for n in vol.shape)
    axis = edge_ids % 3
    p = edge_ids // 3
    d, h, w = p // (W * H), (p // W) % H, p % W
    v0 = _samples(vol, d, h, w, pad).double()
    v1 = _samples(vol, d + (axis == 2).long(), h + (axis == 1).long(), w + (axis == 0).long(), pad).double()
    t = (level - v0) / (v1 - v0)  # double, with the same double level used to classify
    pos = torch.stack([d, h, w], 1).to(torch.float64) - o  # exact integers
    pos[torch.arange(len(p), device=p.device), 2 - axis] += t  # axis 0 (w) is column 2 of (d, h, w)
    return pos.to(torch.float32)


def _float_below(level: float) -> float:
    """Largest float32 not above ``level``: for float samples, v > level iff v > _float_below(level)."""
    with np.errstate(over="ignore"):
        f = np.float32(level)
    if float(f) > level:
        f = np.nextafter(f, np.float32(-np.inf))
    return float(f)


def _grid_layers(vol, a, b, pad):
    """Layers ``[a, b)`` of the grid (``vol`` plus the optional border) as a tensor."""
    if pad is None:
        return vol[a:b]
    D, H, W = vol.shape
    out = vol.new_full((b - a, H + 2, W + 2), pad)
    lo, hi = max(a - 1, 0), min(b - 1, D)  # stored layers inside [a, b)
    if hi > lo:
        out[lo + 1 - a:hi + 1 - a, 1:-1, 1:-1] = vol[lo:hi]
    return out


def _mc_torch_slab(vol, level, d0, d1, pad=None):
    """Triangle ids / centre records of the cells with d in [d0, d1) (vectorized PyTorch)."""
    o = 0 if pad is None else 1
    D, H, W = (n + 2 * o for n in vol.shape)
    device = vol.device
    amb_t, base_t, ntri_t, tris_t, ncen_t, cloop_t, face_t, edge_t = _tables(device)
    sub = _grid_layers(vol, d0, d1 + 1, pad)  # slab-sized, also when padded
    level_f = _float_below(level)
    n = d1 - d0
    corner = [sub[(k >> 2) & 1:((k >> 2) & 1) + n, (k >> 1) & 1:((k >> 1) & 1) + H - 1, k & 1:(k & 1) + W - 1]
              for k in range(8)]
    cfg = torch.zeros(corner[0].shape, dtype=torch.uint8, device=device)
    for k in range(8):
        cfg |= (corner[k] > level_f).to(torch.uint8) << k  # exact: same as comparing with the double level
    act = ((cfg != 0) & (cfg != 255)).nonzero()
    empty = (torch.empty(0, 3, dtype=torch.long, device=device), torch.empty(0, 13, dtype=torch.long, device=device))
    if act.numel() == 0:
        return empty
    dl, h, w = act.unbind(1)
    d = dl + d0
    cell_lin = (d * (H - 1) + h) * (W - 1) + w
    cfg = cfg[dl, h, w].long()
    vals = torch.stack([c[dl, h, w] for c in corner], 1).double() - level  # double, same as the CUDA kernel
    amb = amb_t[cfg]
    bits = torch.zeros_like(cfg)
    rank = torch.zeros_like(cfg)
    for fi in range(6):
        c0, c1, c2, c3 = (int(x) for x in face_t[fi])
        is_amb = (amb >> fi) & 1
        p02, p13 = vals[:, c0] * vals[:, c2], vals[:, c1] * vals[:, c3]
        connect = torch.where(vals[:, c0] > 0, p02 > p13, p13 > p02).long()
        bits |= (connect * is_amb) << rank
        rank += is_amb
    var = base_t[cfg] + bits
    e = edge_t.long()  # [12, 4] = (dx, dy, dz, axis)
    centre0 = 3 * D * H * W

    def ids(cells, codes):  # codes: local edge ids (< 12) or centre slots (>= 12)
        cc = codes.clamp(max=11)
        g = (((d[cells] + e[cc, 2]) * H + (h[cells] + e[cc, 1])) * W + (w[cells] + e[cc, 0])) * 3 + e[cc, 3]
        return torch.where(codes < 12, g, centre0 + cell_lin[cells] * T.MAX_CENTRES + (codes - 12))

    nt = ntri_t[var].long()
    cell = torch.repeat_interleave(torch.arange(len(var), device=device), nt)
    k = torch.arange(len(cell), device=device) - torch.repeat_interleave(torch.cumsum(nt, 0) - nt, nt)
    codes = tris_t[var[cell]].view(-1, T.MAX_TRIS, 3)[torch.arange(len(cell), device=device), k].long()
    tri = ids(cell[:, None].expand(-1, 3), codes)

    nc = ncen_t[var].long()
    ccell = torch.repeat_interleave(torch.arange(len(var), device=device), nc)
    if ccell.numel() == 0:
        return tri, empty[1]
    slot = torch.arange(len(ccell), device=device) - torch.repeat_interleave(torch.cumsum(nc, 0) - nc, nc)
    mask = cloop_t[var[ccell], slot].long()
    edges = torch.arange(12, device=device)
    in_loop = ((mask[:, None] >> edges) & 1).bool()  # [C, 12]
    # loop edges first in increasing edge order, -1 padding (same layout as the CUDA kernel)
    order = torch.argsort((~in_loop).to(torch.int8) * 16 + edges, dim=1, stable=True)
    sel = torch.gather(edges.expand(len(ccell), -1), 1, order)
    valid = torch.gather(in_loop, 1, order)
    loop_ids = torch.where(valid, ids(ccell[:, None].expand(-1, 12), sel), torch.full_like(sel, -1))
    return tri, torch.cat([ids(ccell, slot + 12)[:, None], loop_ids], 1)


def _mc_torch(vol: torch.Tensor, level: float, max_cells: int = 1 << 24, pad=None):
    """Same output as the CUDA kernel, processed in depth slabs of at most ``max_cells`` cells."""
    pad = _pad_f32(pad)
    o = 0 if pad is None else 1
    D, H, W = (n + 2 * o for n in vol.shape)
    step = max(1, max_cells // ((H - 1) * (W - 1)))
    parts = [_mc_torch_slab(vol, level, d0, min(d0 + step, D - 1), pad) for d0 in range(0, D - 1, step)]
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])


def marching_cubes(vol: torch.Tensor, level: float = 0.5, *, pad: Optional[float] = None, backend: str = "auto"):
    """Iso-surface of ``vol [D, H, W]`` at ``level`` (inside = values above ``level``).

    ``pad``: if not None, a virtual one-sample border of this value surrounds the volume (a value
    below ``level`` closes the surface at the volume boundary); nothing is copied.
    Returns ``(vertex_ids [V] int64, verts [V, 3] float32 in (d, h, w) index coordinates of
    ``vol``, faces [F, 3] int64)`` on ``vol``'s device; see :func:`vertex_axis` for the vertex ids.
    ``backend``: "cuda", "torch" or "auto" (CUDA kernels for CUDA tensors when available).
    """
    if vol.dim() != 3:
        raise ValueError(f"vol must be [D, H, W], got {tuple(vol.shape)}")
    if backend not in ("auto", "cuda", "torch"):
        raise ValueError(f"backend must be 'auto', 'cuda' or 'torch', got {backend!r}")
    if backend == "cuda" and not vol.is_cuda:
        raise ValueError("backend='cuda' needs a CUDA tensor")
    vol = vol.float().contiguous()
    if vol.numel() and not bool(torch.isfinite(torch.stack(torch.aminmax(vol))).all()):  # NaN propagates
        raise ValueError("vol contains non-finite values")
    device = vol.device
    level = float(level)  # double; float samples are compared exactly (see _float_below)
    pad = _pad_f32(pad)
    o = 0 if pad is None else 1
    D, H, W = (n + 2 * o for n in vol.shape)  # grid, border included
    if vol.numel() == 0 or min(D, H, W) < 2:
        return (torch.empty(0, dtype=torch.long, device=device), torch.empty(0, 3, device=device),
                torch.empty(0, 3, dtype=torch.long, device=device))
    use_cuda = backend == "cuda" or (backend == "auto" and vol.is_cuda and _ext.available())
    with torch.no_grad():
        if use_cuda:
            tri, rec = _ext.load().marching_cubes(vol, level, o, 0.0 if pad is None else pad, *_tables(device),
                                                  T.MAX_CENTRES)
        else:
            tri, rec = _mc_torch(vol, level, pad=pad)
        ids, inv = torch.unique(tri.reshape(-1), return_inverse=True)
        faces = inv.view(-1, 3)
        n_edge = int(torch.searchsorted(ids, torch.tensor(3 * D * H * W, device=device)))
        pos = torch.empty(len(ids), 3, device=device)
        pos[:n_edge] = edge_positions(vol, ids[:n_edge], level, pad)
        if n_edge < len(ids):  # centre vertices: mean of their loop's vertices (records sorted by id)
            loops = rec[:, 1:]
            valid = loops >= 0
            idx = torch.searchsorted(ids[:n_edge], loops.clamp(min=0))
            p = pos[idx] * valid[..., None]
            pos[n_edge:] = p.sum(1) / valid.sum(1, keepdim=True)
        return ids, pos, faces
