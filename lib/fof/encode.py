"""Mesh -> Fourier Occupancy Field (FOF).

Conventions
-----------
* Normalized device coordinates: x to the right, y up, z towards the camera,
  orthographic projection along -z.  The image covers x, y in [-1, 1]
  (``align_corners=False``): pixel (row h, column w) of an H x W image has its
  centre at ``x = (w + 0.5) * 2 / W - 1``, ``y = 1 - (h + 0.5) * 2 / H``.
* Faces are counter-clockwise when seen from outside (outward normals), which
  is what the ``"robust"`` and ``"winding"`` fill rules rely on.
* FOF of a pixel: with ``t = (z - z_min) / (z_max - z_min)`` in [0, 1] and the
  occupancy o(t) of the pixel's ray,
      a_0 = 2 * int o(t) dt,      a_c = 2 * int o(t) cos(c*pi*t) dt,
  so that ``o(t) ~= a_0 / 2 + sum_c a_c cos(c*pi*t)`` (see ``decode.py``).
  With the default z range [-1, 1], ``a_0`` is the occupied length along z.

The encoder computes every ray/surface crossing exactly (no depth peeling, no
fixed number of layers): vertices are snapped to a 1/256 pixel grid and
coverage is decided with exact integer edge functions and a top-left rule,
so pixel centres lying on shared edges or vertices are counted exactly once.
"""

import math
import numbers
import warnings
from typing import NamedTuple, Optional, Tuple, Union

import torch

from . import _ext

SUBPIXEL_BITS = 8
_SUB = 1 << SUBPIXEL_BITS
# Vertices farther than this from the image (in pixels) are rejected so that
# fixed-point coordinates fit in int32 and edge functions never overflow int64.
_GUARD = float(1 << 21)
_INT64_MIN = -(1 << 63)

FILL_RULES = {"robust": 0, "parity": 1, "winding": 2, "hybrid": 3}

Resolution = Union[int, Tuple[int, int]]


class Fragments(NamedTuple):
    """All ray/surface crossings, grouped per pixel (CSR layout).

    ``keys[pix_ptr[p]:pix_ptr[p+1]]`` are the crossings of flat pixel
    ``p = (b * H + h) * W + w``, sorted by increasing depth z (far to near).
    Use :meth:`decode` to get depth / exit flag / face index of each key.
    """

    pix_ptr: torch.Tensor  # [B*H*W + 1] int32
    keys: torch.Tensor  # [N] int64 (bit pattern of the uint64 fragment keys)
    shape: Tuple[int, int, int]  # (B, H, W)

    def counts(self) -> torch.Tensor:
        """Number of crossings per pixel, [B, H, W] int64."""
        ptr = self.pix_ptr.long()
        return (ptr[1:] - ptr[:-1]).view(self.shape)

    def decode(self):
        """Returns ``(depth [N] float32, is_exit [N] bool, face [N] int64)``.

        ``is_exit`` is True for camera-facing triangles (a ray travelling
        towards +z leaves the solid there) and False for entries.
        """
        return decode_keys(self.keys)


class FOFRender(NamedTuple):
    """Result of :func:`render_fof`.  Batch dims follow the input."""

    fof: Optional[torch.Tensor]  # [B, C, H, W]
    mask: torch.Tensor  # [B, H, W] bool: the pixel's ray hits the surface
    depth_front: torch.Tensor  # [B, H, W] largest z (nearest to the camera)
    depth_back: torch.Tensor  # [B, H, W] smallest z
    normal_front: Optional[torch.Tensor]  # [B, 3, H, W] unit normals, 0 where empty
    normal_back: Optional[torch.Tensor]
    face_front: torch.Tensor  # [B, H, W] int64 face index, -1 where empty
    face_back: torch.Tensor
    bary_front: torch.Tensor  # [B, H, W, 3] barycentrics w.r.t. the face's vertices
    bary_back: torch.Tensor


class _Mesh(NamedTuple):
    vxy: torch.Tensor  # [B, V, 2] int32 fixed-point pixel coordinates
    vz: torch.Tensor  # [B, V] float32, NaN for invalid vertices
    faces: torch.Tensor  # [B, F, 3] int32
    normals: Optional[torch.Tensor]  # [B, V, 3] float32
    valid: torch.Tensor  # [B, V] bool
    batched: bool


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def _hw(resolution: Resolution) -> Tuple[int, int]:
    if isinstance(resolution, numbers.Integral):
        h = w = int(resolution)
    else:
        h, w = (int(r) for r in resolution)
    if h <= 0 or w <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    return h, w


def _sanitize_faces(faces: torch.Tensor, num_verts: int) -> torch.Tensor:
    """Integer faces -> int64 with every face that has an index outside
    [0, num_verts) replaced by -1 (checked before any narrowing)."""
    if faces.is_floating_point() or faces.is_complex() or faces.dtype == torch.bool:
        raise TypeError(f"faces must be an integer tensor, got {faces.dtype}")
    # widen first: comparing narrow types against num_verts would wrap the bound
    # (uint64 values >= 2^63 become negative and are rejected as well)
    faces = faces.to(torch.int64)
    bad = ((faces < 0) | (faces >= num_verts)).any(-1, keepdim=True)
    return torch.where(bad, torch.full_like(faces, -1), faces)


def _pad_list(verts, faces, normals):
    if faces is None or len(verts) != len(faces):
        raise ValueError("a list of meshes needs a matching list of faces")
    if normals is not None and len(normals) != len(verts):
        raise ValueError("a list of meshes needs a matching list of normals")
    device = torch.as_tensor(verts[0]).device if len(verts) else torch.device("cpu")
    vs = [torch.as_tensor(v, dtype=torch.float32, device=device).reshape(-1, 3) for v in verts]
    fs = [_sanitize_faces(torch.as_tensor(f, device=device).reshape(-1, 3), v.shape[0]) for f, v in zip(faces, vs)]
    B = len(vs)
    V = max([v.shape[0] for v in vs] + [0])
    F = max([f.shape[0] for f in fs] + [0])
    vp = torch.zeros(B, V, 3, device=device)
    fp = torch.full((B, F, 3), -1, dtype=torch.long, device=device)
    np_ = None if normals is None else torch.zeros(B, V, 3, device=device)
    for b in range(B):
        vp[b, : vs[b].shape[0]] = vs[b]
        fp[b, : fs[b].shape[0]] = fs[b]
        if normals is not None:
            np_[b, : vs[b].shape[0]] = torch.as_tensor(normals[b], dtype=torch.float32, device=device).reshape(-1, 3)
    return vp, fp, np_


def _batch(verts, faces, normals):
    """Normalizes the accepted input layouts to [B, V, 3] / [B, F, 3] tensors."""
    if isinstance(verts, (list, tuple)):
        verts, faces, normals = _pad_list(verts, faces, normals)
        batched = True
    else:
        verts = torch.as_tensor(verts)
        if faces is None:  # triangle soup [..., F, 3, 3]
            if verts.dim() not in (3, 4) or tuple(verts.shape[-2:]) != (3, 3):
                raise ValueError(
                    "without faces, verts must be a triangle soup [B, F, 3, 3] or [F, 3, 3], "
                    f"got {tuple(verts.shape)}"
                )
            batched = verts.dim() == 4
            if not batched:
                verts = verts[None]
            B, F = verts.shape[:2]
            if normals is not None:
                normals = torch.as_tensor(normals, device=verts.device)
                if not batched:
                    normals = normals[None]
                if normals.shape != verts.shape:
                    raise ValueError("per-corner normals must have the same shape as the triangle soup")
                normals = normals.reshape(B, F * 3, 3)
            verts = verts.reshape(B, F * 3, 3)
            faces = torch.arange(F * 3, dtype=torch.int32, device=verts.device).view(1, F, 3).expand(B, F, 3)
        else:
            if verts.dim() not in (2, 3) or verts.shape[-1] != 3:
                raise ValueError(f"verts must be [V, 3] or [B, V, 3], got {tuple(verts.shape)}")
            batched = verts.dim() == 3
            if not batched:
                verts = verts[None]
            B = verts.shape[0]
            faces = torch.as_tensor(faces, device=verts.device)
            if faces.dim() == 2:
                faces = faces[None]
            if faces.dim() != 3 or faces.shape[-1] != 3 or faces.shape[0] not in (1, B):
                raise ValueError(f"faces must be [F, 3] or [B, F, 3], got {tuple(faces.shape)}")
            faces = faces.expand(B, -1, -1)
            if normals is not None:
                normals = torch.as_tensor(normals, device=verts.device)
                if not batched:
                    normals = normals[None]
                if normals.shape != verts.shape:
                    raise ValueError("per-vertex normals must have the same shape as verts")
    if verts.shape[1] >= (1 << 31) or faces.shape[1] >= (1 << 31):
        raise ValueError("meshes are limited to 2^31 - 1 vertices / faces")
    return verts, _sanitize_faces(faces, verts.shape[1]), normals, batched


def _project(verts: torch.Tensor, H: int, W: int):
    """Normalized coordinates -> fixed-point pixel coordinates (shared by all backends)."""
    v = verts.float()
    x, y, z = v.unbind(-1)
    xp = (x + 1.0) * (W * 0.5) - 0.5
    yp = (1.0 - y) * (H * 0.5) - 0.5
    valid = (
        torch.isfinite(xp) & torch.isfinite(yp) & torch.isfinite(z)
        & (xp.abs() <= _GUARD) & (yp.abs() <= _GUARD)
    )
    zero = torch.zeros((), dtype=xp.dtype, device=xp.device)
    X = torch.where(valid, torch.round(xp * _SUB), zero).to(torch.int32)
    Y = torch.where(valid, torch.round(yp * _SUB), zero).to(torch.int32)
    vz = torch.where(valid, z, torch.full((), float("nan"), dtype=z.dtype, device=z.device))
    return torch.stack([X, Y], -1).contiguous(), vz.contiguous(), valid


def _prepare(verts, faces, normals, H, W) -> _Mesh:
    verts, faces, normals, batched = _batch(verts, faces, normals)
    vxy, vz, valid = _project(verts, H, W)
    faces = faces.to(torch.int32).contiguous()
    if normals is not None:
        normals = normals.to(device=vz.device, dtype=torch.float32)
    return _Mesh(vxy, vz, faces, normals, valid, batched)


def _use_cuda(backend: str, device: torch.device) -> bool:
    if backend == "torch":
        return False
    if backend == "cuda":
        if device.type != "cuda":
            raise ValueError("backend='cuda' needs CUDA tensors")
        _ext.load()
        return True
    if backend != "auto":
        raise ValueError(f"backend must be 'auto', 'cuda' or 'torch', got {backend!r}")
    if device.type != "cuda":
        return False
    if _ext.available():
        return True
    global _warned_fallback
    if not _warned_fallback:
        _warned_fallback = True
        warnings.warn(
            f"FOF CUDA extension unavailable ({_ext.last_error()!r}); using the slower PyTorch backend.",
            RuntimeWarning,
            stacklevel=3,
        )
    return False


_warned_fallback = False


def _warn_invalid(mesh: _Mesh):
    if not bool(mesh.valid.all()):
        n = int((~mesh.valid).sum())
        warnings.warn(
            f"{n} vertices are non-finite or too far outside the image; faces using them are skipped.",
            RuntimeWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Fragment keys
# ---------------------------------------------------------------------------


def _make_key_signed(depth: torch.Tensor, is_exit: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
    """Same packing as make_key() in the CUDA code, but returned in a signed,
    order-preserving form (uint64 key - 2^63) so int64 sorting works."""
    u = depth.contiguous().view(torch.int32).long() & 0xFFFFFFFF
    ordv = torch.where(u >= 0x80000000, (~u) & 0xFFFFFFFF, u | 0x80000000)
    return (ordv - 0x80000000) * (1 << 32) + is_exit.long() * (1 << 31) + face


def decode_keys(keys: torch.Tensor):
    """uint64 fragment keys (stored as int64) -> (depth float32, is_exit bool, face int64)."""
    ks = keys ^ _INT64_MIN
    ordv = (ks >> 32) + 0x80000000
    is_exit = ((ks >> 31) & 1).bool()
    face = ks & 0x7FFFFFFF
    u = torch.where(ordv >= 0x80000000, ordv - 0x80000000, (~ordv) & 0xFFFFFFFF)
    depth = torch.where(u >= 0x80000000, u - (1 << 32), u).to(torch.int32).view(torch.float32)
    return depth, is_exit, face


# ---------------------------------------------------------------------------
# Pure PyTorch reference backend (CPU or GPU).  Bit-exact with the CUDA
# rasterizer; intended for testing and as a fallback, not for speed.
# ---------------------------------------------------------------------------


def _edge(ax, ay, bx, by, px, py):
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def _covers(e, dx, dy):
    return (e > 0) | ((e == 0) & ((dy < 0) | ((dy == 0) & (dx > 0))))


def _edge_depth(ax, ay, az, bx, by, bz, px, py):
    """Depth on segment a-b at p, endpoints in canonical order (see CUDA edge_depth)."""
    swap = (ax > bx) | ((ax == bx) & (ay > by))
    ax, bx = torch.where(swap, bx, ax), torch.where(swap, ax, bx)
    ay, by = torch.where(swap, by, ay), torch.where(swap, ay, by)
    az, bz = torch.where(swap, bz, az), torch.where(swap, az, bz)
    use_x = ax != bx
    num = torch.where(use_x, px - ax, py - ay)
    den = torch.where(use_x, bx - ax, by - ay)
    return ((az.double() * (den - num).double() + bz.double() * num.double()) / den.double()).float()


def _sample_depth(e0, e1, e2, x, y, z, area, px, py):
    """Mirrors sample_depth() of the CUDA kernel, bit for bit."""
    num = (e0.double() * z[0].double() + e1.double() * z[1].double()) + e2.double() * z[2].double()
    depth = (num / area.double()).float()
    on0, on1, on2 = e0 == 0, e1 == 0, e2 == 0
    for on, (a, b) in ((on2, (0, 1)), (on1, (2, 0)), (on0, (1, 2))):  # lowest priority first
        if bool(on.any()):
            k = on.nonzero().squeeze(1)
            depth[k] = _edge_depth(x[a][k], y[a][k], z[a][k], x[b][k], y[b][k], z[b][k], px[k], py[k])
    for both, v in (((on2 & on0), 1), ((on1 & on2), 0), ((on0 & on1), 2)):
        depth = torch.where(both, z[v], depth)
    return depth


def _rasterize_torch(vxy, vz, faces, H, W, max_candidates=1 << 22):
    device = vz.device
    B, V = vz.shape
    F = faces.shape[1]
    P = B * H * W
    if B * F == 0 or V == 0:
        return torch.zeros(P + 1, dtype=torch.int32, device=device), torch.empty(0, dtype=torch.long, device=device)

    fl = faces.long()
    ok = ((fl >= 0) & (fl < V)).all(-1)
    fl = fl.clamp(0, V - 1)
    bi = torch.arange(B, device=device).view(B, 1, 1)
    X = vxy[..., 0].long()[bi, fl]
    Y = vxy[..., 1].long()[bi, fl]
    Z = vz[bi, fl]
    ok &= torch.isfinite(Z).all(-1)
    x0, x1, x2 = X.reshape(-1, 3).unbind(-1)
    y0, y1, y2 = Y.reshape(-1, 3).unbind(-1)
    z0, z1, z2 = Z.reshape(-1, 3).unbind(-1)
    ok = ok.reshape(-1)
    area = _edge(x0, y0, x1, y1, x2, y2)
    ok &= area != 0
    is_exit = area < 0  # camera-facing in the y-down pixel frame
    x1, x2 = torch.where(is_exit, x2, x1), torch.where(is_exit, x1, x2)
    y1, y2 = torch.where(is_exit, y2, y1), torch.where(is_exit, y1, y2)
    z1, z2 = torch.where(is_exit, z2, z1), torch.where(is_exit, z1, z2)
    area = area.abs()
    one = _SUB - 1
    i0 = ((torch.minimum(torch.minimum(x0, x1), x2) + one) >> SUBPIXEL_BITS).clamp(min=0)
    i1 = (torch.maximum(torch.maximum(x0, x1), x2) >> SUBPIXEL_BITS).clamp(max=W - 1)
    j0 = ((torch.minimum(torch.minimum(y0, y1), y2) + one) >> SUBPIXEL_BITS).clamp(min=0)
    j1 = (torch.maximum(torch.maximum(y0, y1), y2) >> SUBPIXEL_BITS).clamp(max=H - 1)
    ok &= (i1 >= i0) & (j1 >= j0)
    cnt = torch.where(ok, (i1 - i0 + 1) * (j1 - j0 + 1), torch.zeros_like(i0))

    tri_all = ok.nonzero().squeeze(1)
    cum = cnt[tri_all].cumsum(0)
    pix_chunks, key_chunks = [], []
    start, n_tri = 0, tri_all.numel()
    while start < n_tri:
        base = int(cum[start - 1]) if start > 0 else 0
        end = int(torch.searchsorted(cum, torch.tensor([base + max_candidates], device=device), right=True))
        end = max(end, start + 1)
        sel = tri_all[start:end]
        c = cnt[sel]
        start = end

        tri = torch.repeat_interleave(sel, c)
        excl = torch.cumsum(c, 0) - c
        local = torch.arange(tri.numel(), device=device) - torch.repeat_interleave(excl, c)
        bw = (i1 - i0 + 1)[tri]
        i = i0[tri] + local % bw
        j = j0[tri] + local // bw
        px, py = i << SUBPIXEL_BITS, j << SUBPIXEL_BITS
        X0, Y0, X1, Y1, X2, Y2 = x0[tri], y0[tri], x1[tri], y1[tri], x2[tri], y2[tri]
        e0 = _edge(X1, Y1, X2, Y2, px, py)
        e1 = _edge(X2, Y2, X0, Y0, px, py)
        e2 = _edge(X0, Y0, X1, Y1, px, py)
        cov = _covers(e0, X2 - X1, Y2 - Y1) & _covers(e1, X0 - X2, Y0 - Y2) & _covers(e2, X1 - X0, Y1 - Y0)
        keep = cov.nonzero().squeeze(1)
        tri, i, j, e0, e1, e2 = tri[keep], i[keep], j[keep], e0[keep], e1[keep], e2[keep]
        depth = _sample_depth(
            e0, e1, e2,
            (X0[keep], X1[keep], X2[keep]), (Y0[keep], Y1[keep], Y2[keep]),
            (z0[tri], z1[tri], z2[tri]), area[tri], px[keep], py[keep],
        )
        depth = torch.where(depth == 0, torch.zeros_like(depth), depth)
        b = tri // F
        pix_chunks.append((b * H + j) * W + i)
        key_chunks.append(_make_key_signed(depth, is_exit[tri], tri - b * F))

    if not pix_chunks:
        return torch.zeros(P + 1, dtype=torch.int32, device=device), torch.empty(0, dtype=torch.long, device=device)
    pix = torch.cat(pix_chunks)
    ks = torch.cat(key_chunks)
    if ks.numel() >= (1 << 31) - 1:
        raise RuntimeError("too many fragments")
    order = torch.argsort(ks, stable=True)
    order = order[torch.argsort(pix[order], stable=True)]
    pix, ks = pix[order], ks[order]
    ptr = torch.zeros(P + 1, dtype=torch.long, device=device)
    ptr[1:] = torch.bincount(pix, minlength=P).cumsum(0)
    return ptr.to(torch.int32), ks ^ _INT64_MIN


def _hybrid_indices(is_exit: torch.Tensor, ptr: torch.Tensor):
    """Fragment indices opening / closing the intervals of the "hybrid" rule: a scan over the
    position inside each pixel's list, vectorized across pixels (mirrors the CUDA kernel)."""
    device = is_exit.device
    cnt = ptr[1:] - ptr[:-1]
    pix = (cnt > 0).nonzero().squeeze(1)
    base, length = ptr[:-1][pix], cnt[pix]
    n = len(pix)
    zeros_l = torch.zeros(n, dtype=torch.long, device=device)
    false = torch.zeros(n, dtype=torch.bool, device=device)
    w, start, last, p0, p1, n_out = zeros_l, zeros_l, zeros_l, zeros_l, zeros_l, zeros_l
    has_exit, just_closed, pending = false, false, false
    opens, closes = [], []
    for k in range(int(length.max()) if n else 0):
        active = k < length
        idx = torch.where(active, base + k, base)
        ex = is_exit[idx] & active
        en = ~is_exit[idx] & active
        new_start = en & (w == 0)
        flush = new_start & pending
        opens.append(p0[flush])
        closes.append(p1[flush])
        pending = pending & ~flush
        start = torch.where(new_start, idx, start)
        has_exit = has_exit & ~new_start
        orphan = ex & (w == 0)
        dec = ex & (w > 0)
        w = w + en.long() - dec.long()
        just_closed = just_closed & ~en
        has_exit = has_exit | dec
        last = torch.where(dec, idx, last)
        close = dec & (w == 0)
        pending = pending | close
        p0 = torch.where(close, start, p0)
        p1 = torch.where(close | (orphan & just_closed), idx, p1)
        just_closed = just_closed | close
        n_out = n_out + close.long()
    opens.append(p0[pending])
    closes.append(p1[pending])
    tail = (w > 0) & has_exit
    opens.append(start[tail])
    closes.append(last[tail])
    n_out = n_out + tail.long()
    # even-odd fallback where nothing could be matched
    fb = (n_out == 0) & (length >= 2)
    npairs = length[fb] // 2
    j = torch.arange(int(npairs.sum()), device=device) - torch.repeat_interleave(torch.cumsum(npairs, 0) - npairs, npairs)
    first = torch.repeat_interleave(base[fb], npairs) + 2 * j
    opens.append(first)
    closes.append(first + 1)
    return torch.cat(opens), torch.cat(closes)


def _interval_indices(rule: int, is_exit: torch.Tensor, ptr: torch.Tensor, seg: torch.Tensor):
    """Vectorized fill rules: indices of the fragments that open / close each
    occupied interval.  Mirrors integrate_pixel() in the CUDA code."""
    n = is_exit.numel()
    device = is_exit.device
    ar = torch.arange(n, device=device)
    start = ptr[:-1][seg]
    last_idx = ptr[1:][seg] - 1
    first = ar == start
    last = ar == last_idx
    entry = ~is_exit

    if rule == FILL_RULES["parity"]:
        length = last_idx - start + 1
        keep = ~(last & (length % 2 == 1))
        idx = keep.nonzero().squeeze(1)
    elif rule == FILL_RULES["robust"]:
        prev_exit = torch.ones_like(is_exit)
        prev_exit[1:] = is_exit[:-1]
        prev_exit |= first
        next_exit = torch.zeros_like(is_exit)
        next_exit[:-1] = is_exit[1:]
        next_exit &= ~last
        e_excl = torch.cumsum(entry.long(), 0) - entry.long()
        ent_before = e_excl - e_excl[start]
        x_cum = torch.cumsum(is_exit.long(), 0)
        exits_after = x_cum[last_idx] - x_cum
        kept_entry = entry & prev_exit & (exits_after > 0)
        kept_exit = is_exit & (ent_before > 0) & ~next_exit
        idx = (kept_entry | kept_exit).nonzero().squeeze(1)
    elif rule == FILL_RULES["winding"]:
        # Nonzero rule: winding number w (entries - exits) per segment.
        delta = torch.where(is_exit, -1, 1).long()
        s_glob = torch.cumsum(delta, 0)
        w = s_glob - (s_glob - delta)[start]
        w_prev = w - delta
        starts = (w_prev == 0) & (w != 0)
        ends = (w_prev != 0) & (w == 0)
        s_cum = torch.cumsum(starts.long(), 0)
        unclosed = starts & (w[last_idx] != 0) & (s_cum[last_idx] == s_cum)
        idx = ((starts & ~unclosed) | ends).nonzero().squeeze(1)
    elif rule == FILL_RULES["hybrid"]:
        return _hybrid_indices(is_exit, ptr)
    else:
        raise ValueError(f"unknown fill rule {rule}")
    return idx[0::2], idx[1::2]


def _fof_torch(frags: Fragments, C: int, rule: int, z_min: float, z_max: float, chunk=1 << 16):
    B, H, W = frags.shape
    P = B * H * W
    device = frags.keys.device
    out = torch.zeros(P, C, dtype=torch.float64, device=device)
    if frags.keys.numel() > 0:
        depth, is_exit, _ = decode_keys(frags.keys)
        ptr = frags.pix_ptr.long()
        seg = torch.repeat_interleave(torch.arange(P, device=device), ptr[1:] - ptr[:-1])
        z0 = torch.tensor(z_min, dtype=torch.float32, device=device)
        inv = torch.tensor(1.0 / (z_max - z_min), dtype=torch.float32, device=device)
        t = ((depth - z0) * inv).clamp(0.0, 1.0)
        s_idx, e_idx = _interval_indices(rule, is_exit, ptr, seg)
        t0, t1, pix = t[s_idx].double(), t[e_idx].double(), seg[s_idx]
        m, h = 0.5 * (t0 + t1), 0.5 * (t1 - t0)
        c = torch.arange(C, device=device, dtype=torch.float64)
        for k in range(0, pix.numel(), chunk):
            sl = slice(k, k + chunk)
            term = torch.cos(math.pi * m[sl, None] * c) * torch.sin(math.pi * h[sl, None] * c)
            term[:, 0] = h[sl]
            out.index_add_(0, pix[sl], term)
        scale = torch.empty(C, dtype=torch.float64, device=device)
        scale[0] = 4.0
        scale[1:] = 4.0 / (math.pi * c[1:])
        out *= scale
    return out.float().view(B, H, W, C).permute(0, 3, 1, 2).contiguous()


def _front_back_torch(frags: Fragments, vxy: torch.Tensor, faces: torch.Tensor):
    B, H, W = frags.shape
    P = B * H * W
    device = frags.keys.device
    face = torch.full((2, P), -1, dtype=torch.int32, device=device)
    depth = torch.full((2, P), float("nan"), dtype=torch.float32, device=device)
    bary = torch.zeros((2, P, 3), dtype=torch.float32, device=device)
    ptr = frags.pix_ptr.long()
    pix = ((ptr[1:] - ptr[:-1]) > 0).nonzero().squeeze(1)
    if pix.numel() > 0:
        b = pix // (H * W)
        hw = pix % (H * W)
        px = (hw % W) << SUBPIXEL_BITS
        py = (hw // W) << SUBPIXEL_BITS
        for s, idx in enumerate((ptr[1:][pix] - 1, ptr[:-1][pix])):
            d, _, f = decode_keys(frags.keys[idx])
            fc = faces[b, f].long()
            X = vxy[b[:, None], fc, 0].long()
            Y = vxy[b[:, None], fc, 1].long()
            area = _edge(X[:, 0], Y[:, 0], X[:, 1], Y[:, 1], X[:, 2], Y[:, 2]).double()
            e = torch.stack(
                [
                    _edge(X[:, 1], Y[:, 1], X[:, 2], Y[:, 2], px, py),
                    _edge(X[:, 2], Y[:, 2], X[:, 0], Y[:, 0], px, py),
                    _edge(X[:, 0], Y[:, 0], X[:, 1], Y[:, 1], px, py),
                ],
                -1,
            ).double()
            face[s, pix] = f.to(torch.int32)
            depth[s, pix] = d
            bary[s, pix] = (e / area[:, None]).float()
    return face.view(2, B, H, W), depth.view(2, B, H, W), bary.view(2, B, H, W, 3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _rasterize_prepared(mesh: _Mesh, H: int, W: int, use_cuda: bool) -> Fragments:
    B = mesh.vz.shape[0]
    if use_cuda:
        pix_ptr, keys = _ext.load().rasterize(mesh.vxy, mesh.vz, mesh.faces, H, W)
    else:
        pix_ptr, keys = _rasterize_torch(mesh.vxy, mesh.vz, mesh.faces, H, W)
    _warn_invalid(mesh)
    return Fragments(pix_ptr, keys, (B, H, W))


@torch.no_grad()
def rasterize(verts, faces=None, resolution: Resolution = 512, *, backend: str = "auto") -> Fragments:
    """Exact multi-layer rasterization: every ray/surface crossing of every pixel.

    Args:
        verts: ``[B, V, 3]`` / ``[V, 3]`` vertices (with ``faces``), a triangle
            soup ``[B, F, 3, 3]`` / ``[F, 3, 3]`` (``faces=None``), or a list of
            ``[V_i, 3]`` tensors (with a list of faces).  Normalized coordinates.
        faces: ``[F, 3]`` shared or ``[B, F, 3]`` per-mesh vertex indices.
            Out-of-range indices (e.g. -1 padding) and zero-area faces are skipped.
        resolution: ``H`` or ``(H, W)``.
        backend: ``"auto"`` (CUDA extension for CUDA tensors), ``"cuda"`` or ``"torch"``.
    """
    H, W = _hw(resolution)
    mesh = _prepare(verts, faces, None, H, W)
    return _rasterize_prepared(mesh, H, W, _use_cuda(backend, mesh.vz.device))


@torch.no_grad()
def fragments_to_fof(
    frags: Fragments,
    num_coeffs: int = 128,
    *,
    fill: str = "hybrid",
    z_range: Tuple[float, float] = (-1.0, 1.0),
    backend: str = "auto",
) -> torch.Tensor:
    """Integrates the cosine basis over the occupied intervals -> ``[B, C, H, W]``."""
    if fill not in FILL_RULES:
        raise ValueError(f"fill must be one of {sorted(FILL_RULES)}, got {fill!r}")
    num_coeffs = int(num_coeffs)
    if num_coeffs <= 0:
        raise ValueError("num_coeffs must be positive")
    z_min, z_max = (float(z) for z in z_range)
    if not z_max > z_min:
        raise ValueError("z_range must be increasing")
    B, H, W = frags.shape
    if _use_cuda(backend, frags.keys.device):
        return _ext.load().fof_integrate(frags.pix_ptr, frags.keys, B, H, W, num_coeffs, z_min, z_max, FILL_RULES[fill])
    return _fof_torch(frags, num_coeffs, FILL_RULES[fill], z_min, z_max)


@torch.no_grad()
def mesh_to_fof(
    verts,
    faces=None,
    num_coeffs: int = 128,
    resolution: Resolution = 512,
    *,
    fill: str = "hybrid",
    z_range: Tuple[float, float] = (-1.0, 1.0),
    backend: str = "auto",
) -> torch.Tensor:
    """Encodes meshes as Fourier Occupancy Fields.

    Returns ``[B, num_coeffs, H, W]`` float32 (``[num_coeffs, H, W]`` for an
    unbatched mesh).  See :func:`rasterize` for the accepted mesh layouts.

    Fill rules (how sorted crossings become occupied intervals):
      * ``"hybrid"`` (default): winding number clamped at zero (exact union of nested or
        interpenetrating parts), exit runs merged into the preceding interval (duplicated layers),
        an interval still open at the end of the ray closed at its last exit, and the even-odd
        rule as a fallback when nothing can be matched (flipped orientation).  Lowest error of
        the four rules on corrupted human solids (holes, flips, duplicated layers, overlaps).
      * ``"robust"``: the original FOF rule: an interval opens at the first entry seen from
        outside and closes at the last exit before the next entry.  Use it to reproduce the
        ground truth of existing checkpoints.
      * ``"winding"``: nonzero rule on the winding number (entries - exits).
        Exact union of overlapping / self-intersecting closed parts.
      * ``"parity"``: even-odd rule; ignores face orientation entirely.
    No rule can recover the intended solid from meshes with holes; repair such meshes offline.
    Except for "hybrid", an interval still open at the end of the ray is dropped; intervals are
    clipped to ``z_range`` before integration.
    """
    H, W = _hw(resolution)
    mesh = _prepare(verts, faces, None, H, W)
    use_cuda = _use_cuda(backend, mesh.vz.device)
    frags = _rasterize_prepared(mesh, H, W, use_cuda)
    out = fragments_to_fof(frags, num_coeffs, fill=fill, z_range=z_range, backend="cuda" if use_cuda else "torch")
    return out if mesh.batched else out[0]


@torch.no_grad()
def ray_diagnostics(verts, faces=None, resolution: Resolution = 512, *, backend: str = "auto") -> dict:
    """Data-quality check for meshes used as FOF ground truth.

    Returns per-pixel maps ``[B, H, W]``: ``crossings`` (number of surface crossings),
    ``unbalanced`` (entries != exits: holes or flipped orientation along the ray) and
    ``rules_disagree`` (the fill rules produce different occupied lengths), plus the
    per-mesh fractions of covered pixels affected (``frac_unbalanced``, ``frac_disagree``).
    ``unbalanced`` is zero on closed, consistently oriented meshes; ``rules_disagree`` is also
    set by nested or interpenetrating parts (the even-odd rule makes cavities of them).
    """
    H, W = _hw(resolution)
    mesh = _prepare(verts, faces, None, H, W)
    use_cuda = _use_cuda(backend, mesh.vz.device)
    frags = _rasterize_prepared(mesh, H, W, use_cuda)
    B = frags.shape[0]
    P = B * H * W
    ptr = frags.pix_ptr.long()
    counts = ptr[1:] - ptr[:-1]
    _, is_exit, _ = decode_keys(frags.keys)
    seg = torch.repeat_interleave(torch.arange(P, device=counts.device), counts)
    balance = torch.zeros(P, dtype=torch.long, device=counts.device).index_add_(0, seg, 1 - 2 * is_exit.long())
    be = "cuda" if use_cuda else "torch"
    a0 = torch.stack([fragments_to_fof(frags, 1, fill=r, backend=be)[:, 0] for r in FILL_RULES])
    disagree = (a0.amax(0) - a0.amin(0)) > 1e-6
    covered = (counts > 0).view(B, H, W)
    n_cov = covered.flatten(1).sum(1).clamp(min=1)
    out = {
        "crossings": counts.view(B, H, W),
        "unbalanced": (balance != 0).view(B, H, W),
        "rules_disagree": disagree,
        "frac_unbalanced": (balance != 0).view(B, H * W).sum(1) / n_cov,
        "frac_disagree": disagree.flatten(1).sum(1) / n_cov,
    }
    if not mesh.batched:
        out = {k: v[0] for k, v in out.items()}
    return out


def _interpolate(attr: torch.Tensor, faces: torch.Tensor, face: torch.Tensor, bary: torch.Tensor) -> torch.Tensor:
    """Per-vertex attributes [B, V, K] at the rendered surface -> [B, K, H, W]
    (zero where no surface is hit)."""
    B, H, W = face.shape
    out = attr.new_zeros(B, H, W, attr.shape[-1])
    hit = face >= 0
    if bool(hit.any()):
        b, h, w = hit.nonzero(as_tuple=True)
        corners = faces[b, face[b, h, w].long()].long()  # [n, 3], always valid for hit pixels
        out[b, h, w] = (attr[b[:, None], corners] * bary[b, h, w][..., None]).sum(1)
    return out.permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def render_fof(
    verts,
    faces=None,
    normals=None,
    num_coeffs: Optional[int] = 128,
    resolution: Resolution = 512,
    *,
    fill: str = "hybrid",
    z_range: Tuple[float, float] = (-1.0, 1.0),
    background_depth: Tuple[float, float] = (-1.0, 1.0),
    backend: str = "auto",
) -> FOFRender:
    """FOF plus front/back depth and normal maps from a single rasterization.

    ``normals`` are per-vertex ``[B, V, 3]`` (or per-corner ``[B, F, 3, 3]`` for
    a triangle soup); the interpolated normals are re-normalized.  Pass
    ``num_coeffs=None`` to skip the FOF.  ``background_depth`` fills
    (front, back) depth where no surface is hit.  Depth/normals describe the
    raw nearest/farthest surfaces, independent of the fill rule.
    """
    H, W = _hw(resolution)
    mesh = _prepare(verts, faces, normals, H, W)
    use_cuda = _use_cuda(backend, mesh.vz.device)
    frags = _rasterize_prepared(mesh, H, W, use_cuda)
    be = "cuda" if use_cuda else "torch"
    fof = None
    if num_coeffs:
        fof = fragments_to_fof(frags, num_coeffs, fill=fill, z_range=z_range, backend=be)
    if use_cuda:
        face, depth, bary = _ext.load().front_back(frags.pix_ptr, frags.keys, mesh.vxy, mesh.faces, H, W)
    else:
        face, depth, bary = _front_back_torch(frags, mesh.vxy, mesh.faces)
    mask = face[0] >= 0
    depth_front = torch.where(mask, depth[0], torch.full_like(depth[0], background_depth[0]))
    depth_back = torch.where(mask, depth[1], torch.full_like(depth[1], background_depth[1]))
    normal_front = normal_back = None
    if mesh.normals is not None:
        nrm = []
        for s in range(2):
            n = _interpolate(mesh.normals, mesh.faces, face[s], bary[s])
            length = n.norm(dim=1, keepdim=True)
            nrm.append(torch.where(length > 0, n / length.clamp_min(1e-12), torch.zeros_like(n)))
        normal_front, normal_back = nrm
    out = FOFRender(
        fof, mask, depth_front, depth_back, normal_front, normal_back,
        face[0].long(), face[1].long(), bary[0], bary[1],
    )
    if not mesh.batched:
        out = FOFRender(*(None if x is None else x[0] for x in out))
    return out
