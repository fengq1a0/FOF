"""Fourier Occupancy Field -> occupancy volume -> triangle mesh.

The occupancy of pixel (h, w) at normalized depth t in [0, 1] is
``o(t) = a_0 / 2 + sum_{c>=1} a_c cos(c*pi*t)`` (see ``encode.py``).  Depth
samples sit at voxel centres ``t_d = (d + 0.5) / D``.

Iso-surfacing runs either on the GPU (``lib.fof.mc``: crack-free marching cubes with generated
tables, milliseconds at 512^3) or with scikit-image's Lewiner marching cubes on the CPU.

"Lossless" (FOF-X) decoding keeps the vertices that lie on depth-axis grid edges (exact for each
pixel's 1D occupancy, optionally Newton-refined onto the exact root of the cosine series) and
re-solves the in-between vertices with the least-squares uniform-Laplacian energy over all
vertices, i.e. a discrete thin-plate interpolation of the per-column surface samples.  The
default solver runs a fixed number of Jacobi-PCG iterations from the marching-cubes positions:
in our measurements 20-50 iterations match or beat the exact minimizer (early stopping
regularizes the ill-conditioned low-frequency / tangential modes), and it is GPU real-time.
"""

import math
import numbers
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

Resolution = Union[int, Tuple[int, int]]


def fof_basis(num_coeffs: int, depth_res: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Cosine basis sampled at the depth voxel centres, ``[depth_res, num_coeffs]``."""
    t = (torch.arange(depth_res, dtype=torch.float64) + 0.5) / depth_res
    c = torch.arange(num_coeffs, dtype=torch.float64)
    basis = torch.cos(math.pi * t[:, None] * c[None, :])
    basis[:, 0] = 0.5
    return basis.to(device=device, dtype=dtype)


def _size(resolution: Resolution) -> Tuple[int, int]:
    if isinstance(resolution, numbers.Integral):
        return int(resolution), int(resolution)
    return tuple(int(r) for r in resolution)


def _resize(fof: torch.Tensor, resolution: Optional[Resolution]) -> torch.Tensor:
    if resolution is None or tuple(fof.shape[-2:]) == _size(resolution):
        return fof
    # The basis is linear in the coefficients, so bilinear resizing of the
    # coefficient images equals trilinear upsampling of the volume in x/y while
    # keeping the depth axis exact.
    return F.interpolate(fof, size=_size(resolution), mode="bilinear", align_corners=False)


@torch.no_grad()
def fof_to_occupancy(fof: torch.Tensor, depth_res: int = 512, resolution: Optional[Resolution] = None) -> torch.Tensor:
    """Evaluates the occupancy on a ``[D, H, W]`` grid (float32, autocast disabled).

    Args:
        fof: ``[C, H, W]`` or ``[B, C, H, W]`` coefficients.
        depth_res: number of depth samples ``D``.
        resolution: optional output ``H`` or ``(H, W)``; coefficients are resized bilinearly.
    Returns ``[D, H, W]`` or ``[B, D, H, W]``.
    """
    batched = fof.dim() == 4
    x = fof if batched else fof[None]
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = _resize(x.float(), resolution)
        basis = fof_basis(x.shape[1], depth_res, device=x.device, dtype=torch.float32)
        occ = torch.einsum("dc,bchw->bdhw", basis, x)
    return occ if batched else occ[0]


# ---------------------------------------------------------------------------
# Marching cubes and coordinates
# ---------------------------------------------------------------------------


def _pad_value(level):
    """The "empty space" around a closed volume: a float32 value strictly below the level."""
    if level > 0:
        return 0.0
    pad = np.float32(float(level) - 1.0)
    while float(pad) >= level:  # |level| >= 2^24: level - 1 can round back up to the level
        pad = np.nextafter(pad, np.float32(-np.inf))
    return float(pad)


def _mc_skimage(occ, level, close, allow_degenerate):
    """skimage Lewiner MC -> (verts (d, h, w) ndarray, faces ndarray) or None."""
    from skimage.measure import marching_cubes

    if torch.is_tensor(occ):
        if close:
            occ = F.pad(occ.float(), (1, 1, 1, 1, 1, 1), value=_pad_value(level))
        vol = occ.float().cpu().numpy()
    else:
        vol = np.asarray(occ, dtype=np.float32)
        if close:
            vol = np.pad(vol, 1, mode="constant", constant_values=_pad_value(level))
    if vol.size == 0 or not (vol.min() <= level <= vol.max()) or vol.min() == vol.max():
        return None
    try:
        verts, faces, _, _ = marching_cubes(vol, level=level, allow_degenerate=allow_degenerate)
    except (ValueError, RuntimeError):
        return None
    if close:
        verts = verts - 1.0
    return verts, faces.astype(np.int64)


def _mc_gpu(occ, level, close, backend):
    """Crack-free MC (lib.fof.mc) -> (vertex_ids, verts (d, h, w), faces, vertex-id grid shape) or
    None.  ``close`` uses the kernels' virtual border instead of a padded copy of the volume."""
    from .mc import marching_cubes

    occ = torch.as_tensor(occ).float()
    ids, verts, faces = marching_cubes(occ, level, pad=_pad_value(level) if close else None, backend=backend)
    if len(faces) == 0:
        return None
    o = 2 if close else 0
    return ids, verts, faces, tuple(n + o for n in occ.shape)


def _to_world(verts_idx, D, H, W, z_range):
    """(d, h, w) grid indices -> normalized (x, y, z) float32 (numpy array or tensor)."""
    z_min, z_max = (float(z) for z in z_range)
    if torch.is_tensor(verts_idx):
        v = verts_idx.double()
        out = torch.stack([(v[:, 2] + 0.5) * (2.0 / W) - 1.0, 1.0 - (v[:, 1] + 0.5) * (2.0 / H),
                           z_min + (v[:, 0] + 0.5) * ((z_max - z_min) / D)], 1)
        return out.float()
    d, h, w = (verts_idx[:, k].astype(np.float64) for k in range(3))
    return np.stack(
        [(w + 0.5) * (2.0 / W) - 1.0, 1.0 - (h + 0.5) * (2.0 / H), z_min + (d + 0.5) * ((z_max - z_min) / D)],
        axis=1,
    ).astype(np.float32)


def _outward(faces):
    # (d, h, w) -> (x, y, z) = (w, -h, d) is a proper rotation; both marching cubes produce
    # triangles that are clockwise around outward normals in (d, h, w), hence the swap.
    if torch.is_tensor(faces):
        return faces[:, [0, 2, 1]].contiguous()
    return np.ascontiguousarray(faces[:, [0, 2, 1]])


def _empty_mesh(tensors=False, device=None):
    if tensors:
        return torch.zeros(0, 3, device=device), torch.zeros(0, 3, dtype=torch.long, device=device)
    return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64)


def _use_gpu_mc(backend, device):
    from . import _ext

    if backend == "skimage":
        return False
    if backend in ("cuda", "torch"):
        return True
    if backend != "auto":
        raise ValueError(f"backend must be 'auto', 'cuda', 'torch' or 'skimage', got {backend!r}")
    return device.type == "cuda" and _ext.available()


@torch.no_grad()
def volume_to_mesh(
    occ,
    level: float = 0.5,
    *,
    close: bool = True,
    z_range: Tuple[float, float] = (-1.0, 1.0),
    allow_degenerate: bool = False,
    backend: str = "auto",
    return_tensors: bool = False,
):
    """Iso-surface of an occupancy volume ``[D, H, W]`` laid out like
    :func:`fof_to_occupancy` (tensor or array) -> ``(verts, faces)`` in
    normalized coordinates with outward faces.  ``backend``: "auto" (GPU marching
    cubes for CUDA tensors, else scikit-image), "cuda", "torch" or "skimage"."""
    D, H, W = occ.shape
    device = occ.device if torch.is_tensor(occ) else torch.device("cpu")
    if _use_gpu_mc(backend, device):
        mc = _mc_gpu(occ, level, close, "auto" if backend == "auto" else backend)
        if mc is None:
            return _empty_mesh(return_tensors, device)
        v, f = _to_world(mc[1], D, H, W, z_range), _outward(mc[2])
        return (v, f) if return_tensors else (v.cpu().numpy(), f.cpu().numpy())
    mc = _mc_skimage(occ, level, close, allow_degenerate)
    if mc is None:
        return _empty_mesh(return_tensors, device)
    v, f = _to_world(mc[0], D, H, W, z_range), _outward(mc[1])
    return (torch.from_numpy(v).to(device), torch.from_numpy(f).to(device)) if return_tensors else (v, f)


# ---------------------------------------------------------------------------
# Lossless decoding
# ---------------------------------------------------------------------------


def _depth_edge_vertices(verts_idx: np.ndarray) -> np.ndarray:
    """skimage vertices lying on depth-axis grid edges (exactly integral h and w).

    Marching cubes interpolates only along the edge it places a vertex on, so these
    coordinates are exact integers for depth-edge vertices.  (The GPU marching cubes
    reports each vertex's edge explicitly, see ``lib.fof.mc.vertex_axis``.)
    """
    h, w = verts_idx[:, 1], verts_idx[:, 2]
    return (h == np.round(h)) & (w == np.round(w))


def _refine_roots(verts_idx, fixed, fof_hw, level, depth_res, iters=30):
    """Moves depth-edge vertices onto the exact root of the cosine series along their pixel's
    ray: safeguarded Newton / bisection inside the marching-cubes edge, keeping the iterate with
    the smallest residual.  ``verts_idx``/``fixed`` are numpy arrays or tensors."""
    as_numpy = not torch.is_tensor(verts_idx)
    dev = fof_hw.device
    v = torch.as_tensor(verts_idx).to(dev)
    fx = torch.as_tensor(fixed).to(dev)
    D = depth_res
    C, H, W = fof_hw.shape
    d = v[:, 0].double()
    d0 = torch.floor(d)
    h = torch.round(v[:, 1]).long()
    w = torch.round(v[:, 2]).long()
    sel = fx & (d0 >= 0) & (d0 + 1 <= D - 1) & (h >= 0) & (h < H) & (w >= 0) & (w < W)  # skip padding caps
    idx = sel.nonzero().squeeze(1)
    if idx.numel() == 0:
        return verts_idx
    a = fof_hw[:, h[idx], w[idx]].T.double().clone()  # [n, C]
    a[:, 0] *= 0.5
    c = torch.arange(C, device=dev, dtype=torch.float64)

    def f(t):
        return (a * torch.cos(math.pi * t[:, None] * c)).sum(1) - level

    def df(t):
        return -(a * (math.pi * c) * torch.sin(math.pi * t[:, None] * c)).sum(1)

    lo = (d0[idx] + 0.5) / D
    hi = lo + 1.0 / D
    f_lo, f_hi = f(lo), f(hi)
    bracketed = (f_lo * f_hi) <= 0
    t = ((d[idx] + 0.5) / D).clamp(lo, hi)
    ft = f(t)
    best_t, best_f = t.clone(), ft.abs()
    for cand_t, cand_f in ((lo, f_lo), (hi, f_hi)):  # roots exactly at an end point
        take = cand_f.abs() < best_f
        best_t, best_f = torch.where(take, cand_t, best_t), torch.where(take, cand_f.abs(), best_f)
    done = ~bracketed | (best_f == 0)
    for _ in range(iters):
        dft = df(t)
        newton = t - ft / torch.where(dft != 0, dft, torch.ones_like(dft))
        use_newton = (dft != 0) & (newton > lo) & (newton < hi)
        t_new = torch.where(use_newton, newton, 0.5 * (lo + hi))
        f_new = f(t_new)
        left = (f_lo * f_new) <= 0  # root in [lo, t_new]
        hi = torch.where(done, hi, torch.where(left, t_new, hi))
        lo = torch.where(done, lo, torch.where(left, lo, t_new))
        f_lo = torch.where(done | left, f_lo, f_new)
        better = ~done & (f_new.abs() < best_f)
        best_t, best_f = torch.where(better, t_new, best_t), torch.where(better, f_new.abs(), best_f)
        converged = use_newton & ((t_new - t).abs() <= 1e-15)
        t, ft = torch.where(done, t, t_new), torch.where(done, ft, f_new)
        done |= converged | (best_f == 0) | ((hi - lo) <= 1e-15)
    out = v.clone()
    keep = idx[bracketed]
    out[keep, 0] = (best_t[bracketed] * D - 0.5).to(out.dtype)
    return out.cpu().numpy() if as_numpy else out.to(torch.as_tensor(verts_idx).device)


def _pcg(matvec, b, x, diag_inv, tol, max_iter):
    """Jacobi-preconditioned CG for several right-hand sides (columns) at once, on numpy
    arrays or torch tensors.  ``tol=None`` runs exactly ``max_iter`` iterations without any
    convergence test (no host synchronization on CUDA).  Returns ``(x, converged)``, where
    convergence is checked on the true relative residual."""
    bnorm = ((b * b).sum(0)) ** 0.5 + 1e-30

    def converged(res):
        return bool(((((res * res).sum(0)) ** 0.5) <= tol * bnorm).all())

    r = b - matvec(x)
    z = r * diag_inv
    p = 1.0 * z
    rz = (r * z).sum(0)
    for _ in range(max_iter):
        if tol is not None and converged(r):
            r = b - matvec(x)  # the recursive residual drifts: confirm with the true one
            if converged(r):
                break
            z = r * diag_inv
            p = 1.0 * z
            rz = (r * z).sum(0)
        Ap = matvec(p)
        pAp = (p * Ap).sum(0)
        alpha = rz / (pAp + (pAp <= 0)) * (pAp > 0)
        x = x + alpha * p
        r = r - alpha * Ap
        z = r * diag_inv
        rz_new = (r * z).sum(0)
        beta = rz_new / (rz + (rz <= 0)) * (rz > 0)
        p = z + beta * p
        rz = rz_new
    return x, (True if tol is None else converged(b - matvec(x)))


def _anchored_free(free, fixed, e, V):
    """Free vertices whose free-vertex component touches a fixed vertex (numpy, scipy)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    ff = free[e[:, 0]] & free[e[:, 1]]
    graph = sp.coo_matrix((np.ones(int(ff.sum())), (e[ff, 0], e[ff, 1])), shape=(V, V))
    _, label = connected_components(graph, directed=False)
    anchored = np.zeros(label.max() + 1, dtype=bool)
    anchored[label[e[free[e[:, 0]] & fixed[e[:, 1]], 0]]] = True  # free vertex next to a fixed one
    anchored[label[e[free[e[:, 1]] & fixed[e[:, 0]], 1]]] = True
    return free & anchored[label]


def _ls_matrix_free(verts: torch.Tensor, faces: torch.Tensor, fixed: torch.Tensor, iterations: int,
                    check_anchors: bool = True, fidelity: float = 0.0) -> torch.Tensor:
    """FOF-X least-squares energy ``sum_i |L_i x|^2`` (rows for every vertex, depth-edge vertices
    fixed), ``iterations`` Jacobi-PCG steps from the marching-cubes positions, on ``verts``'
    device with gathers/scatters only (no matrix assembly, no host synchronization inside PCG).

    Free components without any anchor are left untouched: their energy has a null space (and
    open ones would shrink).  With ``check_anchors`` they are found on the CPU; pass False when
    the volume was padded (``close=True``): every component of a closed marching-cubes surface
    has a depth-edge vertex (its topmost vertex lies on a depth edge, or the loop above it
    reaches a depth edge or a higher vertex), so the system is positive definite as is.
    ``fidelity`` optionally adds ``fidelity * |x - x_mc|^2`` on the free vertices."""
    V = len(verts)
    dev = verts.device
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = torch.sort(e, dim=1).values
    e = e[e[:, 0] != e[:, 1]]
    e = torch.unique(e, dim=0)
    src = torch.cat([e[:, 0], e[:, 1]])
    dst = torch.cat([e[:, 1], e[:, 0]])
    deg = torch.bincount(src, minlength=V).double()
    has_nb = (deg > 0).double()[:, None]
    inv_deg = torch.where(deg > 0, 1.0 / deg.clamp(min=1), torch.zeros_like(deg))[:, None]
    free = ~fixed & (deg > 0)
    if check_anchors:
        free = torch.from_numpy(_anchored_free(free.cpu().numpy(), fixed.cpu().numpy(), e.cpu().numpy(), V)).to(dev)
    fidx = free.nonzero().squeeze(1)
    if fidx.numel() == 0:
        return verts

    def lap(u):  # L u = mean of neighbours - u (rows of isolated vertices are zero)
        return torch.zeros_like(u).index_add_(0, src, u[dst]) * inv_deg - u * has_nb

    def lap_t(y):  # L^T y
        return torch.zeros_like(y).index_add_(0, src, y[dst] * inv_deg[dst]) - y * has_nb

    def embed(t):
        u = torch.zeros(V, t.shape[1], dtype=t.dtype, device=dev)
        u[fidx] = t
        return u

    x = verts.double()
    known = torch.where(free[:, None], torch.zeros_like(x), x)
    x0 = x[fidx]
    b = -lap_t(lap(known))[fidx] + fidelity * x0
    diag = (torch.zeros(V, dtype=torch.float64, device=dev).index_add_(0, src, inv_deg[dst, 0] ** 2) + has_nb[:, 0])[fidx]
    t, _ = _pcg(lambda t: lap_t(lap(embed(t)))[fidx] + fidelity * t, b, x0, (1.0 / (diag + fidelity))[:, None],
                None, iterations)
    out = verts.clone()
    out[fidx] = t.to(verts.dtype)
    return out


def _anchored_smooth(verts, faces, fixed, *, solver="direct", tol=1e-7, max_iter=25, device="cpu", fidelity=0.0):
    """Same energy as :func:`_ls_matrix_free` with scipy (numpy arrays in and out); free
    components without anchors are left untouched.

    ``solver="direct"``: exact minimizer (sparse LU of the normal equations).
    ``solver="pcg"``: ``max_iter`` Jacobi-PCG iterations from the marching-cubes positions
    (``tol=None``) or until ``tol`` (``device`` may be CUDA); optional ``fidelity`` term.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    if solver not in ("direct", "pcg"):
        raise ValueError(f"solver must be 'direct' or 'pcg', got {solver!r}")
    V = verts.shape[0]
    if V == 0 or faces.shape[0] == 0:
        return verts
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).astype(np.int64)
    e = e[e[:, 0] != e[:, 1]]
    e = np.unique(np.sort(e, axis=1), axis=0)
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    deg = np.bincount(src, minlength=V)
    # unanchored components make the system singular (and would shrink): leave them as they are
    free = _anchored_free(~fixed & (deg > 0), fixed, e, V)
    if not free.any():
        return verts

    x_all = verts.astype(np.float64)
    L = (sp.csr_matrix((1.0 / deg[src], (src, dst)), shape=(V, V)) - sp.diags((deg > 0).astype(np.float64))).tocsc()
    A = L[:, free].tocsr()
    k = L[:, ~free] @ x_all[~free]
    M = (A.T @ A).tocsr()
    b = -(A.T @ k)
    if solver == "direct":
        try:
            x = spla.splu(M.tocsc(), permc_spec="COLAMD").solve(b)
        except (RuntimeError, MemoryError) as err:
            warnings.warn(f"lossless decoding: direct solve failed ({err}); keeping marching-cubes vertices",
                          RuntimeWarning)
            return verts
    else:
        if fidelity:
            M = (M + fidelity * sp.identity(M.shape[0], format="csr")).tocsr()
            b = b + fidelity * x_all[free]
        diag_inv = 1.0 / M.diagonal()[:, None]
        dev = torch.device(device)
        if dev.type == "cuda":
            with warnings.catch_warnings():  # "sparse CSR support is in beta"
                warnings.simplefilter("ignore", UserWarning)
                Mt = torch.sparse_csr_tensor(torch.from_numpy(M.indptr).long(), torch.from_numpy(M.indices).long(),
                                             torch.from_numpy(M.data), size=M.shape, device=dev)
            xt, ok = _pcg(lambda v: Mt @ v, torch.from_numpy(b).to(dev), torch.from_numpy(x_all[free]).to(dev),
                          torch.from_numpy(diag_inv).to(dev), tol, max_iter)
            x = xt.cpu().numpy()
        else:
            x, ok = _pcg(lambda v: M @ v, b, x_all[free], diag_inv, tol, max_iter)
        if not ok:
            warnings.warn("lossless decoding: PCG did not reach the tolerance", RuntimeWarning)
    if not np.isfinite(x).all():
        warnings.warn("lossless decoding: non-finite solution; keeping marching-cubes vertices", RuntimeWarning)
        return verts
    out = verts.copy()
    out[free] = x.astype(verts.dtype)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@torch.no_grad()
def fof_to_mesh(
    fof: torch.Tensor,
    depth_res: int = 512,
    resolution: Optional[Resolution] = None,
    level: float = 0.5,
    *,
    backend: str = "auto",
    lossless: bool = False,
    refine_roots: bool = False,
    solver: str = "pcg",
    iterations: int = 25,
    close: bool = True,
    z_range: Tuple[float, float] = (-1.0, 1.0),
    allow_degenerate: bool = False,
    return_tensors: bool = False,
):
    """Extracts the ``level`` iso-surface of a single FOF ``[C, H, W]``.

    Args:
        depth_res: number of depth samples of the evaluated volume.
        resolution: optional ``H`` or ``(H, W)`` to resample the coefficient images before
            evaluation (``None`` keeps the FOF resolution).
        backend: marching cubes: "auto" (GPU kernels for CUDA tensors, else scikit-image),
            "cuda", "torch" (vectorized PyTorch, any device) or "skimage" (Lewiner, CPU).
        lossless: FOF-X decoding: keep the depth-edge vertices (exact w.r.t. each pixel's 1D
            occupancy) and re-solve the between-pixel vertices with the anchored
            least-squares Laplacian (removes the staircase pattern).
        refine_roots: snap depth-edge vertices onto the exact root of the cosine series instead
            of the linear interpolation of marching cubes.
        solver, iterations: ``"pcg"`` (default) runs ``iterations`` Jacobi-PCG steps from the
            marching-cubes positions (matrix-free on the GPU path); ``"direct"`` computes the
            exact minimizer with scipy (CPU).
        close: surround the volume with empty space so the surface is closed where the object
            touches the image border or the depth range (a virtual border on the GPU path: no copy).
        allow_degenerate: keep zero-area triangles (scikit-image backend only).
        return_tensors: return tensors on the FOF's device instead of numpy arrays.
    Returns ``(verts [V, 3] float32, faces [F, 3] int64)`` in normalized coordinates with outward
    (counter-clockwise) faces; empty if the field never reaches ``level``.
    """
    if fof.dim() != 3:
        raise ValueError(f"fof must be [C, H, W], got {tuple(fof.shape)}")
    if solver not in ("pcg", "direct"):
        raise ValueError(f"solver must be 'pcg' or 'direct', got {solver!r}")
    device = fof.device
    with torch.autocast(device_type=device.type, enabled=False):
        fof_hw = _resize(fof[None].float(), resolution)[0]
    _, H, W = fof_hw.shape
    D = int(depth_res)
    occ = fof_to_occupancy(fof_hw, D)

    if _use_gpu_mc(backend, device):
        mc = _mc_gpu(occ, level, close, "auto" if backend == "auto" else backend)
        del occ
        if mc is None:
            return _empty_mesh(return_tensors, device)
        from .mc import vertex_axis

        ids, verts, faces, shape = mc
        fixed = vertex_axis(ids, shape) == 2
        if refine_roots:
            verts = _refine_roots(verts, fixed, fof_hw, level, D)
        if lossless:
            if solver == "pcg":
                verts = _ls_matrix_free(verts, faces, fixed, iterations, check_anchors=not close)
            else:
                verts = torch.from_numpy(_anchored_smooth(verts.cpu().numpy(), faces.cpu().numpy(),
                                                          fixed.cpu().numpy(), solver="direct")).to(device)
        v, f = _to_world(verts, D, H, W, z_range), _outward(faces)
        return (v, f) if return_tensors else (v.cpu().numpy(), f.cpu().numpy())

    mc = _mc_skimage(occ, level, close, allow_degenerate)
    del occ
    if mc is None:
        return _empty_mesh(return_tensors, device)
    verts, faces = mc
    if lossless or refine_roots:
        fixed = _depth_edge_vertices(verts)
        if refine_roots:
            verts = _refine_roots(verts, fixed, fof_hw, level, D)
        if lossless:
            verts = _anchored_smooth(verts, faces, fixed, solver=solver, tol=None, max_iter=iterations, device=device)
    v, f = _to_world(verts, D, H, W, z_range), _outward(faces)
    return (torch.from_numpy(v).to(device), torch.from_numpy(f).to(device)) if return_tensors else (v, f)
