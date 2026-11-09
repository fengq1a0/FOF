"""Tests for the crack-free marching cubes (lib/fof/mc.py, lib/fof/mc_tables.py)."""

import os
import sys
from collections import Counter

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.fof import _ext, fof_to_mesh, marching_cubes, mesh_to_fof, volume_to_mesh  # noqa: E402
from lib.fof import mc_tables as T  # noqa: E402
from lib.fof.mc import vertex_axis  # noqa: E402

HAS_CUDA = torch.cuda.is_available() and _ext.available()
BACKENDS = ["torch"] + (["cuda"] if HAS_CUDA else [])
needs_cuda = pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA and the compiled extension")


def dev(backend):
    return torch.device("cuda" if backend == "cuda" else "cpu")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _all_variants():
    for cfg in range(256):
        faces = [f for f in range(6) if T.AMB[cfg] >> f & 1]
        for bits in range(1 << len(faces)):
            yield cfg, {f: bool(bits >> j & 1) for j, f in enumerate(faces)}, T.BASE[cfg] + bits


def test_tables_are_valid_cell_triangulations():
    """Per variant: the boundary of the triangulation equals the face contour segments (each
    used once, consistently directed), interior edges are used twice in opposite directions,
    and no triangle or diagonal lies on a cube face."""
    assert T.BASE[255] + 1 == len(T.NTRI)
    for cfg, dec, var in _all_variants():
        codes = T.TRIS[var][: 3 * T.NTRI[var]].reshape(-1, 3).tolist()
        assert codes == [list(t) for t in T.triangulate(cfg, dec)[0]]
        directed = Counter()
        for a, b, c in codes:
            directed.update([(a, b), (b, c), (c, a)])
            on_face = [T._EDGE_FACES[x] if x < 12 else 0 for x in (a, b, c)]
            assert not (on_face[0] & on_face[1] & on_face[2]), (cfg, dec, "triangle on a face")
        segments = set()
        for f in range(6):
            segments |= set(T._face_segments(cfg, f, dec.get(f, False)))
        boundary = {e for e, n in directed.items() if (e[1], e[0]) not in directed}
        assert boundary == segments, (cfg, dec)
        assert all(n == 1 for n in directed.values()), (cfg, dec)
        for (a, b), n in directed.items():
            if (a, b) not in segments:  # interior edge: both directions, never on a common face
                assert (b, a) in directed
                if a < 12 and b < 12:
                    assert not (T._EDGE_FACES[a] & T._EDGE_FACES[b]), (cfg, dec, "diagonal on a face")
    # complementary configurations with the complementary decisions describe the same surface
    assert T.NTRI[T.BASE[0]] == 0 and T.NTRI[T.BASE[255]] == 0


# ---------------------------------------------------------------------------
# Meshes
# ---------------------------------------------------------------------------


def topology(verts, faces):
    """(edge-manifold & consistently oriented, every vertex link is one cycle, Euler char.)"""
    V = len(verts)
    f = faces.astype(np.int64)
    de = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    key = de[:, 0] * V + de[:, 1]
    _, cnt = np.unique(key, return_counts=True)
    edges_ok = bool((cnt == 1).all() and np.isin(de[:, 1] * V + de[:, 0], key).all())
    # vertex links: nodes (centre, other), one link edge per incident triangle
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    centre = f.reshape(-1)
    a = np.stack([f[:, 1], f[:, 2], f[:, 0]], 1).reshape(-1)
    b = np.stack([f[:, 2], f[:, 0], f[:, 1]], 1).reshape(-1)
    nodes, inv = np.unique(np.concatenate([centre * V + a, centre * V + b]), return_inverse=True)
    n = len(nodes)
    g = sp.coo_matrix((np.ones(len(a)), (inv[: len(a)], inv[len(a):])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    comps_per_vertex = np.bincount(np.unique(np.stack([nodes // V, lab], 1), axis=0)[:, 0], minlength=V)
    used = np.bincount(centre, minlength=V) > 0
    vertices_ok = bool((comps_per_vertex[used] == 1).all())
    n_edges = len(np.unique(np.sort(de, 1), axis=0))
    return edges_ok, vertices_ok, V - n_edges + len(f)


def signed_volume(verts, faces):
    v = verts.astype(np.float64)[:, [2, 1, 0]]  # (w, h, d): triangles are ccw around outward normals
    return np.einsum("ij,ij->i", v[faces[:, 0]], np.cross(v[faces[:, 1]], v[faces[:, 2]])).sum() / 6


def volumes():
    rng = np.random.default_rng(0)
    yield "random", np.pad(rng.random((10, 12, 9)).astype(np.float32), 1), 0.5
    yield "binary", np.pad((rng.random((14, 11, 13)) > 0.5).astype(np.float32), 1), 0.5
    yield "checker", np.pad(np.indices((8, 8, 8)).sum(0) % 2, 1).astype(np.float32), 0.5
    yield "ties", np.pad(rng.integers(0, 3, (9, 9, 9)).astype(np.float32) * 0.25, 1), 0.25  # values equal to the level
    x = np.linspace(-1, 1, 24)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    yield "ellipsoid", np.pad(0.7 - np.sqrt(X ** 2 + 1.5 * Y ** 2 + Z ** 2), 1, constant_values=-1).astype(np.float32), 0.0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name,vol,level", list(volumes()))
def test_closed_oriented_manifold(backend, name, vol, level):
    ids, v, f = marching_cubes(torch.from_numpy(vol).to(dev(backend)), level, backend=backend)
    v, f = v.cpu().numpy(), f.cpu().numpy()
    edges_ok, vertices_ok, chi = topology(v, f)
    assert edges_ok, name
    if name != "ties":  # several edges can interpolate onto the same grid corner at exact ties
        assert vertices_ok, name
    inside = float((vol > level).sum())
    if name == "checker":  # decider ties separate the inside corners: one octahedron (1/6) per voxel
        assert signed_volume(v, f) == pytest.approx(inside / 6)
    else:
        assert 0.5 * inside < signed_volume(v, f) < 1.2 * inside + 1  # outward, plausible volume
    if name == "ellipsoid":
        assert chi == 2
    # every vertex lies on its edge (or inside its cell) and on the iso-level of the edge
    axis = vertex_axis(ids, vol.shape).cpu().numpy()
    on_edge = axis >= 0
    frac = v[on_edge] - np.floor(v[on_edge])
    other = np.ones_like(frac, dtype=bool)
    other[np.arange(len(frac)), 2 - axis[on_edge]] = False
    assert (frac[other] == 0).all()


@pytest.mark.parametrize("backend", BACKENDS)
def test_degenerate_volumes(backend):
    d = dev(backend)
    for vol in (torch.zeros(5, 6, 7), torch.ones(4, 4, 4), torch.zeros(1, 5, 5), torch.zeros(2, 2, 2)):
        ids, v, f = marching_cubes(vol.to(d), 0.5, backend=backend)
        assert len(ids) == len(v) == len(f) == 0
    vol = torch.zeros(2, 2, 2, device=d)
    vol[0, 0, 0] = 1.0
    ids, v, f = marching_cubes(vol, 0.5, backend=backend)
    assert len(f) == 1 and sorted(map(tuple, v.cpu().tolist())) == [(0, 0, 0.5), (0, 0.5, 0), (0.5, 0, 0)]


@needs_cuda
def test_cuda_matches_torch():
    rng = np.random.default_rng(1)
    for _ in range(3):
        vol = torch.from_numpy(np.pad(rng.random((20, 17, 23)).astype(np.float32), 1))
        a = marching_cubes(vol.cuda(), 0.5, backend="cuda")
        b = marching_cubes(vol, 0.5, backend="torch")
        assert all(torch.equal(x.cpu(), y) for x, y in zip(a, b))


def test_matches_skimage_on_smooth_fields():
    from skimage.measure import marching_cubes as sk_mc

    x = np.linspace(-1, 1, 40)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    vol = (0.6 - np.sqrt(X ** 2 + 1.3 * Y ** 2 + (Z - 0.1) ** 2)).astype(np.float32)
    _, v, f = marching_cubes(torch.from_numpy(vol), 0.0, backend="torch")
    sv, sf, _, _ = sk_mc(vol, 0.0)
    ours = {tuple(p) for p in np.round(v.numpy().astype(np.float64), 4)}
    theirs = {tuple(p) for p in np.round(sv.astype(np.float64), 4)}
    assert ours == theirs  # same edge vertices (no ambiguous cells here)

    def area(V, F):
        return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum()

    assert area(v.numpy().astype(np.float64), f.numpy()) == pytest.approx(area(sv.astype(np.float64), sf), rel=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fof_to_mesh_gpu_path(backend):
    trimesh = pytest.importorskip("trimesh")
    m = trimesh.creation.icosphere(4, radius=0.6)
    m.apply_scale([1.0, 0.8, 0.7])
    d = dev(backend)
    fof = mesh_to_fof(torch.tensor(m.vertices, dtype=torch.float32, device=d), torch.tensor(m.faces, device=d), 32, 64)
    ref_v, ref_f = fof_to_mesh(fof.cpu(), depth_res=64, backend="skimage")
    for lossless in (False, True):
        v, f = fof_to_mesh(fof, depth_res=64, backend=backend, lossless=lossless, return_tensors=True)
        assert v.device == fof.device and f.device == fof.device
        out = trimesh.Trimesh(v.cpu().numpy(), f.cpu().numpy(), process=False)
        assert out.is_watertight and out.volume > 0
        assert out.volume == pytest.approx(trimesh.Trimesh(ref_v, ref_f, process=False).volume, rel=2e-3)
        if lossless:
            raw = fof_to_mesh(fof, depth_res=64, backend=backend, return_tensors=True)[0]
            assert (v - raw).abs().max() > 1e-4  # the smoothing did something


def test_matrix_free_solver_matches_scipy():
    from lib.fof.decode import _anchored_smooth, _ls_matrix_free

    x = np.linspace(-1, 1, 30)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    vol = np.pad(0.6 - np.abs(X) - 0.5 * np.abs(Y) - 0.3 * Z ** 2, 1, constant_values=-1).astype(np.float32)
    ids, v, f = marching_cubes(torch.from_numpy(vol), 0.0, backend="torch")
    fixed = vertex_axis(ids, vol.shape) == 2
    a = _ls_matrix_free(v.double(), f, fixed, 25).numpy()
    b = _anchored_smooth(v.double().numpy(), f.numpy(), fixed.numpy(), solver="pcg", tol=None, max_iter=25)
    assert np.abs(a - b).max() < 1e-8
    assert np.array_equal(a[fixed.numpy()], v.double().numpy()[fixed.numpy()])


@pytest.mark.parametrize("backend", BACKENDS)
def test_extreme_values_and_nan(backend):
    d = dev(backend)
    vol = torch.full((2, 2, 2), -1e20, device=d)
    vol[0, 0, 0] = vol[0, 1, 1] = 2e20  # ambiguous face whose products overflow float32
    a = marching_cubes(vol, 0.0, backend=backend)
    b = marching_cubes(vol / 1e20, 0.0, backend=backend)
    assert torch.equal(a[2], b[2]) and torch.allclose(a[1], b[1])  # scale invariant
    vol = torch.tensor([3e38] + [-3e38] * 7, device=d).view(2, 2, 2)
    _, v, _ = marching_cubes(vol, 0.0, backend=backend)
    assert torch.allclose(v.sort(1).values.cpu(), torch.tensor([[0.0, 0.0, 0.5]] * 3))  # midpoints, no overflow
    with pytest.raises(ValueError):
        marching_cubes(torch.tensor([1.0, float("nan"), 0, 0, 0, 0, 0, 0], device=d).view(2, 2, 2), 0.5, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_lossless_keeps_unanchored_components(backend):
    """An open wall through the whole depth range (close=False) has no depth-edge vertex."""
    d = dev(backend)
    fof = torch.tensor([[[0.0, 0.0], [2.0, 2.0]]], device=d)

    def area(v, f):
        return 0.5 * torch.linalg.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]).norm(dim=1).sum()

    raw = fof_to_mesh(fof, depth_res=4, close=False, backend=backend, return_tensors=True)
    for solver in ("pcg", "direct"):
        v, f = fof_to_mesh(fof, depth_res=4, close=False, backend=backend, lossless=True, solver=solver,
                           return_tensors=True)
        assert torch.equal(f, raw[1]) and torch.allclose(v, raw[0]) and float(area(v, f)) > 1.0


def test_level_precision_and_overflow():
    # level not representable in float32: classification and interpolation must agree
    vol = torch.tensor([1.0, 2.0]).expand(2, 2, 2).contiguous()
    assert len(marching_cubes(vol, 1 - 2 ** -25, backend="torch")[2]) == 0
    # overflow in (value - level) before widening
    vol = torch.tensor([3, -3, -3, -0.5, -3, -3, -3, -3]).reshape(2, 2, 2) * 1e38
    a = marching_cubes(vol, -1e38, backend="torch")
    b = marching_cubes(vol / 1e38, -1.0, backend="torch")
    assert torch.equal(a[2], b[2])


def test_torch_backend_slabs_match():
    from lib.fof.mc import _mc_torch

    rng = np.random.default_rng(5)
    vol = torch.from_numpy(np.pad(rng.random((11, 9, 10)).astype(np.float32), 1))
    one = _mc_torch(vol, 0.5)
    sliced = _mc_torch(vol, 0.5, max_cells=37)  # forces many slabs
    assert torch.equal(one[0], sliced[0]) and torch.equal(one[1], sliced[1])
    inner = vol[1:-1, 1:-1, 1:-1].contiguous()
    for max_cells in (1 << 24, 37):  # virtual border, built slab by slab
        border = _mc_torch(inner, 0.5, max_cells=max_cells, pad=0.0)
        assert torch.equal(one[0], border[0]) and torch.equal(one[1], border[1])


@pytest.mark.parametrize("backend", BACKENDS)
def test_virtual_border_matches_explicit_padding(backend):
    d = dev(backend)
    rng = np.random.default_rng(7)
    vols = [rng.random((9, 12, 10)), (rng.random((6, 5, 7)) > 0.3) * 1.0, np.ones((3, 1, 4)), np.ones((1, 1, 1))]
    for vol in vols:  # surfaces touching the volume boundary everywhere
        t = torch.tensor(vol, dtype=torch.float32, device=d)
        for level, pad in ((0.5, 0.0), (0.5, -0.5), (0.25, 0.25), (0.5, 0.7)):
            ids, pos, faces = marching_cubes(t, level, pad=pad, backend=backend)
            ids_e, pos_e, faces_e = marching_cubes(F.pad(t, (1,) * 6, value=pad), level, backend=backend)
            assert (len(faces) > 0 or pad > level) and torch.equal(ids, ids_e) and torch.equal(faces, faces_e)
            assert torch.allclose(pos, pos_e - 1, rtol=0, atol=1e-5)  # positions in vol's index frame
            assert torch.equal(vertex_axis(ids, t.shape, padded=True), vertex_axis(ids_e, [n + 2 for n in t.shape]))
            if pad < level:
                edges_ok, vertices_ok, _ = topology(pos.cpu().numpy(), faces.cpu().numpy())
                assert edges_ok and vertices_ok  # closed by the border
    with pytest.raises(ValueError):
        marching_cubes(torch.ones(2, 2, 2, device=d), 0.5, pad=float("inf"), backend=backend)
    assert len(marching_cubes(torch.ones(0, 3, 3, device=d), 0.5, pad=0.0, backend=backend)[2]) == 0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("level", [-33554432.5, -2.0 ** 24, -0.5, 0.0])
def test_closing_border_stays_below_extreme_levels(backend, level):
    # level - 1 rounds back up to the level in float32 when |level| >= 2^24
    d = "cuda" if backend == "cuda" else "cpu"
    v, f = volume_to_mesh(torch.ones(2, 2, 2, device=d), level=level, close=True, backend=backend)
    assert len(f) > 0


@needs_cuda
def test_virtual_border_does_not_copy_the_volume():
    vol = torch.zeros(160, 160, 160, device="cuda")  # 16 MiB
    vol[70:90, 70:90, 70:90] = 1.0
    marching_cubes(vol, 0.5, pad=0.0)  # warm-up (extension, tables)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    _, _, f = marching_cubes(vol, 0.5, pad=0.0)
    torch.cuda.synchronize()
    assert len(f) > 0
    assert torch.cuda.max_memory_allocated() - base < vol.numel() * vol.element_size() // 4
