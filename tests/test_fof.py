"""Tests for lib.fof (mesh -> FOF -> mesh).

    pytest tests/                  # torch backend on CPU, plus CUDA if available
"""

import itertools
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.fof import (  # noqa: E402
    FOF,
    FOF_Normal,
    fof_basis,
    fof_to_mesh,
    fof_to_occupancy,
    mesh_to_fof,
    rasterize,
    render,
    render_fof,
)
from lib.fof import _ext  # noqa: E402

trimesh = pytest.importorskip("trimesh")

HAS_CUDA = torch.cuda.is_available() and _ext.available()
BACKENDS = ["torch"] + (["cuda"] if HAS_CUDA else [])
needs_cuda = pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA and the compiled extension")


def dev(backend):
    return torch.device("cuda" if backend == "cuda" else "cpu")


def tensors(mesh, backend):
    v = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=dev(backend))
    f = torch.tensor(np.asarray(mesh.faces), dtype=torch.long, device=dev(backend))
    return v, f


def pixel_centres(n):
    return (torch.arange(n, dtype=torch.float64) + 0.5) * 2 / n - 1


def analytic_box_fof(bounds, H, W, C):
    (x0, y0, z0), (x1, y1, z1) = bounds
    xc = pixel_centres(W)
    yc = -pixel_centres(H)
    inside = (yc[:, None] > y0) & (yc[:, None] < y1) & (xc[None] > x0) & (xc[None] < x1)
    t0, t1 = (max(z0, -1) + 1) / 2, (min(z1, 1) + 1) / 2
    c = torch.arange(C, dtype=torch.float64)
    a = 2 / (c.clamp(min=1) * math.pi) * (torch.sin(c * math.pi * t1) - torch.sin(c * math.pi * t0))
    a[0] = 2 * (t1 - t0)
    return inside[None].double() * a[:, None, None]


def slab(x0, x1, y0, y1, z0, z1):
    return trimesh.creation.box(bounds=[[x0, y0, z0], [x1, y1, z1]])


def stack(*meshes):
    return trimesh.util.concatenate(meshes)


# ---------------------------------------------------------------------------
# Encoder: analytic cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_box_matches_closed_form(backend):
    bounds = [[-0.53, -0.37, -0.45], [0.41, 0.62, 0.3]]
    v, f = tensors(trimesh.creation.box(bounds=bounds), backend)
    H, W, C = 48, 64, 24
    fof = mesh_to_fof(v, f, C, (H, W), backend=backend)
    assert fof.shape == (C, H, W) and fof.dtype == torch.float32 and fof.is_contiguous()
    ref = analytic_box_fof(bounds, H, W, C)
    assert (fof.cpu().double() - ref).abs().max() < 1e-6
    counts = rasterize(v, f, (H, W), backend=backend).counts()
    assert set(counts.unique().tolist()) <= {0, 2}


@pytest.mark.parametrize("backend", BACKENDS)
def test_depth_is_clipped_to_range(backend):
    # occupancy outside z in [-1, 1] must not leak into the coefficients
    bounds = [[-0.5, -0.5, -2.0], [0.5, 0.5, 2.0]]
    v, f = tensors(trimesh.creation.box(bounds=bounds), backend)
    fof = mesh_to_fof(v, f, 8, 32, backend=backend).cpu()
    ref = analytic_box_fof(bounds, 32, 32, 8)
    assert (fof.double() - ref).abs().max() < 1e-6
    assert fof[0].max() <= 2.0 + 1e-6


@pytest.mark.parametrize("backend", BACKENDS)
def test_marching_cubes_mesh_is_encoded_exactly(backend):
    """Re-encode a marching-cubes mesh of a random binary volume at the same
    resolution: every pixel centre sits on mesh vertices/edges, the hardest case
    for coverage ties.  The occupied length of each column is known exactly."""
    from skimage.measure import marching_cubes

    rng = np.random.default_rng(0)
    D = H = W = 32
    vol = np.zeros((D, H, W), np.float32)
    for _ in range(12):
        lo = rng.integers(2, 24, 3)
        hi = lo + rng.integers(2, 8, 3)
        vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 1
    verts, faces, _, _ = marching_cubes(np.pad(vol, 1), 0.5)
    verts -= 1
    d, h, w = verts.T
    xyz = np.stack([(w + 0.5) * 2 / W - 1, 1 - (h + 0.5) * 2 / H, (d + 0.5) * 2 / D - 1], 1)
    mesh = trimesh.Trimesh(xyz, faces[:, [0, 2, 1]], process=False)
    assert mesh.is_watertight and mesh.volume > 0
    v, f = tensors(mesh, backend)
    frags = rasterize(v, f, H, backend=backend)
    counts = frags.counts()
    assert int((counts % 2).sum()) == 0
    depth, is_exit, _ = frags.decode()
    ptr = frags.pix_ptr.long().cpu()
    ex = is_exit.cpu()
    for p in torch.nonzero(ptr[1:] > ptr[:-1]).squeeze(1)[::7].tolist():  # alternating entries / exits
        seq = ex[ptr[p]:ptr[p + 1]]
        assert torch.equal(seq, torch.arange(len(seq)) % 2 == 1)
    expected = torch.from_numpy(vol.sum(0) * 2.0 / D)
    for fill in ["robust", "parity", "winding"]:
        fof = mesh_to_fof(v, f, 4, H, fill=fill, backend=backend).cpu()
        assert (fof[0] - expected).abs().max() < 1e-5, fill


@pytest.mark.parametrize("backend", BACKENDS)
def test_shared_edge_through_pixel_centre(backend):
    """Float32 counter-example of the original kernel: pixel (256, 256) lies on the
    shared diagonal and was rejected by both triangles.  Exactly one must claim it."""
    H = 512
    A, B = (124.4248047, 44.2951164), (339.0508118, 389.6289978)
    Cr, Dl = (420.0, 80.0), (40.0, 420.0)
    pix = np.array([A, Cr, B, Dl], np.float64)
    x = (pix[:, 0] + 0.5) * 2 / H - 1
    y = 1 - (pix[:, 1] + 0.5) * 2 / H
    front = np.stack([x, y, np.full(4, 0.5)], 1)
    back = np.stack([x, y, np.full(4, -0.5)], 1)
    verts = np.concatenate([front, back])
    # front quad split along A-B (camera facing), back quad (facing away), sides
    faces = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7]]
    for i in range(4):
        j = (i + 1) % 4
        faces += [[i, j, j + 4], [i, j + 4, i + 4]]
    mesh = trimesh.Trimesh(verts, faces, process=False)
    if mesh.volume < 0:
        mesh.invert()
    assert mesh.is_watertight
    v, f = tensors(mesh, backend)
    counts = rasterize(v, f, H, backend=backend).counts()
    assert counts[255:258, 255:258].eq(2).all()
    fof = mesh_to_fof(v, f, 2, H, backend=backend)
    assert torch.allclose(fof[0, 255:258, 255:258].cpu(), torch.ones(3, 3), atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_closed_meshes_have_even_crossings(backend):
    rng = np.random.default_rng(1)
    for mesh in [trimesh.creation.icosphere(4, radius=0.7), trimesh.creation.torus(0.55, 0.2)]:
        R = trimesh.transformations.random_rotation_matrix(rng.random(3))
        mesh = mesh.copy()
        mesh.apply_transform(R)
        v, f = tensors(mesh, backend)
        counts = rasterize(v, f, 96, backend=backend).counts()
        assert int((counts % 2).sum()) == 0
        fofs = [mesh_to_fof(v, f, 16, 96, fill=r, backend=backend) for r in ["robust", "parity", "winding"]]
        assert torch.allclose(fofs[0], fofs[1], atol=1e-6) and torch.allclose(fofs[0], fofs[2], atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fill_rules_on_overlapping_solids(backend):
    # Along z: E(-.9) E(-.6) X(-.2) E(0) X(.4) X(.8)
    meshes = stack(slab(-.5, .5, -.5, .5, -.9, .4), slab(-.5, .5, -.5, .5, -.6, -.2), slab(-.5, .5, -.5, .5, 0, .8))
    v, f = tensors(meshes, backend)
    a0 = {r: mesh_to_fof(v, f, 1, 8, fill=r, backend=backend)[0, 4, 4].item() for r in ["robust", "winding", "parity"]}
    assert a0["winding"] == pytest.approx(1.7, abs=1e-6)  # union [-.9, .8]
    assert a0["robust"] == pytest.approx(0.7 + 0.8, abs=1e-6)  # legacy: [-.9, -.2] U [0, .8]
    assert a0["parity"] == pytest.approx(0.3 + 0.2 + 0.4, abs=1e-6)  # [-.9,-.6] U [-.2,0] U [.4,.8]


@pytest.mark.parametrize("backend", BACKENDS)
def test_inverted_and_open_meshes(backend):
    box = slab(-.5, .5, -.5, .5, -.5, .5)
    inv = box.copy()
    inv.invert()
    v, f = tensors(inv, backend)
    assert mesh_to_fof(v, f, 1, 8, fill="robust", backend=backend).abs().max() == 0  # no entries
    assert mesh_to_fof(v, f, 1, 8, fill="winding", backend=backend)[0, 4, 4].item() == pytest.approx(1.0, abs=1e-6)
    assert mesh_to_fof(v, f, 1, 8, fill="parity", backend=backend)[0, 4, 4].item() == pytest.approx(1.0, abs=1e-6)
    # a lone back-facing triangle opens an interval that never closes -> dropped
    tri = torch.tensor([[[-.9, -.9, 0], [.9, -.9, 0], [0, .9, 0]]], device=dev(backend))
    tri = tri[:, [0, 2, 1]]  # make it face away from the camera
    for r in ["robust", "parity", "winding"]:
        assert mesh_to_fof(tri, None, 2, 16, fill=r, backend=backend).abs().max() == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_many_layers(backend):
    # 12 full-frame slabs = 24 crossings per pixel (the old static buffer held 4)
    slabs = [slab(-1.5, 1.5, -1.5, 1.5, -0.95 + 0.16 * k, -0.95 + 0.16 * k + 0.08) for k in range(12)]
    v, f = tensors(stack(*slabs), backend)
    counts = rasterize(v, f, 32, backend=backend).counts()
    assert counts.min() == 24 and counts.max() == 24
    fof = mesh_to_fof(v, f, 1, 32, backend=backend)
    assert torch.allclose(fof, torch.full_like(fof, 12 * 0.08), atol=1e-5)


# ---------------------------------------------------------------------------
# Encoder: API
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_input_layouts_agree(backend):
    rng = np.random.default_rng(2)
    meshes = []
    for k in range(3):
        m = trimesh.creation.icosphere(3 - (k == 2), radius=0.4 + 0.1 * k)
        m.apply_translation(rng.uniform(-0.2, 0.2, 3))
        meshes.append(m)
    d = dev(backend)
    ref = torch.stack([mesh_to_fof(*tensors(m, backend), 8, 40, backend=backend) for m in meshes])
    # list of meshes (different sizes, padded internally)
    lst = mesh_to_fof([tensors(m, backend)[0] for m in meshes], [tensors(m, backend)[1] for m in meshes], 8, 40,
                      backend=backend)
    assert torch.equal(lst, ref)
    # padded batch with -1 faces
    V = max(len(m.vertices) for m in meshes)
    F = max(len(m.faces) for m in meshes)
    vb = torch.zeros(3, V, 3, device=d)
    fb = torch.full((3, F, 3), -1, dtype=torch.long, device=d)
    for b, m in enumerate(meshes):
        vb[b, :len(m.vertices)], fb[b, :len(m.faces)] = tensors(m, backend)
    assert torch.equal(mesh_to_fof(vb, fb, 8, 40, backend=backend), ref)
    # triangle soup, zero-padded like lib/dataset_mesh_only.py
    soup = torch.zeros(3, F, 3, 3, device=d)
    for b, m in enumerate(meshes):
        v, f = tensors(m, backend)
        soup[b, :len(m.faces)] = v[f]
    assert torch.equal(mesh_to_fof(soup, None, 8, 40, backend=backend), ref)
    # shared faces with a batch of vertex sets
    m = meshes[0]
    v, f = tensors(m, backend)
    vb2 = torch.stack([v, v * 0.5])
    out = mesh_to_fof(vb2, f, 8, 40, backend=backend)
    assert torch.equal(out[0], ref[0])


@pytest.mark.parametrize("backend", BACKENDS)
def test_render_depth_and_normals(backend):
    bounds = [[-0.5, -0.4, -0.3], [0.45, 0.5, 0.2]]
    box = trimesh.creation.box(bounds=bounds)
    v, f = tensors(box, backend)
    # per-vertex normals pointing straight out of the front/back faces would be
    # ambiguous at box corners, so use a triangle soup with face normals per corner
    soup = v[f]
    fn = torch.tensor(np.asarray(box.face_normals), dtype=torch.float32, device=v.device)
    nsoup = fn[:, None, :].expand(-1, 3, -1)
    r = render_fof(soup, None, nsoup, 4, 32, backend=backend)
    inside = analytic_box_fof(bounds, 32, 32, 1)[0] > 0
    m = r.mask.cpu()
    assert torch.equal(m, inside)
    assert torch.allclose(r.depth_front.cpu()[m], torch.full((int(m.sum()),), 0.2))
    assert torch.allclose(r.depth_back.cpu()[m], torch.full((int(m.sum()),), -0.3))
    assert (r.depth_front.cpu()[~m] == -1).all() and (r.depth_back.cpu()[~m] == 1).all()
    nf, nb = r.normal_front.cpu(), r.normal_back.cpu()
    assert torch.allclose(nf[:, m], torch.tensor([0.0, 0, 1])[:, None].expand(3, int(m.sum())), atol=1e-6)
    assert torch.allclose(nb[:, m], torch.tensor([0.0, 0, -1])[:, None].expand(3, int(m.sum())), atol=1e-6)
    assert (nf[:, ~m] == 0).all()
    assert torch.allclose(r.bary_front.sum(-1).cpu()[m], torch.ones(int(m.sum())), atol=1e-6)
    assert r.fof.shape == (4, 32, 32)


@pytest.mark.parametrize("backend", BACKENDS)
def test_legacy_wrappers(backend):
    m = trimesh.creation.icosphere(3, radius=0.6)
    v, f = tensors(m, backend)
    soup = v[f][None]
    nsoup = torch.tensor(np.asarray(m.vertex_normals), dtype=torch.float32, device=v.device)[f][None]
    a = FOF(1, 32).to(v.device)(soup, 8)
    b = render(soup, 8, 32)
    fof, dF, dB, nF, nB = FOF_Normal(1, 32).to(v.device)(soup, nsoup, 8)
    ref = mesh_to_fof(v, f, 8, 32)[None]
    for x in (a, b, fof):
        assert torch.allclose(x, ref, atol=1e-6)
    assert dF.shape == (1, 32, 32) and nF.shape == (1, 3, 32, 32)
    assert (dF >= dB)[nF.norm(dim=1) > 0].all()


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_and_offscreen(backend):
    d = dev(backend)
    assert mesh_to_fof(torch.zeros(2, 0, 3, 3, device=d), None, 4, 8, backend=backend).abs().sum() == 0
    box = slab(3, 4, 3, 4, -.5, .5)  # outside the image
    v, f = tensors(box, backend)
    assert mesh_to_fof(v, f, 4, 8, backend=backend).abs().sum() == 0
    r = render_fof(v, f, None, None, 8, backend=backend)
    assert r.fof is None and not r.mask.any() and r.normal_front is None


def test_invalid_inputs():
    v = torch.rand(10, 3)
    with pytest.raises(ValueError):
        mesh_to_fof(v, None, 4, 8)
    with pytest.raises(TypeError):
        mesh_to_fof(v, torch.zeros(2, 3), 4, 8)
    with pytest.raises(ValueError):
        mesh_to_fof(v, torch.zeros(2, 3, dtype=torch.long), 4, 8, fill="evenodd")
    with pytest.raises(ValueError):
        mesh_to_fof(v, torch.zeros(2, 3, dtype=torch.long), 0, 8)
    with pytest.raises(ValueError):
        mesh_to_fof(v, torch.zeros(2, 3, dtype=torch.long), 4, 8, backend="cuda")
    box = slab(-.5, .5, -.5, .5, -.5, .5)
    vb = torch.tensor(box.vertices, dtype=torch.float32)
    vb[0, 0] = float("nan")
    with pytest.warns(RuntimeWarning):
        out = mesh_to_fof(vb, torch.tensor(box.faces), 2, 8)
    assert torch.isfinite(out).all()


@needs_cuda
def test_cuda_matches_torch_bitwise():
    rng = np.random.default_rng(3)
    m = trimesh.creation.icosphere(5, radius=0.75)
    m.vertices += rng.normal(scale=0.01, size=m.vertices.shape)
    for H in (64, 257):
        v, f = tensors(m, "cuda")
        a = rasterize(v, f, H, backend="cuda")
        b = rasterize(v, f, H, backend="torch")
        assert torch.equal(a.pix_ptr, b.pix_ptr) and torch.equal(a.keys, b.keys)
        ra = render_fof(v, f, None, 32, H, backend="cuda")
        rb = render_fof(v, f, None, 32, H, backend="torch")
        assert (ra.fof - rb.fof).abs().max() < 1e-5
        assert torch.equal(ra.face_front, rb.face_front) and torch.equal(ra.bary_back, rb.bary_back)


@needs_cuda
def test_cuda_stream_and_device_guard():
    m = trimesh.creation.icosphere(3, radius=0.6)
    v, f = tensors(m, "cuda")
    ref = mesh_to_fof(v, f, 16, 64)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        out = mesh_to_fof(v * 1.0, f, 16, 64)
    s.synchronize()
    assert torch.equal(out, ref)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


def test_basis_reconstructs_interval():
    C, D = 128, 256
    t0, t1 = 0.3, 0.65
    c = torch.arange(C, dtype=torch.float64)
    a = 2 / (c.clamp(min=1) * math.pi) * (torch.sin(c * math.pi * t1) - torch.sin(c * math.pi * t0))
    a[0] = 2 * (t1 - t0)
    occ = fof_basis(C, D, dtype=torch.float64) @ a
    t = (torch.arange(D, dtype=torch.float64) + 0.5) / D
    far = ((t - t0).abs() > 0.05) & ((t - t1).abs() > 0.05)
    assert ((occ - ((t > t0) & (t < t1)).double()).abs()[far] < 0.05).all()
    crossings = torch.nonzero((occ[1:] - 0.5) * (occ[:-1] - 0.5) < 0).squeeze(1)
    assert len(crossings) == 2


def _sphere_fof(H=96, C=48, radius=0.6):
    m = trimesh.creation.icosphere(5, radius=radius)
    m.apply_scale([1.0, 0.8, 0.7])
    v, f = tensors(m, "torch")
    return m, mesh_to_fof(v, f, C, H, backend="torch")


def test_fof_to_mesh_roundtrip():
    m, fof = _sphere_fof()
    verts, faces = fof_to_mesh(fof, depth_res=96)
    out = trimesh.Trimesh(verts, faces, process=False)
    assert out.is_watertight and out.volume > 0  # outward faces
    assert out.volume == pytest.approx(m.volume, rel=0.02)
    assert np.abs(trimesh.proximity.signed_distance(m, verts[::5])).max() < 2.5 * 2 / 96
    occ = fof_to_occupancy(fof, 96)
    assert occ.shape == (96, 96, 96)


def test_lossless_decoding():
    m, fof = _sphere_fof()
    raw_v, raw_f = fof_to_mesh(fof, depth_res=96)
    err_raw = np.abs(trimesh.proximity.signed_distance(m, raw_v)).mean()
    for kw in [dict(lossless=True), dict(lossless=True, refine_roots=True)]:
        v, f = fof_to_mesh(fof, depth_res=96, **kw)
        assert np.array_equal(f, raw_f) and np.isfinite(v).all()
        out = trimesh.Trimesh(v, f, process=False)
        assert out.is_watertight and out.volume > 0
        assert np.abs(trimesh.proximity.signed_distance(m, v)).mean() < err_raw


def test_anchored_smooth_keeps_anchors():
    from skimage.measure import marching_cubes

    from lib.fof.decode import _anchored_smooth, _depth_edge_vertices

    _, fof = _sphere_fof()
    vol = fof_to_occupancy(fof, 96).numpy()
    verts, faces, _, _ = marching_cubes(np.pad(vol, 1), 0.5)
    verts -= 1
    fixed = _depth_edge_vertices(verts)
    assert 0.2 < fixed.mean() < 0.9
    out = _anchored_smooth(verts, faces, fixed)
    assert np.array_equal(out[fixed], verts[fixed])
    assert np.abs(out[~fixed] - verts[~fixed]).max() > 1e-3
    # an isolated free component (no anchors) must not move or collapse
    tet_v = np.array([[0.5, 0.5, 0.5], [3.5, 0.5, 0.5], [0.5, 3.5, 0.5], [0.5, 0.5, 3.5]], np.float32)
    tet_f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    assert np.array_equal(_anchored_smooth(tet_v, tet_f, np.zeros(4, bool)), tet_v)


def test_fof_to_mesh_nonsquare_and_border():
    bounds = [[-0.5, -0.4, -0.6], [1.5, 0.3, 0.2]]  # crosses the right image border
    v, f = tensors(trimesh.creation.box(bounds=bounds), "torch")
    fof = mesh_to_fof(v, f, 64, (40, 56), backend="torch")
    verts, faces = fof_to_mesh(fof, depth_res=72)
    out = trimesh.Trimesh(verts, faces, process=False)
    assert out.is_watertight and out.volume > 0
    lo, hi = out.bounds
    assert np.allclose(lo, [-0.5, -0.4, -0.6], atol=0.04)
    # the cap closing the cut at the border lies within half a pixel of x = 1
    assert np.allclose(hi[1:], [0.3, 0.2], atol=0.04) and abs(hi[0] - 1.0) < 1.0 / 56
    open_v, open_f = fof_to_mesh(fof, depth_res=72, close=False)
    assert not trimesh.Trimesh(open_v, open_f, process=False).is_watertight


def test_fof_to_mesh_empty():
    v, f = fof_to_mesh(torch.zeros(16, 8, 8))
    assert v.shape == (0, 3) and f.shape == (0, 3)


# ---------------------------------------------------------------------------
# Regressions found in review
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_batches(backend):
    d = dev(backend)
    assert mesh_to_fof(torch.zeros(0, 2, 3, 3, device=d), None, 4, 8, backend=backend).shape == (0, 4, 8, 8)
    assert mesh_to_fof([], [], 4, 8).shape == (0, 4, 8, 8)  # no tensors: no device to run on
    r = render_fof(torch.zeros(1, 0, 3, 3, device=d), None, torch.zeros(1, 0, 3, 3, device=d), 4, 8, backend=backend)
    assert not r.mask.any() and (r.normal_front == 0).all()


@pytest.mark.parametrize("backend", BACKENDS)
def test_out_of_range_faces_are_skipped(backend):
    d = dev(backend)
    v = torch.tensor([[-.5, -.5, 0], [.5, -.5, 0], [0, .5, 0]], device=d)
    # int64 indices that would wrap to valid int32 indices
    wrapped = torch.tensor([[2 ** 32, 2 ** 32 + 1, 2 ** 32 + 2]], device=d)
    assert rasterize(v, wrapped, 16, backend=backend).keys.numel() == 0
    # a face indexing past its own mesh must stay invalid when batched with a bigger mesh
    big = torch.cat([v, torch.zeros(2, 3, device=d)])
    lst = rasterize([v, big], [torch.tensor([[0, 1, 3]], device=d), torch.tensor([[0, 1, 2]], device=d)], 16,
                    backend=backend)
    assert lst.counts()[0].sum() == 0 and lst.counts()[1].sum() > 0
    # narrow integer dtypes must not wrap the bound (uint8 faces, 256 vertices)
    v256 = torch.cat([v, torch.zeros(253, 3, device=d)])
    assert rasterize(v256, torch.tensor([[0, 1, 2]], dtype=torch.uint8, device=d), 16, backend=backend).keys.numel() > 0
    # invalid faces next to valid ones do not break normal interpolation
    faces = torch.tensor([[99, 99, 99], [0, 1, 2]], device=d)
    r = render_fof(v, faces, torch.tensor([[0, 0, 1.0]] * 3, device=d), 2, 16, backend=backend)
    assert r.mask.any() and torch.allclose(r.normal_front[:, r.mask], torch.tensor([0, 0, 1.0], device=d)[:, None])


def _wall_fof():
    fof = torch.zeros(1, 16, 16)
    fof[:, 4:12, 4:12] = 2  # full-depth prism: long free side walls, few anchors
    return fof


def test_lossless_solver_converges():
    from skimage.measure import marching_cubes

    from lib.fof.decode import _anchored_smooth, _depth_edge_vertices

    vol = np.pad(fof_to_occupancy(_wall_fof(), 32).numpy(), 1)
    verts, faces, _, _ = marching_cubes(vol, 0.5)
    verts -= 1
    fixed = _depth_edge_vertices(verts)
    direct = _anchored_smooth(verts, faces, fixed, solver="direct")
    auto = _anchored_smooth(verts, faces, fixed)  # CPU: direct
    assert np.allclose(direct, auto)
    with pytest.warns(RuntimeWarning):
        _anchored_smooth(verts, faces, fixed, solver="pcg", max_iter=20, fidelity=0.0)
    pcg = _anchored_smooth(verts, faces, fixed, solver="pcg", max_iter=20000, fidelity=0.0)
    assert np.abs(pcg - direct).max() < 1e-3
    # the default fidelity term keeps long, weakly anchored walls from drifting
    reg = _anchored_smooth(verts, faces, fixed, solver="pcg", tol=None, max_iter=25)
    assert np.abs(reg - verts).max() < np.abs(direct - verts).max()
    # a reported convergence must hold for the true residual at the requested tolerance
    import scipy.sparse as sp

    from lib.fof.decode import _pcg

    rng = np.random.default_rng(0)
    Q = sp.random(300, 300, density=0.02, random_state=0)
    M = (Q.T @ Q + sp.identity(300) * 1e-3).tocsr()
    b = rng.normal(size=(300, 3))
    x, ok = _pcg(lambda v: M @ v, b, np.zeros_like(b), 1.0 / M.diagonal()[:, None], 1e-9, 10000)
    assert ok and (np.linalg.norm(b - M @ x, axis=0) <= 1e-9 * np.linalg.norm(b, axis=0)).all()
    v, f = fof_to_mesh(_wall_fof(), depth_res=32, lossless=True)
    assert np.isfinite(v).all() and trimesh.Trimesh(v, f, process=False).is_watertight


def test_refine_roots_keeps_exact_roots():
    fof = torch.zeros(2, 4, 4)
    fof[0], fof[1] = 1.0, 0.5  # o(t) = 0.5 + 0.5 cos(pi t): crossing exactly at z = 0
    for D in (4, 16):
        v, _ = fof_to_mesh(fof, depth_res=D, close=False)
        vr, _ = fof_to_mesh(fof, depth_res=D, close=False, refine_roots=True)
        assert np.abs(v[:, 2]).max() < 1e-6 and np.abs(vr[:, 2]).max() < 1e-6
    fof[0], fof[1] = 1.2, 0.3  # generic root: cos(pi t) = -1/3
    vr, _ = fof_to_mesh(fof, depth_res=8, close=False, refine_roots=True)
    t = (vr[:, 2].astype(np.float64) + 1) / 2
    assert np.abs(0.6 + 0.3 * np.cos(np.pi * t) - 0.5).max() < 1e-6


def test_decoding_ignores_autocast():
    _, fof = _sphere_fof(H=48, C=16)
    ref_v, ref_f = fof_to_mesh(fof, depth_res=48)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        v, f = fof_to_mesh(fof, depth_res=48)
        occ = fof_to_occupancy(fof, 48)
    assert occ.dtype == torch.float32
    assert np.array_equal(f, ref_f) and np.allclose(v, ref_v)


def test_recon_reproduces_legacy_decoder():
    import torch.nn.functional as F
    from skimage.measure import marching_cubes

    from lib.utils import Recon

    _, fof = _sphere_fof(H=64, C=32)
    for res in (64, 48):
        v, f = Recon(None, 32).decode(fof, res)
        # original lib/utils.py: 512 depth samples, trilinear resize, skimage MC, axis swap
        z = (torch.arange(512, dtype=torch.float32) + 0.5) / 512
        basis = torch.cos(torch.arange(32, dtype=torch.float32).view(1, 32) * z.view(512, 1) * np.pi)
        basis[:, 0] = 0.5
        vol = torch.einsum("dc,chw->dhw", basis, fof)
        vol = F.interpolate(vol[None, None], (res, res, res), mode="trilinear", align_corners=False)[0, 0]
        ov, of, _, _ = marching_cubes(vol.numpy(), level=0.5)
        ov = (ov + 0.5) / (res / 2) - 1
        ov[:, 1] *= -1
        assert np.array_equal(f, of[:, [0, 2, 1]])
        assert np.array_equal(v, ov[:, [2, 1, 0]])


# ---------------------------------------------------------------------------
# Fill rules: exhaustive short crossing sequences vs. scalar references
# ---------------------------------------------------------------------------


def _reference_intervals(seq, rule):
    """seq: is_exit flags of the crossings, ordered by depth; returns (open, close) positions."""
    n, out = len(seq), []
    if rule == "parity":
        return [(k, k + 1) for k in range(0, n - 1, 2)]
    if rule == "winding":
        w, t0 = 0, None
        for k, ex in enumerate(seq):
            w_new = w + (-1 if ex else 1)
            if w == 0:
                t0 = k
            elif w_new == 0:
                out.append((t0, k))
            w = w_new
        return out
    if rule == "robust":
        state, t0, t1 = 0, None, None
        for k, ex in enumerate(seq):
            if state == 0 and not ex:
                state, t0 = 1, k
            elif state == 1 and ex:
                state, t1 = 2, k
            elif state == 2:
                if ex:
                    t1 = k
                else:
                    out.append((t0, t1))
                    state, t0 = 1, k
        return out + ([(t0, t1)] if state == 2 else [])
    # hybrid
    w, start, last, has_exit, just_closed = 0, None, None, False, False
    for k, ex in enumerate(seq):
        if not ex:
            if w == 0:
                start, has_exit = k, False
            w += 1
            just_closed = False
        elif w > 0:
            w -= 1
            has_exit, last = True, k
            if w == 0:
                out.append([start, k])
                just_closed = True
        elif just_closed:
            out[-1][1] = k
    if w > 0 and has_exit:
        out.append([start, last])
    if not out:
        out = [(k, k + 1) for k in range(0, n - 1, 2)]
    return [tuple(x) for x in out]


def _sequence_batch():
    from lib.fof.encode import _INT64_MIN, _make_key_signed

    seqs = [list(s) for n in range(9) for s in itertools.product([False, True], repeat=n)]
    depth, ex, ptr = [], [], [0]
    for s in seqs:
        depth += [-0.9 + 1.8 * k / max(len(s), 1) for k in range(len(s))]
        ex += s
        ptr.append(len(depth))
    depth = torch.tensor(depth, dtype=torch.float32)
    keys = _make_key_signed(depth, torch.tensor(ex, dtype=torch.bool), torch.arange(len(depth))) ^ _INT64_MIN
    return seqs, depth, keys, torch.tensor(ptr, dtype=torch.int32)


@pytest.mark.parametrize("rule", ["robust", "parity", "winding", "hybrid"])
def test_fill_rules_exhaustive(rule):
    from lib.fof.encode import FILL_RULES, Fragments, _fof_torch

    seqs, depth, keys, ptr = _sequence_batch()
    expected = torch.tensor([sum(float(depth[ptr[i] + b] - depth[ptr[i] + a]) for a, b in _reference_intervals(s, rule))
                             for i, s in enumerate(seqs)])
    frags = Fragments(ptr, keys, (1, 1, len(seqs)))
    got = _fof_torch(frags, 1, FILL_RULES[rule], -1.0, 1.0)[0, 0, 0]
    assert torch.allclose(got, expected, atol=1e-5), rule
    if HAS_CUDA:
        got = _ext.load().fof_integrate(ptr.cuda(), keys.cuda(), 1, 1, len(seqs), 1, -1.0, 1.0, FILL_RULES[rule])
        assert torch.allclose(got[0, 0, 0].cpu(), expected, atol=1e-5), rule


@pytest.mark.parametrize("backend", BACKENDS)
def test_hybrid_rule_on_defects(backend):
    box = slab(-.5, .5, -.5, .5, -.5, .5)
    # flipped front face (two triangles): only parity and hybrid recover the solid
    f = np.asarray(box.faces).copy()
    front = np.nonzero(box.face_normals[:, 2] > 0.9)[0]
    f[front] = f[front][:, ::-1]
    flipped = trimesh.Trimesh(box.vertices, f, process=False)
    v, ff = tensors(flipped, backend)
    for rule, ok in (("hybrid", True), ("parity", True), ("robust", False), ("winding", False)):
        a0 = mesh_to_fof(v, ff, 1, 8, fill=rule, backend=backend)[0, 4, 4].item()
        assert (abs(a0 - 1.0) < 1e-6) == ok, rule
    # duplicated inner partial layer (an extra entry): robust and hybrid keep the full column
    inner = slab(-.4, .4, -.4, .4, -.45, .45)
    back = np.nonzero(inner.face_normals[:, 2] < -0.9)[0]
    layer = trimesh.Trimesh(inner.vertices, inner.faces[back], process=False)
    v, ff = tensors(stack(box, layer), backend)
    for rule, ok in (("hybrid", True), ("robust", True), ("winding", False), ("parity", False)):
        a0 = mesh_to_fof(v, ff, 1, 8, fill=rule, backend=backend)[0, 4, 4].item()
        assert (abs(a0 - 1.0) < 1e-6) == ok, rule
    # interpenetrating parts: union (hybrid = winding)
    meshes = stack(slab(-.5, .5, -.5, .5, -.9, .4), slab(-.5, .5, -.5, .5, -.6, -.2), slab(-.5, .5, -.5, .5, 0, .8))
    v, ff = tensors(meshes, backend)
    assert mesh_to_fof(v, ff, 1, 8, fill="hybrid", backend=backend)[0, 4, 4].item() == pytest.approx(1.7, abs=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_ray_diagnostics(backend):
    from lib.fof import ray_diagnostics

    box = slab(-.5, .5, -.5, .5, -.5, .5)
    v, f = tensors(box, backend)
    rep = ray_diagnostics(v, f, 16, backend=backend)
    assert not rep["unbalanced"].any() and not rep["rules_disagree"].any() and rep["crossings"].max() == 2
    keep = box.face_normals[:, 2] < 0.9  # remove the camera-facing side: every ray only enters
    v, f = tensors(trimesh.Trimesh(box.vertices, box.faces[keep], process=False), backend)
    rep = ray_diagnostics(v, f, 16, backend=backend)
    assert float(rep["frac_unbalanced"]) == 1.0 and (rep["crossings"][rep["unbalanced"]] == 1).all()
    rep = ray_diagnostics(torch.zeros(0, 2, 3, 3, device=dev(backend)), None, 8, backend=backend)  # empty batch
    assert rep["unbalanced"].shape == (0, 8, 8) and rep["frac_unbalanced"].shape == (0,)
