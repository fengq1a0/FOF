"""Tests for lib.fof.repair (offline orientation repair of ground-truth meshes)."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.fof import mesh_to_fof, orient_outward  # noqa: E402

trimesh = pytest.importorskip("trimesh")


def normals(v, f):
    p = np.asarray(v, dtype=np.float64)[f]
    return np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])


def outward(v, f, f_true):
    """Per face: True if ``f`` has the orientation of ``f_true``."""
    return (normals(v, f) * normals(v, f_true)).sum(1) > 0


def parts():
    sphere = trimesh.creation.icosphere(3, radius=0.3)
    torus = trimesh.creation.torus(0.35, 0.1)
    torus.apply_translation([0.2, 0.1, -0.3])
    box = trimesh.creation.box(extents=[0.3, 0.4, 0.2])
    box.apply_translation([-0.3, 0.0, 0.4])
    return trimesh.util.concatenate([sphere, torus, box]), len(sphere.faces), len(torus.faces)


def flip(f, mask):
    f = f.copy()
    f[mask] = f[mask][:, ::-1]
    return f


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_clean_meshes_are_unchanged(dtype):
    m, _, _ = parts()
    f = np.asarray(m.faces, dtype=dtype)
    out = orient_outward(m.vertices, f)
    assert out.dtype == dtype and np.array_equal(out, f)


def test_flipped_faces_and_inside_out_parts():
    m, n_sphere, n_torus = parts()
    v, f = np.asarray(m.vertices), np.asarray(m.faces)
    rng = np.random.default_rng(0)
    bad = flip(f, rng.random(len(f)) < 0.3)
    bad = flip(bad, np.arange(len(f)) >= n_sphere + n_torus)  # the box ends up mostly inside out
    out = orient_outward(v, bad)
    assert outward(v, out, f).all()
    assert np.array_equal(np.sort(out, 1), np.sort(f, 1))  # same triangles, same order
    assert outward(v, orient_outward(v, f[:, ::-1]), f).all()  # everything inside out


def test_non_manifold_edges_and_non_orientable_pieces_do_not_block_the_rest():
    m = trimesh.creation.icosphere(3, radius=0.5)
    v, f = np.asarray(m.vertices), np.asarray(m.faces)
    # a fin on the first edge (three faces on one edge) and a Moebius strip
    fin = v[f[0, :2]].mean(0) * 1.2
    n = 24
    u = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([np.cos(u), np.sin(u), np.zeros(n)], 1) * 0.3 + [0, 0, 0.7]
    w = np.stack([np.cos(u / 2) * np.cos(u), np.cos(u / 2) * np.sin(u), np.sin(u / 2)], 1) * 0.05
    strip = []
    for i in range(n):
        a, b, c, d = i, (i + 1) % n, n + i, n + (i + 1) % n
        if i == n - 1:  # the half twist: top and bottom swap where the strip closes
            b, d = n, 0
        strip += [[a, b, d], [a, d, c]]
    V = len(v)
    v2 = np.vstack([v, [fin], ring + w, ring - w])
    f2 = np.vstack([f, [[f[0, 1], f[0, 0], V]], np.array(strip) + V + 1])
    bad = flip(f2, np.random.default_rng(1).random(len(f2)) < 0.3)
    out = orient_outward(v2, bad)
    assert outward(v2, out[: len(f)], f).all()


def test_identical_vertices_are_welded():
    m = trimesh.creation.icosphere(3, radius=0.5)
    v = np.asarray(m.vertices)[m.faces].reshape(-1, 3)  # triangle soup: every face has its own vertices
    f = np.arange(len(v)).reshape(-1, 3)
    bad = flip(f, np.random.default_rng(2).random(len(f)) < 0.3)
    out = orient_outward(v, bad)
    assert outward(v, out, f).all()
    assert np.array_equal(np.sort(out, 1), f)  # vertex ids are kept


def test_repair_restores_the_fof_of_multi_part_rays():
    """Two parts on the same rays, the near one inside out: crossings E X X E, which no per-ray rule
    can tell from a duplicated layer plus a hole."""
    far = trimesh.creation.box(bounds=[[-0.5, -0.5, -0.8], [0.5, 0.5, -0.2]])
    near = trimesh.creation.box(bounds=[[-0.3, -0.3, 0.1], [0.3, 0.3, 0.7]])
    m = trimesh.util.concatenate([far, near])
    v, f = np.asarray(m.vertices), np.asarray(m.faces)
    bad = flip(f, np.arange(len(f)) >= len(far.faces))

    def fof(faces):
        return mesh_to_fof(torch.tensor(v, dtype=torch.float32), torch.tensor(faces), 8, 32, backend="torch")

    truth = fof(f)
    assert not torch.allclose(fof(bad), truth, atol=1e-4)
    assert torch.allclose(fof(orient_outward(v, bad)), truth, atol=1e-5)


def test_input_validation():
    v = np.zeros((3, 3))
    assert orient_outward(v, np.zeros((0, 3), np.int32)).shape == (0, 3)
    with pytest.raises(ValueError):
        orient_outward(v, [[0, 1, 3]])
    with pytest.raises(ValueError):
        orient_outward(v, [[0, 1, -1]])
    with pytest.raises(TypeError):
        orient_outward(v, [[0.0, 1.0, 2.0]])
    with pytest.raises(ValueError):
        orient_outward(v, [0, 1, 2])
