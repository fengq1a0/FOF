"""Offline repair of ground-truth meshes before encoding.

The fill rules of :func:`mesh_to_fof` read the orientation of every crossing (entry or exit).
Along a single ray a flipped part can look exactly like other defects (``E X X E`` is a
duplicated layer plus a hole, or a second part turned inside out), so no per-ray rule can
undo wrong orientation everywhere.  Orientation is a property of the whole surface and is
repaired here, once per mesh.
"""

import numpy as np


def orient_outward(verts, faces) -> np.ndarray:
    """Returns ``faces`` reoriented counter-clockwise seen from outside.

    1. Consistent winding: two faces sharing an edge must traverse it in opposite directions.
       The orientation of one face per part is propagated along a BFS tree over the manifold
       edges (edges of exactly two faces), so non-manifold edges and non-orientable pieces
       don't stop the repair of the rest of the mesh.
    2. Outward: every part (faces connected through manifold edges) whose signed volume about
       its centroid is negative is flipped.  Parts are taken to be solids: a surface meant to
       bound a cavity would be turned outwards.

    Vertices with identical coordinates count as one vertex (e.g. along UV seams); the result
    keeps the vertex ids and the order of the faces.  Clean, outward meshes are returned
    unchanged.  Unlike ``open3d``'s ``orient_triangles``, which only makes neighbours agree
    and gives up on the whole mesh at the first non-manifold edge.

    Args:
        verts: ``[V, 3]`` vertex positions (array-like).
        faces: ``[F, 3]`` vertex indices (integer array-like).
    Returns:
        ``[F, 3]`` numpy array with the dtype of ``faces``.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import breadth_first_order, connected_components

    v = np.asarray(verts, dtype=np.float64)
    f = np.array(faces, copy=True)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"expected verts [V, 3] and faces [F, 3], got {v.shape} and {f.shape}")
    if not np.issubdtype(f.dtype, np.integer):
        raise TypeError(f"faces must be integers, got {f.dtype}")
    F = len(f)
    if F == 0:
        return f
    if f.min() < 0 or f.max() >= len(v):
        raise ValueError("face indices out of range")

    # connectivity on welded vertices; half-edge h = (a[h], b[h]) belongs to face h // 3
    _, vid = np.unique(v, axis=0, return_inverse=True)
    g = vid.reshape(-1)[f].astype(np.int64)
    a, b = g.reshape(-1), np.roll(g, -1, axis=1).reshape(-1)
    h = np.flatnonzero(a != b)
    key = np.minimum(a[h], b[h]) * (int(g.max()) + 1) + np.maximum(a[h], b[h])
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    two = cnt[inv] == 2
    h = h[two][np.argsort(inv[two], kind="stable")].reshape(-1, 2)  # both half-edges of each manifold edge
    fa, fb = h[:, 0] // 3, h[:, 1] // 3
    rel = a[h[:, 0]] == a[h[:, 1]]  # same direction along the shared edge: opposite orientation
    keep = fa != fb
    fa, fb, rel = fa[keep], fb[keep], rel[keep]
    _, first = np.unique(np.minimum(fa, fb) * F + np.maximum(fa, fb), return_index=True)
    fa, fb, rel = fa[first], fb[first], rel[first]  # one relation per pair of neighbouring faces

    # BFS forest: a virtual node F links the first face of every part; edge data = rel + 1
    n_parts, part = connected_components(csr_matrix((np.ones(len(fa)), (fa, fb)), shape=(F, F)), directed=False)
    roots = np.unique(part, return_index=True)[1]
    graph = csr_matrix(
        (np.concatenate([rel, rel, np.zeros(n_parts, bool)]).astype(np.int8) + 1,
         (np.concatenate([fa, fb, np.full(n_parts, F)]), np.concatenate([fb, fa, roots]))),
        shape=(F + 1, F + 1),
    )
    _, pred = breadth_first_order(graph, F, directed=True, return_predecessors=True)
    up = np.append(pred[:F], F)
    flip = np.append(np.asarray(graph[up[:F], np.arange(F)]).reshape(-1) == 2, False)
    # flip relative to the part's root = parity of the tree path, by pointer jumping
    while True:
        up2 = up[up]
        if np.array_equal(up2, up):
            break
        flip ^= flip[up]
        up = up2
    flip = flip[:F]
    f[flip] = f[flip][:, [0, 2, 1]]

    # outward: signed volume of every part about its centroid
    p = v[f]
    n = np.bincount(part, minlength=n_parts)[:, None]
    centre = np.stack([np.bincount(part, p[:, :, k].mean(1), minlength=n_parts) for k in range(3)], 1) / n
    q = p - centre[part][:, None]
    vol = np.einsum("ij,ij->i", q[:, 0], np.cross(q[:, 1], q[:, 2]))
    inside_out = (np.bincount(part, vol, minlength=n_parts) < 0)[part]
    f[inside_out] = f[inside_out][:, [0, 2, 1]]
    return f
