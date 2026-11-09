"""Marching-cubes tables generated from first principles (nothing hand-typed).

Index space of a volume ``vol[d, h, w]``; a cell's corners and edges:
  * corner k has offsets ``(dx, dy, dz) = (k & 1, k >> 1 & 1, k >> 2 & 1)`` along (w, h, d);
  * edge e has axis ``a = e // 4`` (0: w, 1: h, 2: d) and runs from corner ``EDGE_CORNERS[e][0]``
    (its start point) to ``EDGE_CORNERS[e][1]``;
  * face f = 2 * axis + side holds the corners whose offset along ``axis`` equals ``side``,
    listed in cyclic order in ``FACE_CORNERS[f]``.
A corner is *inside* when its value is greater than the iso-level (bit k of the cube's configuration).

On a face whose corners alternate inside/outside (an ambiguous face) the contour can connect either
diagonal.  The choice is made at run time from the face's own four values (asymptotic decider, see
``connect_inside``), so the two cells sharing the face always agree and the surface has no cracks.
Hence every configuration has one triangulation per assignment of its ambiguous faces:
``TRIS[BASE[cfg] + bits]`` where bit j is the decision of the j-th ambiguous face (in face order).

Triangulations are built by cutting every face into iso-contour segments, orienting each segment so
that the inside of the face lies on its right when seen from outside the cube, chaining the segments
into loops and fan-triangulating every loop.  Triangles are counter-clockwise around normals that
point from inside (high values) to outside in the right-handed (w, h, d) frame.
"""

import functools

import numpy as np

def _corner_offset(k):
    return np.array([k & 1, (k >> 1) & 1, (k >> 2) & 1], dtype=np.float64)


def _edge_corners():
    out = []
    for e in range(12):
        a, r = divmod(e, 4)
        u, v = r & 1, (r >> 1) & 1
        if a == 0:
            k0 = 2 * u + 4 * v
            out.append((k0, k0 + 1))
        elif a == 1:
            k0 = u + 4 * v
            out.append((k0, k0 + 2))
        else:
            k0 = u + 2 * v
            out.append((k0, k0 + 4))
    return out


EDGE_CORNERS = _edge_corners()
_EDGE_OF = {frozenset(c): e for e, c in enumerate(EDGE_CORNERS)}


def _face_corners():
    faces = []
    for a in range(3):
        b, c = [x for x in range(3) if x != a]
        for s in range(2):
            cyc = []
            for ob, oc in ((0, 0), (1, 0), (1, 1), (0, 1)):
                off = [0, 0, 0]
                off[a], off[b], off[c] = s, ob, oc
                cyc.append(off[0] + 2 * off[1] + 4 * off[2])
            faces.append(cyc)
    return faces


FACE_CORNERS = _face_corners()


def _face_normal(f):
    n = np.zeros(3)
    n[f // 2] = 1.0 if f % 2 else -1.0
    return n


def ambiguous_faces(cfg):
    """Bit mask of the faces of ``cfg`` whose corners alternate inside/outside."""
    mask = 0
    for f, cyc in enumerate(FACE_CORNERS):
        ins = [(cfg >> k) & 1 for k in cyc]
        if ins[0] == ins[2] and ins[1] == ins[3] and ins[0] != ins[1]:
            mask |= 1 << f
    return mask


def _face_segments(cfg, f, connect_inside):
    """Directed iso-contour segments (edge_from, edge_to) on face f."""
    cyc = FACE_CORNERS[f]
    ins = [(cfg >> k) & 1 for k in cyc]
    pos = [_corner_offset(k) for k in cyc]
    edges = [_EDGE_OF[frozenset((cyc[i], cyc[(i + 1) % 4]))] for i in range(4)]
    crossed = [i for i in range(4) if ins[i] != ins[(i + 1) % 4]]
    if not crossed:
        return []
    if len(crossed) == 2:
        pairs = [(crossed[0], crossed[1])]
        ref_list = [pos[ins.index(1)]]  # any inside corner lies on the inside side
    else:
        # 4 crossings: cut off either the two outside or the two inside corners
        cut = [i for i in range(4) if ins[i] == (0 if connect_inside else 1)]
        pairs = [((i - 1) % 4, i) for i in cut]
        centre = np.mean(pos, axis=0)
        ref_list = [centre if connect_inside else pos[i] for i in cut]
    n = _face_normal(f)
    out = []
    for (i, j), ref in zip(pairs, ref_list):
        ea, eb = edges[i], edges[j]
        a = 0.5 * sum(_corner_offset(k) for k in EDGE_CORNERS[ea])
        b = 0.5 * sum(_corner_offset(k) for k in EDGE_CORNERS[eb])
        # inside on the right when seen from outside: cross(b - a, ref - a) . n < 0
        if np.dot(np.cross(b - a, ref - a), n) > 0:
            ea, eb = eb, ea
        out.append((ea, eb))
    return out


def _edge_faces(e):
    """Faces (as a bit mask) that contain cube edge e."""
    k0, k1 = EDGE_CORNERS[e]
    return sum(1 << f for f, cyc in enumerate(FACE_CORNERS) if k0 in cyc and k1 in cyc)


_EDGE_FACES = [_edge_faces(e) for e in range(12)]
_EDGE_MID = [0.5 * (_corner_offset(EDGE_CORNERS[e][0]) + _corner_offset(EDGE_CORNERS[e][1])) for e in range(12)]


@functools.lru_cache(maxsize=None)
def _triangulate_loop_cached(loop):
    return _triangulate_loop(list(loop))


def _triangulate_loop(loop):
    """Minimum-weight triangulation of a loop that never connects two of its vertices lying on a
    common cube face by a diagonal.  Such a diagonal would lie in the face and coincide with a
    diagonal of the neighbouring cell (an edge shared by four triangles).  This can only happen
    for loops crossing an ambiguous face twice.  Returns None if no such triangulation exists."""
    n = len(loop)
    if n == 3:
        return [tuple(loop)]
    mid = [_EDGE_MID[e] for e in loop]

    def allowed(i, j):
        return j - i == 1 or (i == 0 and j == n - 1) or not (_EDGE_FACES[loop[i]] & _EDGE_FACES[loop[j]])

    inf = float("inf")
    cost = [[inf] * n for _ in range(n)]
    split = [[-1] * n for _ in range(n)]
    for i in range(n - 1):
        cost[i][i + 1] = 0.0
    for length in range(2, n):
        for i in range(n - length):
            j = i + length
            if not allowed(i, j):
                continue
            w = 0.0 if (i == 0 and j == n - 1) else float(np.linalg.norm(mid[i] - mid[j]))
            for k in range(i + 1, j):
                c = cost[i][k] + cost[k][j] + w
                if c < cost[i][j] - 1e-12:
                    cost[i][j], split[i][j] = c, k
    if cost[0][n - 1] == inf:
        return None
    tris, stack = [], [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        k = split[i][j]
        tris.append((loop[i], loop[k], loop[j]))  # increasing loop order keeps the orientation
        stack += [(i, k), (k, j)]
    return sorted(tris)


CENTRE = 12  # pseudo edge ids 12.. : extra vertices at the centroid of a loop (cell interior)


def triangulate(cfg, decisions):
    """Triangulation of configuration ``cfg``; ``decisions[f]`` says whether ambiguous face f
    connects its inside diagonal.  Returns ``(triangles, centre_loops)``: triangles are triples of
    edge ids, where id ``CENTRE + s`` denotes an extra vertex at the centroid of the loop
    ``centre_loops[s]`` (used only when a loop has no triangulation without face diagonals, the
    situation that MC33 resolves with its interior vertex)."""
    succ = {}
    for f in range(6):
        for ea, eb in _face_segments(cfg, f, decisions.get(f, False)):
            assert ea not in succ, "inconsistent segment orientation"
            succ[ea] = eb
    assert sorted(succ) == sorted(succ.values()), "segments do not form loops"
    tris, centres, seen = [], [], set()
    for start in sorted(succ):
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        e = succ[start]
        while e != start:
            loop.append(e)
            seen.add(e)
            e = succ[e]
        t = _triangulate_loop_cached(tuple(loop))
        if t is None:
            c = CENTRE + len(centres)
            centres.append(loop)
            t = [(loop[i], loop[(i + 1) % len(loop)], c) for i in range(len(loop))]
        tris += t
    return tris, centres


def build_tables():
    """Returns ``(amb [256] uint8, base [256] int32, ntri [V] uint8, tris [V, 3*max_tris] int8,
    ncen [V] uint8, cloop [V, max_centres] int32)``; ``cloop`` holds 12-bit masks of the edges of
    each centre vertex's loop."""
    amb = np.zeros(256, np.uint8)
    base = np.zeros(256, np.int32)
    rows = []
    for cfg in range(256):
        mask = ambiguous_faces(cfg)
        faces = [f for f in range(6) if mask >> f & 1]
        amb[cfg], base[cfg] = mask, len(rows)
        for bits in range(1 << len(faces)):
            rows.append(triangulate(cfg, {f: bool(bits >> j & 1) for j, f in enumerate(faces)}))
    max_tris = max(len(t) for t, _ in rows)
    max_cen = max(1, max(len(c) for _, c in rows))
    ntri = np.array([len(t) for t, _ in rows], np.uint8)
    tris = np.full((len(rows), 3 * max_tris), -1, np.int8)
    ncen = np.array([len(c) for _, c in rows], np.uint8)
    cloop = np.zeros((len(rows), max_cen), np.int32)
    for v, (t, c) in enumerate(rows):
        if t:
            tris[v, : 3 * len(t)] = np.array(t, np.int8).reshape(-1)
        for s, loop in enumerate(c):
            cloop[v, s] = sum(1 << e for e in loop)
    return amb, base, ntri, tris, ncen, cloop


AMB, BASE, NTRI, TRIS, NCEN, CLOOP = build_tables()
MAX_TRIS = TRIS.shape[1] // 3
MAX_CENTRES = CLOOP.shape[1]
