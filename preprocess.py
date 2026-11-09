import os
import numpy as np
import tqdm
import open3d as o3d
from scipy.spatial import ConvexHull, QhullError

from lib.fof import orient_outward

def load_obj(filename):
    v = []
    f = []    
    vt = []
    ft = []
    with open(filename, "r") as fi:
        lines = fi.readlines()
        for line in lines:
            tmp = line.split()
            if tmp[0] == "v":
                v.append([float(tmp[1]), float(tmp[2]), float(tmp[3])])
            elif tmp[0] == "vt":
                vt.append([float(tmp[1]), float(tmp[2])])
            elif tmp[0] =="f":
                ttt = [ tmp[1].split("/"),
                        tmp[2].split("/"),
                        tmp[3].split("/")]
                f.append([  int(ttt[0][0])-1, 
                            int(ttt[1][0])-1, 
                            int(ttt[2][0])-1])
                if len(ttt[0]) > 1:
                    ft.append([ int(ttt[0][1])-1, 
                                int(ttt[1][1])-1, 
                                int(ttt[2][1])-1])
    v = np.array(v, dtype=np.float32)
    f = np.array(f, dtype=np.uint32)
    if len(vt)==0:
        vt = None
    else:
        vt = np.array(vt, dtype=np.float32)
    if len(ft)==0:
        ft = None
    else:
        ft = np.array(ft, dtype=np.uint32)
    return v, f, vt, ft

def min_enclosing_circle(p, seed=0):
    """Smallest circle containing the 2D points p [N, 2] -> (centre [2], radius^2).

    Randomized incremental algorithm (Welzl) on the convex hull, expected linear time.
    """
    p = np.asarray(p, dtype=np.float64)
    o = (p.min(0) + p.max(0)) / 2  # work about the centre of the bounding box (round-off)
    p = p - o
    if len(p) > 3:
        try:
            p = p[ConvexHull(p).vertices]
        except QhullError:  # all points on a line: its two extreme points
            d = p - p[0]
            t = d @ d[np.argmax((d * d).sum(1))]
            p = p[[np.argmin(t), np.argmax(t)]]
    p = p[np.random.default_rng(seed).permutation(len(p))]
    tol = 1e-12 * float((p * p).sum(1).max())  # relative to the size of the footprint

    def circle(*q):
        if len(q) == 1:
            return q[0], 0.0
        if len(q) == 3:
            b, c = q[1] - q[0], q[2] - q[0]
            d = 2 * (b[0] * c[1] - b[1] * c[0])
            if d != 0:
                u = np.array([c[1] * (b @ b) - b[1] * (c @ c), b[0] * (c @ c) - c[0] * (b @ b)]) / d
                return q[0] + u, float(u @ u)
            q = max(((q[0], q[1]), (q[0], q[2]), (q[1], q[2])), key=lambda e: (e[0] - e[1]) @ (e[0] - e[1]))
        c = (q[0] + q[1]) / 2
        return c, float((q[0] - c) @ (q[0] - c))

    def outside(c, r2, q):
        return (q - c) @ (q - c) > r2 + tol

    c, r2 = circle(p[0])
    for i in range(1, len(p)):
        if outside(c, r2, p[i]):
            c, r2 = circle(p[i])
            for j in range(i):
                if outside(c, r2, p[j]):
                    c, r2 = circle(p[i], p[j])
                    for k in range(j):
                        if outside(c, r2, p[k]):
                            c, r2 = circle(p[i], p[j], p[k])
    return c + o, r2


def get_C(v):
    """Centre (smallest circle around the x-z footprint, middle of the height) and scale to a height of 1.8."""
    ma = v[:,1].max()
    mi = v[:,1].min()
    scale = 1.8/(ma-mi)
    (cx, cz), _ = min_enclosing_circle(v[:, [0, 2]])
    C = np.array([cx, (ma+mi) / 2, cz], dtype=np.float32)
    return C, scale


def work(mesh_in, mesh_out, smpl_in, smpl_out, num_f):
    """Cleans and normalizes one scan and its SMPL(-X) mesh; returns the number of reoriented faces."""
    # 0. get v, f    (vt, ft are not needed)
    v,f,vt,ft = load_obj(mesh_in)

    # 1. merge points with the same coordinates (map to the first occurrence)
    _, first, inverse = np.unique(v, axis=0, return_index=True, return_inverse=True)
    mapping = first[inverse.reshape(-1)].astype(np.int32)
    ff = mapping[f]

    # 2. take the largest connectivity component
    # (remove the floating fragments)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(ff)

    triangle_clusters, cluster_n_triangles, cluster_area = (
    mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    tmp = cluster_n_triangles.max()
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < tmp
    mesh.remove_triangles_by_mask(triangles_to_remove)
    # drop the vertices of the removed fragments (and the merged duplicates),
    # otherwise they still take part in the normalization below
    mesh.remove_unreferenced_vertices()

    # 3. consistent, outward-facing orientation: the FOF fill rules tell entries from exits by it.
    # (open3d's orient_triangles alone keeps parts that are inside out and gives up on the whole
    # mesh at the first non-manifold edge, see lib/fof/repair.py)
    tri = np.asarray(mesh.triangles)
    oriented = orient_outward(np.asarray(mesh.vertices), tri)
    num_reoriented = int((oriented != tri).any(1).sum())
    mesh.triangles = o3d.utility.Vector3iVector(oriented)

    # 4. normalize the scanned mesh (height 1.8, centred)
    new_v = np.asarray(mesh.vertices).astype(np.float32)
    new_f = np.asarray(mesh.triangles).astype(np.int32)
    C, scale = get_C(new_v)
    new_v = (new_v - C) * scale

    # 5. subdivide / simplify to num_f faces, compute vertex normals, save.
    mmmm = o3d.geometry.TriangleMesh()
    mmmm.vertices = o3d.utility.Vector3dVector(new_v)
    mmmm.triangles = o3d.utility.Vector3iVector(new_f)
    tmp = new_f.shape[0]
    cnt = 0
    while tmp < num_f:
        tmp *= 4
        cnt += 1
    if cnt != 0:
        mmmm = mmmm.subdivide_midpoint(number_of_iterations=cnt)
    if np.asarray(mmmm.triangles).shape[0] > num_f:
        mmmm = mmmm.simplify_quadric_decimation(target_number_of_triangles=num_f)
    mmmm.compute_vertex_normals()
    o3d.io.write_triangle_mesh(mesh_out, mmmm)

    # 6. apply the same normalization to the SMPL(-X) mesh
    v,f,vt,ft = load_obj(smpl_in)
    v = (v - C) * scale
    mmmm = o3d.geometry.TriangleMesh()
    mmmm.vertices = o3d.utility.Vector3dVector(v)
    mmmm.triangles = o3d.utility.Vector3iVector(f)
    mmmm.compute_vertex_normals()
    o3d.io.write_triangle_mesh(smpl_out, mmmm)
    return num_reoriented


if __name__ == "__main__":
    base_mesh = "/mnt/16T/fq/datasets/THuman2.0/"
    base_smpl = "/mnt/16T/fq/datasets/THUman.20 Release Smpl-X Paras/"
    base_out = "/mnt/16T/fq/THuman21/"

    os.makedirs(os.path.join(base_out, "smpl-180"), exist_ok=True)
    os.makedirs(os.path.join(base_out, "100k-180"), exist_ok=True)

    a = sorted(os.listdir(base_mesh))
    for name in tqdm.tqdm(a):
        n = work(os.path.join(base_mesh, name, name+".obj"),
                 os.path.join(base_out, "100k-180", name+".ply"),
                 os.path.join(base_smpl, name, "mesh_smplx.obj"),
                 os.path.join(base_out, "smpl-180", name+".ply"),
                 100000)
        if n:
            tqdm.tqdm.write("%s: reoriented %d faces" % (name, n))