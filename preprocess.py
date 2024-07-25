import os
import numpy as np
import tqdm
import open3d as o3d
import cyminiball as cb

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

def get_C(v):
    v_tmp = v.copy()

    ma = v_tmp[:,1].max()
    mi = v_tmp[:,1].min()
    scale = 1.8/(ma-mi)
    center_y = (ma+mi) / 2
    
    v_tmp[:,1] = 0
    C, r2 = cb.compute(v_tmp)
    C[1] = center_y
    C = C.astype(np.float32)
    
    return C, scale


def work(mesh_in, mesh_out, smpl_in, smpl_out, num_f):
    # 0. get v, f    (vt, ft are not needed)
    v,f,vt,ft = load_obj(mesh_in)

    # 1. merge points with the same coordinates
    tmp = []
    for i in range(v.shape[0]):
        tmp.append((v[i,0], v[i,1], v[i,2], i))
    tmp = sorted(tmp)
    mapping = np.zeros(v.shape[0], dtype=np.int32)
    mapping[tmp[0][3]] = tmp[0][3]
    now = tmp[0][3]
    for i in range(1,v.shape[0]):
        if (tmp[i][0] == tmp[i-1][0] and
            tmp[i][1] == tmp[i-1][1] and
            tmp[i][2] == tmp[i-1][2]):
            mapping[tmp[i][3]] = now
        else:
            mapping[tmp[i][3]] = tmp[i][3]
            now = tmp[i][3]
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
    # also clean ft if you have it
    if ft is not None:
        new_ft = []
        for tmp in range(ff.shape[0]):
            if triangles_to_remove[tmp]:
                continue
            else:
                new_ft.append(ft[tmp])
        new_ft = np.array(new_ft, dtype=np.int32)

    # 3. normalize the scanned mesh
    new_v = np.asarray(mesh.vertices).astype(np.float32)
    new_f = np.asarray(mesh.triangles).astype(np.float32)
    C, scale = get_C(new_v)
    new_v = (new_v - C) * scale

    # 5. simplify mesh, compute vertex normal, and then save the mesh.
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

    # 6. do it to the smpl mesh also
    v,f,vt,ft = load_obj(smpl_in)
    v = (v - C) * scale
    mmmm = o3d.geometry.TriangleMesh()
    mmmm.vertices = o3d.utility.Vector3dVector(v)
    mmmm.triangles = o3d.utility.Vector3iVector(f)
    mmmm.compute_vertex_normals()
    o3d.io.write_triangle_mesh(smpl_out, mmmm)


if __name__ == "__main__":
    base_mesh = "/mnt/16T/fq/datasets/THuman2.0/"
    base_smpl = "/mnt/16T/fq/datasets/THUman.20 Release Smpl-X Paras/"
    base_out = "/mnt/16T/fq/THuman21/"

    os.makedirs(os.path.join(base_out, "smpl-180"), exist_ok=True)
    os.makedirs(os.path.join(base_out, "100k-180"), exist_ok=True)

    a = sorted(os.listdir(base_mesh))
    for name in tqdm.tqdm(a):
        work(os.path.join(base_mesh, name, name+".obj"),
             os.path.join(base_out, "100k-180", name+".ply"),
             os.path.join(base_smpl, name, "mesh_smplx.obj"),
             os.path.join(base_out, "smpl-180", name+".ply"),
             100000)