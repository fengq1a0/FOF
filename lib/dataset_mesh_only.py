import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np

import cv2
import open3d as o3d


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, 
                 path="/mnt/data/fq/THuman21/",
                 name_file = "train_th2.txt",
                 num_view=512,
                 use_normal=True,
                 use_smpl = True,
                 use_smpl_normal = False,
                 use_img = False) -> None:
        super().__init__()
        self.path = path
        self.num_view = num_view
        self.use_normal = use_normal
        self.use_smpl = use_smpl
        self.use_smpl_normal = use_smpl_normal
        self.use_img = use_img

        with open(os.path.join(self.path, name_file), "r") as f:
            self.nameList = f.read().split()
        
        self.v = torch.zeros((len(self.nameList), 100000, 3, 3))
        if self.use_normal:
            self.vn = torch.zeros((len(self.nameList), 100000, 3, 3))
        for i in range(len(self.nameList)):
            mesh = o3d.io.read_triangle_mesh(
                os.path.join(self.path, "100k-180", self.nameList[i]+".ply")
            )
            mesh_v = np.asarray(mesh.vertices)
            mesh_f = np.asarray(mesh.triangles)
            mesh_vn = np.asarray(mesh.vertex_normals)
            self.v[i,:mesh_f.shape[0]] = torch.from_numpy(mesh_v[mesh_f].astype(np.float32))
            if self.use_normal:
                self.vn[i,:mesh_f.shape[0]] = torch.from_numpy(mesh_vn[mesh_f].astype(np.float32))
        
        if self.use_smpl:
            self.smpl_v = torch.zeros((len(self.nameList), 13776, 3, 3))
        if self.use_smpl_normal:
            self.smpl_vn = torch.zeros((len(self.nameList), 13776, 3, 3))
        if self.use_smpl or self.use_smpl_normal:
            for i in range(len(self.nameList)):
                mesh = o3d.io.read_triangle_mesh(
                    os.path.join(self.path, "smpl-180", self.nameList[i]+".ply")
                )
                mesh_v = np.asarray(mesh.vertices)
                mesh_f = np.asarray(mesh.triangles)
                mesh_vn = np.asarray(mesh.vertex_normals)
                self.smpl_v[i] = torch.from_numpy(mesh_v[mesh_f].astype(np.float32))
                if self.use_smpl_normal:
                    self.smpl_vn[i] = torch.from_numpy(mesh_vn[mesh_f].astype(np.float32))

        Rs = []
        for i in range(self.num_view):
            y = 2*np.pi*i/self.num_view
            sy = np.sin(y)
            cy = np.cos(y)
            r = np.array([  [ cy, 0.0,  sy],
                            [0.0, 1.0, 0.0],
                            [-sy, 0.0,  cy],], dtype=np.float32)
            Rs.append(r)
        self.Rs = torch.from_numpy(np.array(Rs, dtype=np.float32))
    
    def open_db(self, id):
        np.random.seed(torch.initial_seed() % 2**32)

    def __len__(self):
        return len(self.nameList)*self.num_view

    def __getitem__(self, index):
        pid = index//self.num_view
        vid = index%self.num_view

        name = self.nameList[pid]
        v = self.v[pid]
        R = self.Rs[vid]

        res = {"name":"%s_%04d"%(name,vid), 
               "v":v, 
               "R":R}
        if self.use_normal:
            res["vn"] = self.vn[pid]
        if self.use_smpl:
            res["smpl_v"] = self.smpl_v[pid]
        if self.use_smpl_normal:
            res["smpl_vn"] = self.smpl_vn[pid]
        if self.use_img:
            img = cv2.imread(os.path.join(self.path, 
                                          "img-180", 
                                          "%s_%03d.png" % (name, 512*vid//self.num_view))
            )[:,:,::-1]
            img = torch.from_numpy(img.copy()).permute(2,0,1)
            res["img"] = img
        return res



def projection(v, calib):
    tmp = np.ones((v.shape[0], 4), dtype=np.float32)
    tmp[:,:3] = v*100
    tmp = tmp @ calib.T
    tmp = tmp[:,:3] / tmp[:,3:4]
    return tmp

def load_obj(filename):
    v = []
    f = []    
    with open(filename, "r") as fi:
        lines = fi.readlines()
        for line in lines:
            tmp = line.split()
            if tmp[0] == "v":
                v.append([float(tmp[1]), float(tmp[2]), float(tmp[3])])
            elif tmp[0] =="f":
                f.append([  int(tmp[1].split("/")[0])-1, 
                            int(tmp[2].split("/")[0])-1, 
                            int(tmp[3].split("/")[0])-1])
    v = np.array(v, dtype=np.float32)
    f = np.array(f, dtype=np.int32)
    return v, f

def get_vn(v, f):
    mmmm = o3d.geometry.TriangleMesh()
    mmmm.vertices = o3d.utility.Vector3dVector(v)
    mmmm.triangles = o3d.utility.Vector3iVector(f)
    mmmm.compute_vertex_normals()
    vn = np.asarray(mmmm.vertex_normals).astype(np.float32)
    return vn 

class CapeTestSet(torch.utils.data.Dataset):
    def __init__(self, 
                 path="/mnt/data/fq/cape/",
                 name_file = "all.txt",
                 view = [0, 120, 240]) -> None:
        super().__init__()
        self.path = path
        self.view = view

        with open(os.path.join(self.path, "cape", name_file), "r") as f:
            self.nameList = f.read().split()
    
    def open_db(self, id):
        np.random.seed(torch.initial_seed() % 2**32)

    def __len__(self):
        return len(self.nameList)*len(self.view)

    def __getitem__(self, index):
        pid = index//len(self.view)
        vid = index%len(self.view)

        name = self.nameList[pid]

        calib_data = np.loadtxt(
            os.path.join(self.path, name.replace("cape/","cape_3views/"), "calib", "%03d.txt" % self.view[vid]),
            dtype=float
        )
        extrinsic = calib_data[:4, :4]
        intrinsic = calib_data[4:8, :4]
        intrinsic[1] *= -1
        calib_mat = np.matmul(intrinsic, extrinsic)


        smpl_v, smpl_f = load_obj(
            os.path.join(self.path, name.replace("cape/","cape/smpl/")+".obj")
        )

        v, f = load_obj(
            os.path.join(self.path, name.replace("cape/","cape/scans/")+".obj")
        )

        v = projection(v, calib_mat)
        smpl_v = projection(smpl_v, calib_mat)

        vn  = get_vn(v, f)
        smpl_vn  = get_vn(smpl_v, smpl_f)

        v = torch.from_numpy(v[f].astype(np.float32))
        vn = torch.from_numpy(vn[f].astype(np.float32))
        
        smpl_v = torch.from_numpy(smpl_v[smpl_f].astype(np.float32))
        smpl_vn = torch.from_numpy(smpl_vn[smpl_f].astype(np.float32))

        img = cv2.imread(
            os.path.join(self.path, name.replace("cape/","cape_3views/"), 
                         "render", "%03d.png" % self.view[vid])
        )[:,:,::-1]
        img = torch.from_numpy(img.copy()).permute(2,0,1)/127.5-1

        msk = cv2.imread(
            os.path.join(self.path, name.replace("cape/","cape_3views/"), 
                         "mask", "%03d.png" % self.view[vid]), -1
        )
        msk = torch.from_numpy(msk)[None] > 127.5

        img = img*msk
        
        
        return {
            "name":"%s_%03d" % (name.replace("cape/",""), vid),
            "img": img,
            "msk": msk,
            "v" : v,
            "vn" : vn,
            "smpl_v" : smpl_v,
            "smpl_vn" : smpl_vn
        }

    
if __name__ == "__main__":
    t = CapeTestSet()
    DL = torch.utils.data.DataLoader(
        t, batch_size=3,
        shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True,
        worker_init_fn=t.open_db
    )
    import tqdm
    for i in tqdm.tqdm(DL):
        pass
        

