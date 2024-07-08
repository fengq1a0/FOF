import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np
import open3d as o3d

from lib.utils import toDevice, Recon
from lib.HRNet import get_FOF as Network
from lib.fof import FOF_Normal
from lib.FBNet import define_G


def test(args, model, data, ee, fof_render, netF, netB):
    model.eval()
    netF.eval()
    netB.eval()
    with torch.no_grad():
        data = toDevice(data,0)

        fof_smpl, dF, dB, snF, snB = fof_render(data["smpl_v"], data["smpl_vn"], 16)

        msk = data["image"][:,3:4] > 127.5
        img = data["image"][:,0:3] / 127.5 - 1
        img = img * msk

        nF = netF(torch.cat((img,snF), dim=1))
        nB = netB(torch.cat((img,snB), dim=1))
        nF = nF / torch.norm(nF, dim=1, keepdim=True)
        nB = nB / torch.norm(nB, dim=1, keepdim=True)
        nF = nF*msk
        nB = nB*msk
        img = torch.cat((nF, nB, fof_smpl), dim=1)

        fof = model(img)
        fof = fof * msk

        v,f = ee.decode(fof[0], args["res"])
        tmp = o3d.geometry.TriangleMesh()
        tmp.vertices = o3d.utility.Vector3dVector(v)
        tmp.triangles = o3d.utility.Vector3iVector(f)
        o3d.io.write_triangle_mesh(os.path.join(args["output"], data["name"]), tmp)



       
def main(args):
    print('Resuming from ', args["ckpt"])

    # models
    model = Network(i=args["i"], c=args["c"], dim=args["o"]).to(0)
    state_dict = torch.load(args["ckpt"], map_location="cpu")
    model.load_state_dict(state_dict["model"])
    del state_dict
    
    netF = define_G(6, 3, 64, "global", 4, 9, 1, 3, "instance").to(0)
    state_dict = torch.load("./ckpt/netF.pth", map_location="cpu")
    netF.load_state_dict(state_dict["model"])
    del state_dict

    netB = define_G(6, 3, 64, "global", 4, 9, 1, 3, "instance").to(0)
    state_dict = torch.load("./ckpt/netB.pth", map_location="cpu")
    netB.load_state_dict(state_dict["model"])
    del state_dict

    fof_render = FOF_Normal(1,512).to(0)
    ee = Recon(0, args["o"])


    # loop
    os.makedirs(args["output"], exist_ok=True)
    smplx_dir = sorted(os.listdir(os.path.join(args["input"], "smplx")))
    image_dir = sorted(os.listdir(os.path.join(args["input"], "image")))
    for i in range(len(smplx_dir)):
        # image
        image = os.path.join(args["input"], "image", image_dir[i])
        image = torch.from_numpy(
            cv2.cvtColor(cv2.imread(image, -1), cv2.COLOR_BGRA2RGBA)
        ).permute(2,0,1)[None]
        # smplx (smpl is also fine)
        smplx = os.path.join(args["input"], "smplx", smplx_dir[i])
        smplx = o3d.io.read_triangle_mesh(smplx)
        smplx.compute_vertex_normals()
        smpl_v = np.asarray(smplx.vertices)
        smpl_f = np.asarray(smplx.triangles)
        smpl_vn = np.asarray(smplx.vertex_normals)
        smpl_v = torch.from_numpy(smpl_v[smpl_f]).float()[None]
        smpl_vn = torch.from_numpy(smpl_vn[smpl_f]).float()[None]
        data = {
            "name"  : image_dir[i][:-4] + ".ply",
            "image" : image,
            "smpl_v"  : smpl_v,
            "smpl_vn" : smpl_vn
        }
        test(args, model, data, ee, fof_render, netF, netB)

if __name__ == '__main__':
    #-------------cfg here-------------
    args = {
        "input" : "/mnt/fq_data/monocular-3D-human-reconstruction/evaluation/TH21/",
        "output": "/mnt/fq_data/FOF/tmp_32_6",
        "ckpt"  : "/mnt/fq_data/FOF/ckpt/FOF-X/006.pth",
        "i"     : 6+16,
        "c"     : [32,64,128,256],#[48, 96, 192, 384],
        "o"     : 128,
        "res"   : 512
    }
    main(args)


