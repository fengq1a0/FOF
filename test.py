import os

import cv2
import numpy as np
import open3d as o3d
import torch

from lib.FBNet import define_G
from lib.fof import fof_to_mesh, render_fof
from lib.HRNet import get_FOF as Network
from lib.utils import toDevice


def test(args, model, data, netF, netB):
    model.eval()
    netF.eval()
    netB.eval()
    with torch.no_grad():
        data = toDevice(data, args["dev"])
        # SMPL-X guidance at the input image resolution, with the fill rule used in training
        smpl = render_fof(data["smpl_v"], data["smpl_f"], data["smpl_vn"], 16, tuple(data["image"].shape[-2:]),
                          fill=args["fill"])

        msk = data["image"][:, 3:4] > 127.5
        img = data["image"][:, 0:3] / 127.5 - 1
        img = img * msk

        nF = netF(torch.cat((img, smpl.normal_front), dim=1))
        nB = netB(torch.cat((img, smpl.normal_back), dim=1))
        nF = nF / torch.norm(nF, dim=1, keepdim=True)
        nB = nB / torch.norm(nB, dim=1, keepdim=True)
        nF = nF * msk
        nB = nB * msk

        fof = model(torch.cat((nF, nB, smpl.fof), dim=1))
        fof = fof * msk

    if args["lossless"]:  # native pixel grid, anchored least-squares smoothing
        v, f = fof_to_mesh(fof[0], depth_res=args["res"], lossless=True)
    else:  # marching cubes on a res^3 grid
        v, f = fof_to_mesh(fof[0], depth_res=args["res"], resolution=args["res"])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(f.astype(np.int32))
    o3d.io.write_triangle_mesh(os.path.join(args["output"], data["name"]), mesh)


def load_weights(net, path):
    state_dict = torch.load(path, map_location="cpu")
    net.load_state_dict(state_dict["model"])
    return net


def main(args):
    print("Resuming from ", args["ckpt"])
    dev = args["dev"]
    model = load_weights(Network(i=args["i"], c=args["c"], dim=args["o"]), args["ckpt"]).to(dev)
    netF = load_weights(define_G(6, 3, 64, "global", 4, 9, 1, 3, "instance"), "./ckpt/netF.pth").to(dev)
    netB = load_weights(define_G(6, 3, 64, "global", 4, 9, 1, 3, "instance"), "./ckpt/netB.pth").to(dev)

    os.makedirs(args["output"], exist_ok=True)
    smplx_dir = sorted(os.listdir(os.path.join(args["input"], "smplx")))
    image_dir = sorted(os.listdir(os.path.join(args["input"], "image")))
    assert len(smplx_dir) == len(image_dir), "expects one SMPL-X mesh per image"
    for i in range(len(smplx_dir)):
        # RGBA image: the alpha channel is the foreground mask
        image = cv2.imread(os.path.join(args["input"], "image", image_dir[i]), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[2] != 4:
            raise ValueError(f"{image_dir[i]}: expected an RGBA image with the mask in the alpha channel")
        image = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)).permute(2, 0, 1)[None]
        # SMPL-X (SMPL is also fine), normalized like the training data
        smplx = o3d.io.read_triangle_mesh(os.path.join(args["input"], "smplx", smplx_dir[i]))
        smplx.compute_vertex_normals()
        data = {
            "name": os.path.splitext(image_dir[i])[0] + ".ply",
            "image": image,
            "smpl_v": torch.from_numpy(np.asarray(smplx.vertices)).float()[None],
            "smpl_f": torch.from_numpy(np.asarray(smplx.triangles)).long(),
            "smpl_vn": torch.from_numpy(np.asarray(smplx.vertex_normals)).float()[None],
        }
        test(args, model, data, netF, netB)


if __name__ == '__main__':
    # -------------cfg here-------------
    args = {
        "input": "/mnt/fq_data/monocular-3D-human-reconstruction/evaluation/TH21/",
        "output": "/mnt/fq_data/FOF/tmp_32_6",
        "ckpt": "/mnt/fq_data/FOF/ckpt/FOF-X/006.pth",
        "i": 6 + 16,
        "c": [32, 64, 128, 256],  # [48, 96, 192, 384] for model48.pth
        "o": 128,
        "res": 512,  # output mesh resolution (marching-cubes grid)
        "lossless": False,  # True: FOF-X anchored least-squares decoding (real time on the GPU)
        "fill": "robust",  # the released checkpoints were trained with the original rule
        "dev": 0,
    }
    main(args)
