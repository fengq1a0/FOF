import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.network.HRNet import HRNetV2_W32 as network
from torchmcubes import marching_cubes
import tqdm


class Warper(nn.Module):
    def __init__(self, device) -> None:
        super(Warper, self).__init__()
        self.device = device
        z = (torch.arange(256, dtype=torch.float32, device=device)-127.5)/128
        z = torch.arange(16, dtype=torch.float32, device=device).view(1, 16) * z.view(256, 1) * np.pi
        z = torch.cat([
            z,
            z-(np.pi/2)
        ], dim=1)
        self.z = torch.cos(z)
        self.net = network()

    def forward(self, img, mask):
        ceof = self.net((img/127.5 - 1)*mask)*mask
        ceof = F.interpolate(ceof, size=(256, 256), align_corners =False,mode="bilinear")
        res = torch.einsum("dc, bchw -> bdhw", self.z, ceof)
        return res



torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0")


engine = Warper(device).to(device)
state_dict = torch.load("ckpt/base/000.pth")
engine.net.load_state_dict(state_dict["net"])




img = cv2.imread("input/tmp.png", -1)
mask = img[:,:,3:4]
img = img[:,:,:3]
img = torch.from_numpy(img.transpose((2,0,1)))[None]
mask = torch.from_numpy(mask.transpose((2,0,1)))[None]
img = img.to(device)
mask = mask.to(device) > 127


with torch.no_grad():
    tr = torch.jit.trace(engine, (img, mask))

    for _ in tqdm.tqdm(range(1000)):
        output = tr(img, mask)
        v, f = marching_cubes(output[0], 0.5)