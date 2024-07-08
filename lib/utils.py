import torch
import torch.nn.functional as F
import numpy as np
from skimage.measure import marching_cubes
try:
    from .mc import marching_cubes as mmcc
    from .mc import smooth
    lossless_mc = True
except:
    lossless_mc = False

def toDevice(sample, device):
    if torch.is_tensor(sample):
        return sample.to(device, non_blocking=True)
    elif isinstance(sample, dict):
        return {k: toDevice(v, device) for k, v in sample.items()}
    elif isinstance(sample, list):
        return [toDevice(s, device) for s in sample]
    elif isinstance(sample, tuple):
        return tuple([toDevice(s, device) for s in sample])
    else:
        return sample

class Recon():
    def __init__(self, device, dim) -> None:
        self.device = device
        z = (torch.arange(512, dtype=torch.float32, device=device)+0.5)/512
        tmp = torch.arange(dim, dtype=torch.float32, device=device).view(1, dim)
        z =  tmp * z.view(512, 1) * np.pi
        self.z = torch.cos(z)
        self.z[:,0] = 0.5

    def decode(self, ceof, resolution=512):
        with torch.no_grad():
            res = torch.einsum("dc, chw -> dhw", self.z, ceof)
            res = F.interpolate(res[None,None],
                                (resolution,resolution,resolution),
                                mode="trilinear",
                                align_corners=False)[0,0]
            v, f, _, _ = marching_cubes(res.cpu().numpy(), level = 0.5)
        v += 0.5
        v = v/(resolution/2) - 1
        v[:,1] *= -1
        vv = np.zeros_like(v)
        vv[:,0] = v[:,2]
        vv[:,1] = v[:,1]
        vv[:,2] = v[:,0]

        ff = np.zeros_like(f)
        ff[:,0] = f[:,0]
        ff[:,1] = f[:,2]
        ff[:,2] = f[:,1]
        return vv, ff
    
    def decode_lossless(self, ceof):
        if not lossless_mc:
            print("please compile mc module first!")
            return None, None
        with torch.no_grad():
            res = torch.einsum("dc, chw -> dhw", self.z, ceof)
            v,f, vn, not_z = mmcc(res.cpu().numpy(), 0.5)
            smooth(v,f,not_z)
        return v, f
