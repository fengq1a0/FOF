import torch
import numpy as np
from skimage.measure import marching_cubes

def toDevice(sample, device):
    if torch.is_tensor(sample):
        return sample.to(device, non_blocking=True)
    elif sample is None or type(sample) == str:
        return sample
    elif isinstance(sample, dict):
        return {k: toDevice(v, device) for k, v in sample.items()}
    else:
        return [toDevice(s, device) for s in sample]

class Recon():
    def __init__(self, device) -> None:
        self.device = device
        z = (torch.arange(512, dtype=torch.float32, device=device)-255.5)/256
        z = torch.arange(16, dtype=torch.float32, device=device).view(1, 16) * z.view(512, 1) * np.pi
        z = torch.cat([
            z,
            z-(np.pi/2)
        ], dim=1)
        self.z = torch.cos(z)
        
    def decode(self, ceof):
        with torch.no_grad():
            res = torch.einsum("dc, chw -> dhw", self.z, ceof)
            v, f, _, _ = marching_cubes(res.cpu().numpy(), level = 0.5)
            v += 0.5
            v = v/256 - 1
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
