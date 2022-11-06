import numpy as np
from numba import njit
@njit
def tri_occ(pos, ind, val, num):
    ceof = np.zeros((512, 512, num*2), dtype=np.float32)
    mask = np.zeros((512, 512), dtype=np.bool_)
    ktmp = np.zeros(num, dtype=np.float32)
    for k in range(1, num, 1):
        ktmp[k] = k*np.pi
    prepre = 0
    for i in range(len(pos)):
        xx = pos[i]//512
        yy = pos[i]%512
        mask[xx,yy] = 1
        for j in range(prepre, ind[i], 2):
            ceof[xx, yy, 0] += val[j+1]-val[j]
            for k in range(1, num, 1):
                ceof[xx, yy, k] += (np.sin(ktmp[k]*val[j+1])-np.sin(ktmp[k]*val[j])) / ktmp[k]
                ceof[xx, yy, k+num] += (np.cos(ktmp[k]*val[j])-np.cos(ktmp[k]*val[j+1])) / ktmp[k]
        prepre = ind[i]
    return ceof, mask
    