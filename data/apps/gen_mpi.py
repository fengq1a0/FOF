import numpy as np
from numba import jit
from numba.np.extensions import cross2d


@jit(nopython=True)
def get_mpi(v, f, res):
    mpi = []
    v[:,1] *= -1
    v[:,:2] = (v[:,:2]+1)*(res/2) - 0.5
    for fid in f:
        pts = v[fid]
        iMax = int(np.ceil(np.max(pts[:,0])))
        iMax = min(res, iMax)
        iMin = int(np.ceil(np.min(pts[:,0])))
        iMin = max(0, iMin)
        jMax = int(np.ceil(np.max(pts[:,1])))
        jMax = min(res, jMax)
        jMin = int(np.ceil(np.min(pts[:,1])))
        jMin = max(0, jMin)
        for i in range(iMin, iMax):
            for j in range(jMin, jMax):
                p = np.array([i,j])
                w2 = cross2d(pts[1,:2] - pts[0,:2], p - pts[0,:2])
                w0 = cross2d(pts[2,:2] - pts[1,:2], p - pts[1,:2])
                w1 = cross2d(pts[0,:2] - pts[2,:2], p - pts[2,:2])
                ss = w0+w1+w2
                if ss==0:
                    ss = 1
                w0 /= ss
                w1 /= ss
                w2 /= ss
                if w0>=0 and w1>=0 and w2>=0:
                    mpi.append((j*res+i, w0*pts[0,2]+w1*pts[1,2]+w2*pts[2,2]))
    tmp = sorted(mpi)
    pre = tmp[-1]
    mpi = []
    for i in tmp:
        if i!=pre:
            mpi.append(i)
        pre = i
    
    pos = []
    ind = []

    pre = 0
    flag = False
    while pre < len(mpi):
        nxt = pre+1
        pos.append(mpi[pre][0])
        while nxt < len(mpi) and mpi[nxt][0] == mpi[pre][0]:
            nxt += 1
        if (nxt-pre)%2 == 1:
            flag = True
        pre = nxt
        ind.append(pre)
        
    pos = np.array(pos, dtype=np.uint32)
    ind = np.array(ind, dtype=np.uint32)
    val = np.zeros(len(mpi), dtype=np.float32)
    for i in range(len(mpi)):
        val[i] = mpi[i][1]
    return pos, ind, val, flag

import os
import multiprocessing

def work(d, src, tar, res):
    R = []
    for i in range(256):
        y = np.pi*i/128
        sy = np.sin(y)
        cy = np.cos(y)
        r = np.array([  [ cy, 0.0,  sy],
                        [0.0, 1.0, 0.0],
                        [-sy, 0.0,  cy],] )
        R.append(r)

    a = os.path.join(src, d, "mesh.npz")
    a = np.load(a)
    os.makedirs(os.path.join(tar, d), exist_ok=True)
    
    for view in range(256):
        pos, ind, val, flag = get_mpi(np.matmul(a["v"], R[view].T), a["f"], res)
        if flag:
            print("ERROR on %s/%s/%03d" % (d, view))
            np.savez_compressed("%s/%s/%03d_ERROR.npz" %(tar, d, view), pos=pos, ind = ind, val=val)
        else:
            np.savez_compressed("%s/%s/%03d.npz" %(tar, d, view), pos=pos, ind = ind, val=val)
    print(d)

if __name__=="__main__":
    res = 512 # resolution
    src = "../raw"
    tar = "../mpi"
    dd = sorted(os.listdir(src))
    pool = multiprocessing.Pool(processes = 4)
    for d in dd:
        pool.apply_async(work, (d, src, tar, res))    
    pool.close()
    pool.join()
    