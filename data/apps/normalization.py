import os
import numpy as np
import ctypes as ct

so = ct.cdll.LoadLibrary("../lib/cpp/welzl.so")
def work(name):
    tmp = np.load(name)
    v = tmp["v"].astype(np.float64)
    v = np.ascontiguousarray(v)
    vv = v.copy()
    ans = np.ascontiguousarray(np.zeros(4, dtype = np.float64))
    pv = v.ctypes.data_as(ct.POINTER(ct.c_double))
    pans = ans.ctypes.data_as(ct.POINTER(ct.c_double))
    so.welzl(pv, v.shape[0],pans)
    vv = (vv-ans[:3])/ans[3]
    np.savez_compressed(name, v=vv.astype(np.float32), f = tmp["f"])
    print(name)

base = "../raw"
dd = sorted(os.listdir(base))
for d in dd:
    name = os.path.join(base, d, "mesh.npz")
    work(name)