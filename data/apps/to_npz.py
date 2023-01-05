import numpy as np

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
                ft.append([ int(ttt[0][1])-1, 
                            int(ttt[1][1])-1, 
                            int(ttt[2][1])-1])
    v = np.array(v, dtype=np.float32)
    f = np.array(f, dtype=np.uint32)
    vt = np.array(vt, dtype=np.float32)
    ft = np.array(ft, dtype=np.uint32)
    return v, f, vt, ft

if __name__ == "__main__":
    import os
    import cv2
    import shutil

    dd = sorted(os.listdir("../obj"))
    for d in dd:
        os.makedirs("../raw/%s"%d, exist_ok=True)
        for fi in os.listdir("../obj/%s"%d):
            fname = "../obj/%s/%s" % (d, fi)
            if fi.endswith(".obj"):
                v, f, vt, ft = load_obj(fname)
                np.savez_compressed("../raw/%s/mesh.npz"%d, v=v, f=f)
                np.savez_compressed("../raw/%s/uv.npz"%d, vt=vt, ft=ft)
            elif fi.endswith((".jpg", ".jpeg")) :
                shutil.copy(fname, "../raw/%s/texture.jpg"%d)
            elif fi.endswith(".png"):
                img = cv2.imread(fname)
                cv2.imwrite("../raw/%s/texture.jpg"%d, img)
        print(d)