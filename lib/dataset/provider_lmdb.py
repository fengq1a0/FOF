import cv2
import lmdb
import numpy as np
from io import BytesIO

class LmdbDataProvider():
    def __init__(self, path="/mnt/fq_ssd/fq/FOF/lmdb_512") -> None:
        self.db = None
        self.path = path
    
    def open_db(self):
        self.db = lmdb.open(self.path,
                            subdir=True, 
                            readonly=True, 
                            lock=False, 
                            readahead=False, 
                            meminit=False)

    def get_data_base(self, name, vid, lid):
        with self.db.begin(write=False) as txn:
            mpi = txn.get(("%s_%03d_mpi" % (name, vid)).encode())
            img = txn.get(("%s_%03d_%03d" % (name, vid, lid)).encode())
        img = cv2.imdecode(np.frombuffer(img,np.uint8),-1)
        mpi = np.load(BytesIO(mpi))
        return {"img":img, "mpi":mpi}
