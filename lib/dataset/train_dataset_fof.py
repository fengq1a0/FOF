import torch
import torch.nn as nn
import torch.nn.functional as F
from .ceof_generator import tri_occ
from .provider_lmdb import LmdbDataProvider

class FOFDataset(torch.utils.data.Dataset):
    def __init__(self, name_list = "/mnt/fq_ssd/fq/FOF/namelist/train.txt") -> None:
        super().__init__()
        self.data_provider = LmdbDataProvider()
        self.train_mode = True
        with open(name_list, "r") as f:
            self.name_list = f.read().split()
    
    def open_db(self):
        self.data_provider.open_db()
    
    def __len__(self):
        if self.train_mode:
            return len(self.name_list)*256
        else:
            return len(self.name_list)

    def __getitem__(self, index):
        if self.train_mode:
            pid = index//256
            vid = index%256
            lid = int(torch.randint(4, (1,)))
        else:
            pid = index
            vid = int(torch.randint(256, (1,)))
            lid = int(torch.randint(4, (1,)))
        name = self.name_list[pid]

        data = self.data_provider.get_data_base(name, vid, lid)
        img = torch.from_numpy(data["img"].transpose((2,0,1)))
        ceof, mask = tri_occ(data["mpi"]["pos"], data["mpi"]["ind"], data["mpi"]["val"],16)
        ceof = torch.from_numpy(ceof.transpose((2,0,1)))
        mask = torch.from_numpy(mask[None])

        return {
            "name" : "%s_%03d_%03d"%(name, vid, lid),
            "img" : img,
            "ceof" : ceof,
            "mask" : mask
        }

def random_init(id):
    torch.utils.data.get_worker_info().dataset.open_db()

def fixed_init(id):
    torch.manual_seed(1000000000 + 7 + id)
    torch.utils.data.get_worker_info().dataset.open_db()

def get_dataloader(fixed=False, batch_size = 8):
    return torch.utils.data.DataLoader( FOFDataset(), batch_size=batch_size,
                                        num_workers=12, 
                                        shuffle=not fixed, worker_init_fn=fixed_init if fixed else random_init,
                                        pin_memory=True, persistent_workers=True)
