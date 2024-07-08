import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

import tqdm
from lib.utils import toDevice
from lib.HRNet import get_FOF as Network
from lib.dataset_mesh_only import TrainSet as Dataset
from lib.fof import FOF_Normal

def train(args, model, epoch, DL, optimizer, scaler, fof_render):
    model.train()
    writer = SummaryWriter(log_dir="logs/"+args["name"])
    cnt = epoch*len(DL)

    for it,data in enumerate(tqdm.tqdm(DL)):
        data = toDevice(data, 0)
        smpl = torch.einsum("bntx,byx->bnty", data["smpl_v"], data["R"])
        vv = torch.einsum("bntx,byx->bnty", data["v"], data["R"])
        vn = torch.einsum("bntx,byx->bnty", data["vn"], data["R"])
        fof_smpl, dF, dB, nF, nB = fof_render(smpl, smpl, 16)
        fof_mesh, dF, dB, nF, nB = fof_render(vv, vn, 128)
        if args["amp"]:
            with torch.autocast("cuda"):
                mask = fof_mesh[:,0:1,:,:] != 0 
                fof = model(torch.cat((nF, nB, fof_smpl), dim=1))
                loss_mse = F.mse_loss(torch.masked_select(fof,mask),
                                  torch.masked_select(fof_mesh,mask))*1024
                loss = loss_mse
                
            lbt = {"MSE":loss_mse.item()}
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        for k,v in lbt.items():
            writer.add_scalar(k,v,it+cnt)
    writer.close()
       
def main(args):
    # dataset
    dataset = Dataset()
    DL = torch.utils.data.DataLoader(
        dataset, batch_size=args["bs"], 
        num_workers=8, persistent_workers=True,
        pin_memory=True, shuffle=True,
        worker_init_fn = dataset.open_db)
    print("DL size:", len(DL))

    # model & others
    start_epoch = 0
    model = Network(i=6+16, c=[32,64,128,256]).to(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scaler = torch.cuda.amp.GradScaler()

    # load
    os.makedirs("ckpt/%s"%args["name"], exist_ok=True)
    ckpt_list = sorted(os.listdir(os.path.join("ckpt", args["name"])))
    if len(ckpt_list) > 0:
        ckpt_path = os.path.join("ckpt", args["name"], ckpt_list[-1])
        print('Resuming from ', ckpt_path)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        start_epoch = state_dict["epoch"]
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        scaler.load_state_dict(state_dict["scaler"])
        del state_dict

    fof_render = FOF_Normal(args["bs"],512).to(0)
    for epoch in range(start_epoch,1000,1):
        # train one epoch
        train(args, model, epoch, DL, optimizer, scaler, fof_render)
        # save model
        torch.save({
            "epoch" : epoch+1,
            "model" : model.state_dict(),
            "optimizer" : optimizer.state_dict(),
            "scaler" : scaler.state_dict()
        },"ckpt/%s/%03d.pth"%(args["name"],epoch+1))
        print("ckpt/%s/%03d.pth saved!"%(args["name"],epoch+1))


if __name__ == '__main__':
    #-------------cfg here-------------
    args = {
        "name"  : "FOF-X",
        "amp"   : True,
        "dev"   : 0,
        "bs"    : 8
    }
    #----------------------------------
    torch.backends.cudnn.benchmark = True
    main(args)