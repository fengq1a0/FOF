import os

import torch
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter

from lib.dataset_mesh_only import TrainSet as Dataset
from lib.fof import mesh_to_fof, render_fof
from lib.HRNet import get_FOF as Network
from lib.utils import toDevice


def make_grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # torch < 2.3
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train(args, model, epoch, DL, optimizer, scaler, writer):
    model.train()
    step = epoch * len(DL)
    for it, data in enumerate(tqdm.tqdm(DL)):
        data = toDevice(data, args["dev"])
        R = data["R"]
        smpl = torch.einsum("bntx,byx->bnty", data["smpl_v"], R)
        vv = torch.einsum("bntx,byx->bnty", data["v"], R)
        vn = torch.einsum("bntx,byx->bnty", data["vn"], R)

        # ground truth / inputs, rendered on the fly (triangle soups, zero padded).
        # args["fill"]: "hybrid" handles scan defects best; "robust" reproduces the original GT.
        fof_smpl = mesh_to_fof(smpl, None, 16, args["res"], fill=args["fill"])
        gt = render_fof(vv, None, vn, args["dim"], args["res"], fill=args["fill"])
        fof_mesh = gt.fof
        mask = fof_mesh[:, 0:1] != 0

        with torch.autocast("cuda", enabled=args["amp"]):
            fof = model(torch.cat((gt.normal_front, gt.normal_back, fof_smpl), dim=1))
        loss = F.mse_loss(torch.masked_select(fof.float(), mask), torch.masked_select(fof_mesh, mask)) * 1024

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        writer.add_scalar("MSE", loss.item(), it + step)


def main(args):
    torch.cuda.set_device(args["dev"])
    dataset = Dataset()
    DL = torch.utils.data.DataLoader(
        dataset, batch_size=args["bs"],
        num_workers=8, persistent_workers=True,
        pin_memory=True, shuffle=True,
        worker_init_fn=dataset.open_db)
    print("DL size:", len(DL))

    start_epoch = 0
    model = Network(i=6 + 16, c=[32, 64, 128, 256], dim=args["dim"]).to(args["dev"])
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scaler = make_grad_scaler(args["amp"])

    os.makedirs("ckpt/%s" % args["name"], exist_ok=True)
    ckpt_list = sorted(os.listdir(os.path.join("ckpt", args["name"])))
    if len(ckpt_list) > 0:
        ckpt_path = os.path.join("ckpt", args["name"], ckpt_list[-1])
        print("Resuming from ", ckpt_path)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        start_epoch = state_dict["epoch"]
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        if state_dict.get("scaler"):  # empty when saved with amp=False
            scaler.load_state_dict(state_dict["scaler"])
        del state_dict

    writer = SummaryWriter(log_dir="logs/" + args["name"])
    for epoch in range(start_epoch, 1000):
        train(args, model, epoch, DL, optimizer, scaler, writer)
        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        }, "ckpt/%s/%03d.pth" % (args["name"], epoch + 1))
        print("ckpt/%s/%03d.pth saved!" % (args["name"], epoch + 1))
    writer.close()


if __name__ == '__main__':
    # -------------cfg here-------------
    args = {
        "name": "FOF-X",
        "amp": True,
        "dev": 0,
        "bs": 8,
        "res": 512,
        "dim": 128,
        "fill": "hybrid",
    }
    # ----------------------------------
    torch.backends.cudnn.benchmark = True
    main(args)
