import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time

from lib.dataset.train_dataset_fof import get_dataloader
from lib.network.HRNet import HRNetV2_W32 as network
from lib.utils.utils import toDevice


def train(args):
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:%d"%args.gpu_id)
    print("using device:", device)

    DL = get_dataloader(batch_size=8)
    print("training dataset size:", len(DL))

    start_epoch = 0
    net = network().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0002)

    # continue to train
    os.makedirs("ckpt/%s"%args.name, exist_ok=True)
    if args.continue_train:
        ckpt_list = sorted(os.listdir(os.path.join("ckpt", args.name)))
        if len(ckpt_list) > 0:
            ckpt_path = os.path.join("ckpt", args.name, ckpt_list[-1])
            print('Resuming from ', ckpt_path)
            state_dict = torch.load(ckpt_path)
            start_epoch = state_dict["epoch"]
            net.load_state_dict(state_dict["net"])
            optimizer.load_state_dict(state_dict["optimizer"])
            del state_dict

    # train
    cnt = 0
    loss_board = None
    for epoch in range(start_epoch, args.num_epoch, 1):
        num_iteration = 0
        net.train()

        for data in DL:
            data = toDevice(data, device)
            loss, lbt = net.get_loss(data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss_board==None:
                loss_board = lbt
            else:
                for k,v in lbt.items():
                    loss_board[k] += v
            
            num_iteration += 1
            cnt += 1
            if cnt == args.frequency:
                info = "["+time.strftime("%m.%d-%H:%M:%S", time.localtime())+"]\t"
                info += "Epoch %03d, iteration %08d, " % (epoch+1, num_iteration)
                for k,v in loss_board.items():
                    info += "\t%s:%.6f" % (k,v/cnt)
                print(info)
                cnt = 0
                loss_board = None
        torch.save({
            "epoch" : epoch+1,
            "net" : net.state_dict(),
            "optimizer" : optimizer.state_dict()
        },"ckpt/%s/%03d.pth"%(args.name,epoch+1))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",      type=str, default="base", help="name of the experiment")
    parser.add_argument("--gpu_id",    type=int, default=0,      help="gpu id for cuda")
    parser.add_argument("--num_epoch", type=int, default=500,    help="num epoch to train")
    parser.add_argument("--frequency", type=int, default=4000,   help="frequency to output result")
    parser.add_argument("--continue_train", type=bool, default=True, help="if continue train")
    args = parser.parse_args()
    train(args)
    