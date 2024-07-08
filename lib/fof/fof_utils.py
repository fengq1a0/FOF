import torch
from . import fof

class FOF(torch.nn.Module):
    def __init__(self, B, H):
        super(FOF, self).__init__()
        self.H = H
        cnt = B*H*H
        self.pix_cnt = torch.nn.Parameter(
            torch.zeros(cnt, dtype=torch.int32),
            requires_grad=False)
        self.int_cnt = torch.nn.Parameter(
            torch.zeros(cnt, dtype=torch.int32),
            requires_grad=False)
        self.pix_pre = torch.nn.Parameter(
            torch.zeros(cnt, dtype=torch.int32),
            requires_grad=False)
        self.int_pre = torch.nn.Parameter(
            torch.zeros(cnt, dtype=torch.int32),
            requires_grad=False)
        self.pix = torch.nn.Parameter(
            torch.zeros(cnt, dtype=torch.int32),
            requires_grad=False)
        self.ind = torch.nn.Parameter(
            torch.zeros(cnt, dtype=torch.int32),
            requires_grad=False)
        self.pre_size = fof.get_buffer_size(cnt)
        self.pre_tmp = torch.nn.Parameter(
            torch.zeros(self.pre_size, dtype=torch.uint8),
            requires_grad=False)
        self.int_bbb = torch.nn.Parameter(
            torch.zeros(cnt*4), requires_grad=False)
        self.int_ddd = torch.nn.Parameter(
            torch.zeros(cnt*4, dtype=torch.int32),
            requires_grad=False)
        self.a = torch.nn.Parameter(
            torch.Tensor([     H/2,    -H/2, 1])[None,None,None,:],
            requires_grad=False)
        self.b = torch.nn.Parameter(
            torch.Tensor([ H/2-0.5, H/2-0.5, 0])[None,None,None,:],
            requires_grad=False)

    def forward(self, v_tensor, C):
        tmp = v_tensor*self.a + self.b
        return fof.static_render(tmp, C, self.H, self.pre_size,
                                 self.pix_cnt, self.int_cnt, 
                                 self.pix_pre, self.int_pre, 
                                 self.pix, self.ind, 
                                 self.pre_tmp, self.int_bbb, self.int_ddd)

def render(v, num, res):
    a = torch.Tensor([     res/2,    -res/2, 1])[None,None,None,:].to(v.device)
    b = torch.Tensor([ res/2-0.5, res/2-0.5, 0])[None,None,None,:].to(v.device)
    tmp = v*a + b
    return fof.dynamic_render(tmp, num, res)

class FOF_Normal(torch.nn.Module):
    def __init__(self, B, H):
        super(FOF_Normal, self).__init__()
        self.H = H
        cnt = B*H*H
        self.pix_cnt = torch.nn.Parameter(
            torch.zeros(cnt, dtype=torch.int32),
            requires_grad=False)
        self.int_cnt = torch.nn.Parameter(
            torch.zeros(cnt, dtype=torch.int32),
            requires_grad=False)
        self.pix_pre = torch.nn.Parameter(
            torch.zeros(cnt, dtype=torch.int32),
            requires_grad=False)
        self.int_pre = torch.nn.Parameter(
            torch.zeros(cnt, dtype=torch.int32),
            requires_grad=False)
        self.pix = torch.nn.Parameter(
            torch.zeros(cnt, dtype=torch.int32),
            requires_grad=False)
        self.ind = torch.nn.Parameter(
            torch.zeros(cnt, dtype=torch.int32),
            requires_grad=False)
        self.pre_size = fof.get_buffer_size(cnt)
        self.pre_tmp = torch.nn.Parameter(
            torch.zeros(self.pre_size, dtype=torch.uint8),
            requires_grad=False)
        self.int_bbb = torch.nn.Parameter(
            torch.zeros(cnt*4), requires_grad=False)
        self.int_ddd = torch.nn.Parameter(
            torch.zeros(cnt*4, dtype=torch.int32),
            requires_grad=False)
        self.a = torch.nn.Parameter(
            torch.Tensor([     H/2,    -H/2, 1])[None,None,None,:],
            requires_grad=False)
        self.b = torch.nn.Parameter(
            torch.Tensor([ H/2-0.5, H/2-0.5, 0])[None,None,None,:],
            requires_grad=False)

    def forward(self, v_tensor, vn_tensor, C):
        tmp = v_tensor*self.a + self.b
        res, dF, dB, nF, nB = fof.fof_normal_static_render(tmp, vn_tensor, 
                                 C, self.H, self.pre_size,
                                 self.pix_cnt, self.int_cnt, 
                                 self.pix_pre, self.int_pre, 
                                 self.pix, self.ind, 
                                 self.pre_tmp, self.int_bbb, self.int_ddd)
        ttt = nF.norm(dim=1)[:,None]
        ttt = torch.where(ttt==0, 1, ttt)
        nF/=ttt

        ttt = nB.norm(dim=1)[:,None]
        ttt = torch.where(ttt==0, 1, ttt)
        nB/=ttt
        return res, dF, dB, nF, nB