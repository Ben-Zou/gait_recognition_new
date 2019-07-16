import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import pdb
from torch.nn.parameter import Parameter
import numpy as np
from .utils import LogAct, RecLogAct


class DMCell(nn.Module):

    def __init__(self,
                 inp_num = 5,
                 hid_num = 2,
                 Je = 8.,
                 Jm = -2,
                 I0 = 0.0,
                 dt = 1.,
                 taus = 100.,
                 gamma = 0.1,
                 target_mode="x_target",
                 learning_rule = "force",
                 activation = LogAct(),
                 rec_activation = RecLogAct()):
        super().__init__()

        self.hid_num = hid_num
        self.inp_num = inp_num
        self.Je = Je
        self.Jm = Jm
        self.I0 = I0
        self.alpha = dt/taus
        self.gamma = gamma
        self.win = Parameter(torch.Tensor(hid_num,inp_num))
        self.wr = Parameter(torch.Tensor(hid_num,hid_num))
        self.act = activation
        self.rec_act = rec_activation
        self.learning_rule = learning_rule
        self.target_mode = target_mode

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hid_num)
        # stdv = 0.5
        if self.learning_rule == "force":
            self.win.data = torch.zeros((self.hid_num,self.inp_num))
        else:
            # self.win.data.uniform_(-stdv, stdv)
            self.win.data = torch.zeros((self.hid_num,self.inp_num))
        wr = np.ones((self.hid_num,self.hid_num))*self.Jm
        wr = wr+np.eye(self.hid_num)*self.Je - np.eye(self.hid_num)*self.Jm
        self.wr.data = torch.FloatTensor(wr)
        self.wr.requires_grad = False

    def apply_win(self,w):
        assert torch.Size(w.shape) == self.win.shape, "w shape should be same, but got {}.format"(w.shape)
        self.win.data = torch.FloatTensor(w)

    def forward(self,x,hid,y=None):
        """
        learning_rule is "force" or "bp"
        """
        if y is None:
            s = hid[0]
            # pdb.set_trace()
            rx = F.linear(x,self.win) + self.I0 + F.linear(s,self.wr)
            r = self.act(rx)
            s_new = s + self.alpha*(-s + (1.-s)*self.gamma*r)
            if self.target_mode == "x_target":
                return rx, (s_new,)
            else:
                return r, (s_new,)

        elif y is not None and self.learning_rule == "force":
            if self.target_mode == "x_target":
                y = y
            else:
                y = self.rec_act(y)
            batch_size = x.shape[0]
            s,P = hid
            rx = F.linear(x,self.win) + self.I0 + F.linear(s,self.wr)
            err = rx - y
            r = x

            k_fenmu = F.linear(r, P)
            rPr = torch.sum(k_fenmu * r, 1, True)

            k_fenzi = 1.0 /(1.0 + rPr)
            k = k_fenmu * k_fenzi

            kall = k[:,:,None].repeat(1, 1, self.hid_num)
            # kall = torch.repeat(k[:, :, None], (1, 1, self.hid_num))
            dw = -kall * err[:, None, :]
            self.win.copy_(self.win + torch.mean(dw, 0).transpose(1,0))

            # pdb.set_trace()
            P = P - F.linear(k.t(), k_fenmu.t())/batch_size
            #
            r = self.act(rx)
            s_new = s + self.alpha*(-s + (1.-s)*self.gamma*r)

            return err,r,(s_new, P)
        else:
            raise ValueError("No such inference or training configuration in the Decision Network !")


# class ForceCell(nn.Module):
#
#     def __init__(self,
#                  inp_num = 5,
#                  hid_num = 5,
#                  Je = 8.,
#                  Jm = -2,
#                  I0 = 0.0,
#                  dt = 0.1,
#                  taus = 1.,
#                  gamma = 0.1,):
#         super().__init__()
#         self.hid_num = hid_num
#         self.Je = Je
#         self.Jm = Jm
#         self.I0 = I0
#         self.alpha = dt/taus
#         self.gamma = gamma
#         self.win = Parameter(torch.Tensor(hid_num,inp_num))
#         self.wr = Parameter(torch.Tensor(hid_num,hid_num))
#         self.act = activation
#         self.wr = Parameter(torch.Tensor(hid_num,hid_num))
#         self.init_w()
#
#     def init_w(self):
#         wr = np.ones((self.hid_num,self.hid_num))*self.Jm
#         wr = wr*np.eye(self.Je/self.Jm)
#         self.wr.weight.data.constant_(torch.FloatTensor(wr))
#
#     def forward(self,x,y,hid):
#         batch_size = x.shape[0]
#         P, Wout = hid
#         xout = F.linear(x,Wout) + self.I0 + F.linear(s,self.wr)
#         err = xout - y
#         r = x
#
#         k_fenmu = F.linear(r, P)
#         rPr = torch.sum(k_fenmu * r, 1, True)
#
#         k_fenzi = 1.0 /(1.0 + rPr)
#         k = k_fenmu * k_fenzi
#
#         kall = torch.repeat(k[:, :, None], (1, 1, self.hid_num))
#         dw = -kall * err[:, None, :]
#
#         Wout = Wout + np.mean(dw, 0)
#         P = P - backend.matmul(k.T, k_fenmu)/batch_size
#
#         return err, (Wout, P)


if __name__ == "__main__":
    pass
