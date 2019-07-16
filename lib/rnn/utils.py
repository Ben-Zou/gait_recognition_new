from __future__ import print_function
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init

class LogAct(nn.Module):

    def __init__(self,alpha=1.5,beta=4.,gamma=0.1,thr=6.):
        super().__init__()
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.thr = thr

    def forward(self,x):
        x = (x-self.thr)/self.alpha
        x = torch.exp(x)
        return self.beta/self.gamma*torch.log1p(x)


class RecLogAct(nn.Module):
    def __init__(self,alpha=1.5,beta=4.,gamma=0.1,thr=6.):
        super().__init__()
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.thr = thr

    def forward(self,x):

        x = torch.expm1(self.gamma/self.beta*x)
        return self.thr + self.alpha*torch.log(x)




if __name__ == "__main__":
    x1 = torch.arange(-10,10,0.1)
    x = (torch.tanh(x1)+1.)*60 + 0.1
    y = nn.Sequential(LogAct(),RecLogAct())
    z = y(x)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(x1.cpu().numpy(),z.cpu().numpy())
    plt.plot(x1.cpu().numpy(),x.cpu().numpy())
    plt.savefig("logact.png")



#
