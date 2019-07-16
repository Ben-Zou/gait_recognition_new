from __future__ import print_function
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init

import pdb
cuda = torch.cuda.is_available()

class SfaRNN(nn.Module):

    def __init__(self,in_features,
                      hidden_features,
                      activation='relu',
                      m = 10.,
                      tauu = 10,
                      tauv = 150,
                      dt = 1,
                      bias = True):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.m = m
        self.alpha_u = dt/tauu
        self.alpha_v = dt/tauv

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError("The activation function {} is not defined!".format(activation))

        self.win = Parameter(torch.Tensor(hidden_features,in_features))
        self.wr = Parameter(torch.Tensor(hidden_features,hidden_features))
        if bias:
            self.bias = Parameter(torch.Tensor(hidden_features))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_features)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self,x,hid_state):
        """
        hid_state = (u,v)

        network dynamics:
        1. tauu*du/dt = -u + win*x + wr*r + b -v
        2. tauv*dv/dt = -v+ mu
        """
        u, v = hid_state
        v = (1.-self.alpha_v)*v + self.alpha_v*self.m*u
        v = F.relu(v)
        u = (1-self.alpha_u)*u + self.alpha_u*(
             F.linear(x,self.win) + F.linear(u,self.wr)
             +self.bias -v)
        u = self.act(u)
        return u, (u,v)








#
