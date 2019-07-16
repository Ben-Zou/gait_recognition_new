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

class StpRNN(nn.Module):
    """
    sigma: is the stength of the noise.

    rnn dynamics:
        tau_r*dr/dt = -r + f(win*xin + wr*r + bias + noise)
        noise is ~ sqrt(2*tau_r)*sigma*N(0,1.);
    stp dynamics:
        dx/dt = (1-x)/taud - u*x*r
        du/dt = (U-u)/tauf + U*(1-u)*r
        taud is recovery time, depression variable,
        tauf is calcium concentration time constant, facilitation variable
    for each post synapse:
        Ir(t) = W*u*x*r(t)

    iteration is Euler iteration.

   par,
     activation, activation of recurrent neural network.
     dt, simulation time step of network dynamics.
     tau_r, time constance of firing rate.
     tau_u, time constance of u variables, calcium
     tau_x, time constance of v variables, vechile number.
     U, the saturation calcium.
     rnn_sigma, the sigma of recurrent neural network.
    """

    def __init__(self,
                 in_features,
                 hidden_features,
                 use_stp = True,
                 activation='relu',
                 tauu = 100,
                 tauf = 1500,
                 taud = 100,
                 U = 0.15,
                 dt = 10,
                 bias=True
                 ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.use_stp = use_stp
        self.alpha_u = dt/tauu
        self.alpha_d = dt/taud
        self.alpha_f = dt/tauf
        self.dt_sec = dt/1000.
        self.U = U

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError("The activation function {} is not defined!".format(activation))


        self.win = Parameter(torch.Tensor(in_features,hidden_features))
        self.wr = Parameter(torch.Tensor(hidden_features,hidden_features))
        ## add gaussian white noise.
        # self.noise_gen = torch.distributions.Normal(0.,1.)
        if bias:
            self.bias = Parameter(torch.Tensor(hidden_features))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        reset the parameters.
        """
        stdv = 1.0 / math.sqrt(self.hidden_features)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self,inp, hid_state):
        r, u, x = hid_state
        if self.use_stp:
            x = (1-self.alpha_d)*x + self.alpha_d - self.dt_sec*x*u*r
            u = (1-self.alpha_f)*u + self.alpha_f*self.U + self.dt_sec*self.U*(1-u)*r
            x = torch.clamp(x,0.,1.)
            u = torch.clamp(u,0.,1.)
            r_post = r*x*u
        else:
            r_post = r

        r = (1-self.alpha_u)*r + \
            self.alpha_u*((torch.matmul(inp,self.win)) + torch.matmul(r_post,self.wr) + self.bias)

        r = self.act(r)

        return r, (r,u,x)











#
