import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torch.nn.parameter import Parameter
import numpy as np
import pdb

class SimpleEcho(nn.Module):

	def __init__(self,
				 inp_num=1,
				 hid_num=10,
				 tau = 10,
				 dt = 1,
				 scale = 1.3,
				 spars_p = 0.1,
				 spars_echo = 0.1,
				 scale_echo = 1.0,
				 init_mode = "mode_a",
				 activation = F.tanh):

		super().__init__()
		self.inp_num = inp_num
		self.hid_num = hid_num
		self.alpha = dt/tau
		self.scale = scale
		self.act = activation
		self.spars_p = spars_p
		self.spars_echo = spars_echo
		self.scale_echo = scale_echo
		self.init_mode = "mode_a"

		self.win = Parameter(torch.Tensor(hid_num,inp_num))
		self.wr = Parameter(torch.Tensor(hid_num,hid_num))

		self.init_weights()

	def init_weights(self):

		if self.init_mode == "mode_a":
			wr = np.random.rand(self.hid_num,self.hid_num) - 0.5
			wr[np.random.rand(*wr.shape)>self.spars_p]=0.
			M = np.eye(self.hid_num)*(1.-self.alpha) + wr*self.alpha
			radius = np.max(np.abs(np.linalg.eigvals(wr)))
			wr = wr/(radius-1.+self.alpha)*(self.scale-1.+self.alpha)
			self.wr.data = torch.FloatTensor(wr)
			self.wr.requires_grad = False

			# win = np.random.normal(scale=self.scale_echo,size=(self.hid_num,self.inp_num))
			win = np.random.uniform(low=-self.scale_echo,high=self.scale_echo,size=(self.hid_num,self.inp_num))
			win[np.random.rand(*win.shape)>self.spars_echo]=0.
			# stdv = 1.0 / math.sqrt(self.hid_num)
			self.win.data = torch.FloatTensor(win)
			self.win.requires_grad = False

		if self.init_mode == "mode_b":
			wr = np.random.rand(self.hid_num,self.hid_num) - 0.5
			wr[np.random.rand(*wr.shape)>self.spars_p]=0.
			M = np.eye(self.hid_num)*(1.-self.alpha) + wr*self.alpha
			radius = np.max(np.abs(np.linalg.eigvals(wr)))
			wr = wr/(radius-1.+self.alpha)*(self.scale-1.+self.alpha)
			self.wr.data = torch.FloatTensor(wr)
			self.wr.requires_grad = False

			# win = np.random.normal(scale=self.scale_echo,size=(self.hid_num,self.inp_num))
			win = np.ones((self.hid_num,self.inp_num))
			win[np.random.rand(*win.shape)>self.spars_echo]=0.
			# stdv = 1.0 / math.sqrt(self.hid_num)
			self.win.data = torch.FloatTensor(win)
			self.win.requires_grad = False

		if self.init_mode == "mode_c":
			wr = np.random.rand(self.hid_num,self.hid_num) - 0.5
			wr[np.random.rand(*wr.shape)>self.spars_p]=0.
			M = np.eye(self.hid_num)*(1.-self.alpha) + wr*self.alpha
			radius = np.max(np.abs(np.linalg.eigvals(wr)))
			wr = wr/(radius-1.+self.alpha)*(self.scale-1.+self.alpha)
			self.wr.data = torch.FloatTensor(wr)
			self.wr.requires_grad = False

			# win = np.random.normal(scale=self.scale_echo,size=(self.hid_num,self.inp_num))
			win = np.diag(np.ones((self.hid_num)))
			self.win.data = torch.FloatTensor(win)
			self.win.requires_grad = False





	def forward(self,x,hid):
		u, h = hid
		u_new = u + self.alpha*(-u + F.linear(x,self.win) + F.linear(h,self.wr))
		h_new = self.act(u_new)

		return h_new, (u_new,h_new)

if __name__ == "__main__":
	import pdb
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	torch.manual_seed(200)
	np.random.seed(200)
	torch.cuda.manual_seed(200)


	x = np.arange(0,1000,0.2)
	x1 = np.sin(x)
	x2 = np.sin(x+0.01)

	plt.figure()
	plt.plot(x1,label="x1")
	plt.plot(x2,label="x2")
	plt.legend()
	plt.xlabel("Time")
	plt.ylabel("Amplitude")
	plt.savefig("stim.png")

	hid_num = 100
	net = SimpleEcho(hid_num=hid_num)

	net.cuda()

	r1 = []
	r2 = []
	u = torch.zeros((1,hid_num)).cuda()
	h = torch.zeros((1,hid_num)).cuda()
	x1 = x1.reshape(-1,1,1)
	x2 = x2.reshape(-1,1,1)
	x1 = torch.FloatTensor(x1).cuda()
	x2 = torch.FloatTensor(x2).cuda()
	with torch.no_grad():
		hid1 = hid2 = (u,h)
		for x11,x22 in zip(x1,x2):
			h1,hid1 = net(x11,hid1)
			h2,hid2 = net(x22,hid2)
			r1.append(h1.cpu().numpy())
			r2.append(h2.cpu().numpy())

	r1 = np.array(r1).squeeze()
	r2 = np.array(r2).squeeze()

	plt.figure()
	plt.plot(r1[:,0],label="x1_neuron")
	plt.plot(r2[:,0],label="x2_neuron")
	plt.legend()
	plt.xlabel("Time")
	plt.ylabel("Neuron Activity")
	plt.savefig("neuron.png")































#
