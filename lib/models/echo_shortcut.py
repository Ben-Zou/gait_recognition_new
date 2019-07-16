from .base import BaseModel
import torch
import torch.nn as nn
import numpy as np

import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import pdb
from ..rnn import SimpleEcho
from ..rnn import DMCell


cuda = torch.cuda.is_available()

if cuda:
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

class SCEcho1(BaseModel):
	"""
	This Echo1 model can be trianed with decision making network or liner readout.
	dm_dict: when None, a liner readout is used.

	No ForceLearning is used!

	BPTT is used,

		   |
	|      |(identity mapping)
	|      |
	|----->
	|      |
	|      |(echo network)
		   |

	the output aggregate the identity mapping and echo network output.

	"""

	def __init__(self,n_inp,
				 n_echo,
				 n_dm,
				 echo_dict,
				 identity_scale=1.0,
				 dm_dict=None):

		super().__init__()
		self.n_inp = n_inp
		self.n_echo = n_echo
		self.n_dm = n_dm
		self.dm_dict = dm_dict
		self.identity_scale = identity_scale

		self.simple_echo = SimpleEcho(inp_num=n_inp,
									  hid_num=n_echo,
									  **echo_dict)
		if dm_dict is None:
			self.linear = nn.Linear(self.n_echo+n_inp,self.n_dm,bias=True)
			self.criterion = nn.CrossEntropyLoss()
		else:
			self.dm = DMCell(inp_num=n_echo+n_inp,
						 hid_num=n_dm,
						 **dm_dict)
			self.criterion = nn.MSELoss()

		self.init_weights()

	@property
	def with_dm(self):
		return self.dm_dict != None

	def init_weights(self):
		self.simple_echo.init_weights()
		if self.with_dm:
			self.dm.init_weights()
		else:
			stdv = 1.0 / math.sqrt(self.n_echo)
			self.linear.weight.data = torch.FloatTensor(
							 np.random.uniform(-stdv,stdv,size=(self.n_dm,self.n_echo+self.n_inp)))
			self.linear.bias.data = torch.zeros((self.n_dm))


	def init_states(self,batch_size):
		if self.with_dm:
			self.u_echo = torch.zeros((batch_size,self.n_echo)).to(device)
			self.h_echo = torch.zeros((batch_size,self.n_echo)).to(device)
			self.s_dm = torch.zeros(batch_size,self.n_dm).to(device)
		else:
			self.u_echo = torch.zeros((batch_size,self.n_echo)).to(device)
			self.h_echo = torch.zeros((batch_size,self.n_echo)).to(device)

	def forward_train(self,x):
		"""
		x, a tuple, (X,Y)
		   X, an input sequence. [batch,T,features]
		   Y, a target label, long tensor. [batch,] or [batch,T]
		"""
		assert len(x) == 2
		inp_x, inp_y = x
		inp_x = inp_x.to(device)
		inp_y = inp_y.to(device)
		batch, T, _ = inp_x.shape
		self.init_states(batch)

		echo_states = []
		dm_states = []
		loss = 0

		rr = torch.zeros((batch,self.n_dm)).to(device)
		hh = torch.zeros((batch,self.n_echo)).to(device)
		if self.with_dm:
			for i in range(T):
				self.h_echo, (self.u_echo,_) = self.simple_echo(inp_x[:,i],(self.u_echo,self.h_echo))
				dm_input = torch.cat([self.h_echo,self.identity_scale*inp_x[:,i].reshape(-1,self.n_inp)],dim=1)
				r,(self.s_dm,) = self.dm(dm_input,(self.s_dm,))
				loss += self.criterion(r, inp_y[:,i])

				rr.copy_(r)
				dm_states.append(rr.cpu().detach().numpy())

				hh.copy_(self.h_echo)
				echo_states.append(hh.cpu().detach().numpy())
		else:
			inp_y = inp_y.reshape(-1)
			rx_mean = 0
			for i in np.arange(T):
				self.h_echo, (self.u_echo,_) =  self.simple_echo(inp_x[:,i], (self.u_echo,self.h_echo))
				dm_input = torch.cat([self.h_echo,self.identity_scale*inp_x[:,i].reshape(-1,self.n_inp)],dim=1)
				r = self.linear(dm_input)
				rx_mean = rx_mean + r/T

				rr.copy_(r)
				dm_states.append(rr.cpu().detach().numpy())

				hh.copy_(self.h_echo)
				echo_states.append(hh.cpu().detach().numpy())

			loss += self.criterion(rx_mean, inp_y)
		print("loss is ",loss.cpu().item())
		outputs = dict(
					 loss = loss,
					 echo_states = echo_states,
					 outputs = dm_states
					  )
		return outputs


	def forward_test(self,x):
		"""
		x, a tuple, (X,Y)
		   X, an input sequence. [batch,T,features]
		   Y, a target label, long tensor. [batch,] or [batch,T]
		"""
		assert len(x) == 2
		inp_x, inp_y = x
		inp_x = inp_x.to(device)
		batch, T, _ = inp_x.shape
		self.init_states(batch)

		echo_states = []
		dm_states = []

		hh = torch.zeros((batch,self.n_echo)).to(device)
		rr = torch.zeros((batch,self.n_dm)).to(device)
		if self.with_dm:
			for i in range(T):
				self.h_echo, (self.u_echo,_) = self.simple_echo(inp_x[:,i],(self.u_echo,self.h_echo))
				dm_input = torch.cat([self.h_echo,self.identity_scale*inp_x[:,i].reshape(-1,self.n_inp)],dim=1)
				r,(self.s_dm,) = self.dm(dm_input,(self.s_dm,))

				rr.copy_(r)
				dm_states.append(rr.cpu().detach().numpy())

				hh.copy_(self.h_echo)
				echo_states.append(hh.cpu().detach().numpy())
			dm_states = np.array(dm_states).reshape(T,batch,-1)
			inp_y = inp_y[:,-1].cpu().numpy().argmax(axis=-1)
		else:
			for i in np.arange(T):
				self.h_echo, (self.u_echo,_) =  self.simple_echo(inp_x[:,i], (self.u_echo,self.h_echo))
				dm_input = torch.cat([self.h_echo,self.identity_scale*inp_x[:,i].reshape(-1,self.n_inp)],dim=1)
				r = self.linear(dm_input)

				rr.copy_(r)
				dm_states.append(rr.cpu().detach().numpy())

				hh.copy_(self.h_echo)
				echo_states.append(hh.cpu().detach().numpy())

			dm_states = np.array(dm_states).reshape(T,batch,-1)
			inp_y = inp_y.view(-1).cpu().numpy()

		outputs = dict(
					 echo_states = echo_states,
					 outputs = dm_states,
					 labels = inp_y
					  )
		return outputs



##
