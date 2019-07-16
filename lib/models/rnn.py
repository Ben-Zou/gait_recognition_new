from .base import BaseModel
import torch
import torch.nn as nn
import numpy as np

import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import pdb
# from ..rnn import SimpleEcho
# from ..rnn import DMCell

from ..rnn import LSTM as LSTMCell
from torch.nn import RNNCell
from torch.nn import GRUCell


cuda = torch.cuda.is_available()

if cuda:
	device = torch.device('cuda')
else:
	device = torch.device('cpu')


class PlainRNN(BaseModel):
	"""
	single_loss: single_loss= True, means the loss is
				  only caculated in the last time step.
	"""

	def __init__(self, inp_size,
					   hid_size,
					   out_size,
					   rnn_type="raw_rnn",
					   single_loss=True):
		super().__init__()
		allow_rnn_types = ["raw_rnn","lstm","gru"]
		assert rnn_type in allow_rnn_types
		self.rnn_type = rnn_type
		self.single_loss = single_loss
		self.inp_size = inp_size
		self.hid_size = hid_size
		self.out_size = out_size

		if rnn_type == "raw_rnn":
			self.lstm = RNNCell(inp_size, hid_size)
		if rnn_type == "lstm":
			self.lstm = LSTMCell(inp_size, hid_size)
		if rnn_type == "gru":
			self.lstm = GRUCell(inp_size, hid_size)
		self.fc1 = nn.Linear(hid_size, out_size)
		self.criterion = nn.CrossEntropyLoss()

	def init_weights(self):
		self.lstm.reset_parameters()
		self.fc1.reset_parameters()

	def init_states(self,batch_size):
		if self.rnn_type == "lstm":
			self.h = torch.zeros(batch_size, self.hid_size).to(device)
			self.c = torch.zeros(batch_size, self.hid_size).to(device)

		if self.rnn_type == "raw_rnn" or self.rnn_type == "gru":
			self.h = torch.zeros(batch_size, self.hid_size).to(device)


	def forward_train(self, x):
		assert len(x) == 2
		inp_x, inp_y = x
		inp_x = inp_x.to(device)
		inp_y = inp_y.to(device)
		batch, T, _ = inp_x.shape
		self.init_states(batch)

		dm_states = []
		loss = 0

		rr = torch.zeros((batch,self.out_size)).to(device)
		if self.rnn_type == "lstm":
			for i in range(T):
				y, (self.h,self.c) = self.lstm(inp_x[:,i],(self.h,self.c))
				output = self.fc1(y)
				rr.copy_(output)
				dm_states.append(rr.cpu().detach().numpy())

				if not self.single_loss:
					loss += self.criterion(output, inp_y.reshape(-1))
			if self.single_loss:
				loss += self.criterion(output, inp_y.reshape(-1))

		if self.rnn_type == "raw_rnn" or self.rnn_type == "gru":
			for i in range(T):
				self.h = self.lstm(inp_x[:,i],self.h)
				output = self.fc1(self.h)

				rr.copy_(output)
				dm_states.append(rr.cpu().detach().numpy())

				if not self.single_loss:
					loss += self.criterion(output, inp_y.reshape(-1))
			if self.single_loss:
				loss += self.criterion(output, inp_y.reshape(-1))

		print("loss is ",loss.cpu().item())
		outputs = dict(
					 loss = loss,
					 outputs = dm_states
					  )
		return outputs


	def forward_test(self,x):
		# x, new_state = self.lstm(x, state)
		# x = self.fc1(x
		assert len(x) == 2
		inp_x, inp_y = x
		inp_x = inp_x.to(device)
		batch, T, _ = inp_x.shape
		self.init_states(batch)

		dm_states = []

		rr = torch.zeros((batch,self.out_size)).to(device)
		if self.rnn_type == "lstm":
			for i in range(T):
				y, (self.h,self.c) = self.lstm(inp_x[:,i],(self.h,self.c))
				output = self.fc1(y)

				rr.copy_(output)
				dm_states.append(rr.cpu().detach().numpy())

		if self.rnn_type == "raw_rnn" or self.rnn_type == "gru":
			for i in range(T):
				self.h = self.lstm(inp_x[:,i],self.h)
				output = self.fc1(self.h)

				rr.copy_(output)
				dm_states.append(rr.cpu().detach().numpy())

		if self.single_loss:
			dm_states = (np.array(dm_states)[-1]).reshape(1,batch,-1)
		else:
			# dm_states = (np.array(dm_states)).reshape(T,batch,-1)
			dm_states = np.array(dm_states).reshape(T*batch,-1)
			dm_states_ = np.zeros_like(dm_states)
			index = np.argmax(dm_states,axis=1)
			dm_states_[range(T*batch),index] = 1
			dm_states = dm_states_.reshape(T,batch,-1).mean(axis=0).reshape(1,batch,-1)


		inp_y = inp_y.view(-1).cpu().numpy()

		outputs = dict(
					 outputs = dm_states,
					 labels = inp_y
					  )
		return outputs
