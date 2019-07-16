import torch
import torch.nn as nn
import pdb
import numpy as np

import torchvision as T
import torchvision.transforms.functional as F
import torch.utils.data as data

from .data_utils import target_win, target_loss
from PIL import Image

class gait_frame_1(data.Dataset):
	"""
	Return:
	img,img,0,0,0,...

	label is given in the final time step.
	"""
	def __init__(self,
				 data,
				 T_stim=1,
				 T_wait=5,
				 zero_mean = True,
				 unit_variance = True,
				 ):
		# [num_person,num_trails,50,28*28]
		nc,nt,T,wh= data.shape
		self.data = data.reshape(-1,wh)
		self.label = np.arange(nc).repeat(nt*T)
		self.T_stim = T_stim
		self.T_tot = T_stim + T_wait
		self.wh = wh

		if zero_mean:
			self.mean = np.mean(self.data)
		else:
			self.mean = 0.

		if unit_variance:
			self.var = np.std(self.data)
		else:
			self.var = 255.

	def __getitem__(self,index):
		x = self.data[index]
		x = (x-self.mean)/self.var
		xx = np.zeros((self.T_tot,self.wh))
		xx[:self.T_stim] = x[None,:]
		yy = self.label[index]
		return torch.FloatTensor(xx), torch.FloatTensor([yy]).long()

	def __len__(self,):
		return len(self.data)

class gait_frame_dm_xtarget(data.Dataset):
	# gait_28x28.npy
	# 8x50x50x28x28
	def __init__(self,datax,
				 Tstim = 50,
				 T = 50,
				 input_strength=1.,
				 noise_scale = 0.,
				 T_extra = 0,
				 normalize=True):
		self.data = datax # [8,n,50,28,28]
		c,n,_,w,h = datax.shape
		n_Tstim_repeat = int(Tstim/T)
		# self.data = self.data.reshape(c,n,10,5,w*h).transpose(0,1,3,2,4).reshape(c*n*5,10,w*h)
		self.data = self.data.reshape(c,n,T,w*h).reshape(c*n,T,w*h).repeat(n_Tstim_repeat,axis=1)

		self.mean = np.mean(self.data)
		self.std = np.std(self.data)
		self.normalize = normalize
		self.T_extra = T_extra
		self.noise_scale = noise_scale

		self.label = np.arange(c).repeat(n)
		self.c = c
		self.input_strength = input_strength
		self.Tstim =  Tstim + T_extra
		xx = np.tanh(np.linspace(-2.,2.,self.Tstim))
		self.label_win_i = target_win(xx,self.c)
		self.label_loss_i = target_loss(xx,self.c)

	def __getitem__(self, index):
		x = self.data[index]
		if self.normalize:
			x = (x-self.mean)/self.std*self.input_strength
		else:
			x = x/255.*self.input_strength

		if self.noise_scale !=0:
			x = x + np.random.normal(0.,self.noise_scale,size=x.shape)
		if self.T_extra !=0:
			x = np.concatenate([x,np.zeros((self.T_extra,x.shape[1]))],axis=0)
		y_index = int(self.label[index])
		# y = np.zeros((self.Tstim,self.c)) + 0.01
		y = self.label_loss_i.repeat(self.c).reshape(self.Tstim,self.c)
		y[:,y_index] = self.label_win_i
		return torch.FloatTensor(x),torch.FloatTensor(y)

	def __len__(self):
		return len(self.data)


class gait_frame_2(data.Dataset):
	"""
	x     :    x_t1,x_t2,x_t2,x_t2,...
	label :     y
	"""
	def __init__(self,datax,
				 T = 50,normalize=True):
		self.data = datax # [8,n,50,28,28]
		c,n,_,w,h = datax.shape
		self.data = self.data.reshape(c,n,T,w*h).reshape(c*n,T,w*h)
		self.label = np.arange(c).repeat(n)
		self.c = c
		mean = np.mean(self.data)
		std = np.std(self.data)
		if normalize:
			self.data = (self.data-mean)/std
		else:
			self.data = self.data/255.

	def __getitem__(self, index):
		x = self.data[index]
		y = self.label[index]
		return torch.FloatTensor(x),torch.FloatTensor([y]).long()

	def __len__(self):
		return len(self.data)

class gait_frame_2_noise(data.Dataset):
	"""
	x     :    x_t1,x_t2,x_t2,x_t2,...
	label :     y
	"""
	def __init__(self,datax,
				 T = 50,normalize=True,noise_scale=0.):
		self.data = datax # [8,n,50,28,28]
		c,n,_,w,h = datax.shape
		self.data = self.data.reshape(c,n,T,w*h).reshape(c*n,T,w*h)
		self.label = np.arange(c).repeat(n)
		self.c = c
		mean = np.mean(self.data)
		std = np.std(self.data)
		if normalize:
			self.data = (self.data-mean)/std
		else:
			self.data = self.data/255.

		if noise_scale !=0:
			self.data = self.data + np.random.normal(scale=noise_scale,size=self.data.shape)

	def __getitem__(self, index):
		x = self.data[index]
		y = self.label[index]
		return torch.FloatTensor(x),torch.FloatTensor([y]).long()

	def __len__(self):
		return len(self.data)

if __name__ == "__main__":
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	# test gait_frame_1;
	# ()
	test_gait = np.load("../data/gait_28x28_17.npy")
	print("test_gait shape ", test_gait.shape)
	data_x = test_gait[:2].reshape(2,50,50,-1)
	dataset = gait_frame_1(data_x,T_stim=2,T_wait=4)

	xx, yy = dataset[0]

	pdb.set_trace()
















##
