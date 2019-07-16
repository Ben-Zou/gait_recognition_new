from torch.nn.utils import clip_grad

from .hook import Hook

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import pdb


class Echo1Hook(Hook):

	def __init__(self, plot_keys=[],save_dir="./"):
		self.plot_keys = plot_keys
		self.save_dir = save_dir

	def after_iter(self,runner):
		if "plot_dm_activities" in self.plot_keys:
			if runner.inner_iter %10==0:
				dm_outputs = runner.outputs["outputs"][:,0,:]
				label_index = runner.outputs["labels"]
				index = runner.inner_iter
				plot_dm_activities(dm_outputs,label_index,index,self.save_dir)


	def after_val_epoch(self,runner):
		if "plot_echo1_dm_weights" in self.plot_keys:
			echo_win = runner.model.simple_echo.win.data.cpu().numpy().reshape(-1)
			echo_wr = runner.model.simple_echo.wr.data.cpu().numpy().reshape(-1)
			dm_win = runner.model.dm.win.data.cpu().numpy().reshape(-1)
			plot_echo1_dm_weights(echo_win,echo_wr,dm_win,self.save_dir)
		if "plot_echo1_dm_activities" in self.plot_keys:
			outputs = np.array(runner.outputs["echo_states"])
			plot_echo1_dm_activities(outputs,self.save_dir)


class Echo2Hook(Hook):
	def __init__(self,plot_keys=[]):
		pass
	def after_run(self,runner):
		pass

def plot_echo1_dm_weights(echo_win,echo_wr,dm_win,save_dir):

	plt.figure(figsize=(15,3))
	plt.subplot(1,3,1)
	plt.hist(echo_win,50,label="echo_win")
	plt.title("mean_{:0.3f},std_{:0.3f},max_{:0.3f},min_{:0.3f}".format(
			   np.mean(echo_win),np.std(echo_win),np.max(echo_win),np.min(echo_win)))

	plt.subplot(1,3,2)
	plt.hist(echo_wr,50,label="echo_win")
	plt.title("mean_{:0.3f},std_{:0.3f},max_{:0.3f},min_{:0.3f}".format(
			   np.mean(echo_wr),np.std(echo_wr),np.max(echo_wr),np.min(echo_wr)))

	plt.subplot(1,3,3)
	plt.hist(dm_win,50,label="echo_win")
	plt.title("mean_{:0.3f},std_{:0.3f},max_{:0.3f},min_{:0.3f}".format(
			   np.mean(dm_win),np.std(dm_win),np.max(dm_win),np.min(dm_win)))

	plt.xlabel("Weight Value")
	plt.ylabel("Number")
	plt.tight_layout()
	plt.savefig(os.path.join(save_dir,"weights_distribution.png"))

def plot_echo1_dm_activities(outputs,save_dir):
	# outputs, [T,B,N]
	mean = np.mean(outputs,axis=2).reshape(-1)
	std = np.std(outputs,axis=2).reshape(-1)

	plt.figure(figsize=(10,3))
	plt.subplot(1,2,1)
	plt.plot(outputs[:,0,:10])
	plt.xlabel("Time Step")
	plt.ylabel("Neuron Activities")
	plt.subplot(1,2,2)
	plt.errorbar(range(len(mean)),mean,yerr=std)
	plt.savefig(os.path.join(save_dir,"activities.png"))

def plot_echo2_dm_activities():
	pass

def plot_dm_activities(dm_outputs,label_index,index,save_dir):
	plt.figure()
	#dm-outputs, [num_dm,time_steps]
	T, num_dm = dm_outputs.shape
	# pdb.set_trace()
	label_list = [0]*num_dm
	label_list[label_index[0]] = 1
	for i in range(num_dm):
		plt.plot(dm_outputs[:,i],label=label_list[i])
	plt.legend()
	plt.xlabel("Time Step")
	plt.ylabel("Neural Activities")
	plt.savefig(os.path.join(save_dir, "{}_dm_activities.png".format(index)))
