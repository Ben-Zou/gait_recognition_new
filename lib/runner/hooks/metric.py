from .hook import Hook

import torch
import numpy as np
import os.path as osp

import pdb


class AccMetricHook(Hook):
	"""
	used for caculate the accuracy.
	"""

	def __init__(self,dur=1,save_dir="./"):
		self.dur = dur
		self.num_corrects = 0
		self.tot_samples = 0
		self.save_dir = save_dir
		self.best_acc = 0

	def after_val_iter(self,runner):
		# [T,batch,n_outputs]
		outputs = np.array(runner.outputs["outputs"]).transpose(1,0,2)
		batch, T, num_features = outputs.shape
		outputs = outputs[:,-self.dur:,:].reshape(batch,self.dur,num_features)
		predicts = outputs.mean(axis=1).reshape(-1).argmax(axis=-1)
		labels = runner.outputs["labels"]

		self.num_corrects  += sum(labels == predicts)

		self.tot_samples += batch

	def after_val_epoch(self,runner):
		acc = self.num_corrects/self.tot_samples


		if self.best_acc < acc:
			self.best_acc = acc
			torch.save(runner.model.state_dict(),osp.join(self.save_dir,runner.name+".pth"))

		print("Validation: Iter {}, Epoch {}, Num_samples {},Best val Acc {}, Acc {}".format(
						   runner.iter, runner.epoch, self.tot_samples, self.best_acc, acc))

		if runner.epoch == runner.max_epochs:
			np.savetxt(osp.join(self.save_dir,"{}_dur_{}_acc.txt".format(runner.name,self.dur)),[acc])

		self.num_corrects = 0
		self.tot_samples = 0
















#
