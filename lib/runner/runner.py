import time
import torch
import os.path as osp
from . import Hook


class Runner(object):

	def __init__(self,
				 model,
				 optimizer,
				 name = "test",
				 ):
		self.model = model
		self.optimizer = optimizer
		self.name = name

		self._hooks = []
		self._iter = 0
		self._epoch = 0
		self._inner_iter = 0
		self._max_epochs = 0
		self._max_iters = 0


	def init_runner(self):
		self._hooks = []
		self._iter = 0
		self._epoch = 0
		self._inner_iter = 0
		self._max_epochs = 0
		self._max_iters = 0

	@property
	def iter(self):
		return self._iter

	@property
	def epoch(self):
		return self._epoch

	@property
	def inner_iter(self):
		return self._inner_iter

	@property
	def max_epochs(self):
		return self._max_epochs

	@property
	def max_iters(self):
		return self._max_iters

	@property
	def hooks(self):
		return self._hooks


	def register_hook(self,hook):
		assert isinstance(hook, Hook)
		self._hooks.append(hook)

	def save_checkpoint(self,
						out_dir,
						filename_tmpl="epoch_{}.pth",):
		filename = osp.join(out_dir, filename_tmpl.format(self.epoch+1))
		torch.save(self.model.state_dict(),filename)

	def call_hook(self,fn_name):
		for hook in self._hooks:
			getattr(hook,fn_name)(self)

	def train(self,data_loader,**kwargs):
		self.model.train()
		self.mode = "train"
		self.data_loader = data_loader
		self._max_iters = self._max_epochs*len(data_loader)

		self.call_hook("before_train_epoch")
		for i, data_batch in enumerate(data_loader):
			self._inner_iter = i
			self.call_hook("before_train_iter")
			outputs = self.model(data_batch, train_mode=True, **kwargs)
			self.outputs = outputs
			self.call_hook("after_train_iter")
			self._iter += 1
		self.call_hook("after_train_epoch")
		self._epoch += 1

	def val(self, data_loader, **kwargs):
		self.model.eval()
		self.mode = 'val'
		self.data_loader = data_loader
		self.call_hook("before_val_epoch")
		for i, data_batch in enumerate(data_loader):
			self._inner_iter = i
			self.call_hook("before_val_iter")
			with torch.no_grad():
				outputs = self.model(data_batch, train_mode=False, **kwargs)
			self.outputs = outputs
			self.call_hook('after_val_iter')

		self.call_hook("after_val_epoch")

	def run(self,dataloders,workflow,max_epochs,**kwargs):
		"""
		start running.

		dataloders: a list, for example, [train_dataloader,val_dataloder].
		workflow: a list, for example, [('train',2),('val',1)]
				  means 2 epochs for training, 1 epoch for validation.
		max_epochs: total training epochs
		"""
		assert isinstance(dataloders,list)
		assert len(dataloders) == len(workflow)

		self._max_epochs = max_epochs

		self.call_hook("before_run")

		while self.epoch < max_epochs:
			for i, flow in enumerate(workflow):
				mode, epochs = flow
				epoch_runner = getattr(self,mode)

				##
				for _ in range(epochs):
					if mode == 'train' and self.epoch >= max_epochs:
						return
					epoch_runner(dataloders[i],**kwargs)

		self.call_hook("after_run")
