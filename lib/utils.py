from functools import partial
import numpy as np
from six.moves import map, zip
import torch
import random
import glob
import os.path as osp

def multi_apply(func,*args,**kwargs):
	pfunc = partial(func,**kwargs) if kwargs else func
	map_results = map(func,*args)
	return tuple(map(list,zip(*map_results)))

def set_seed(seed,cuda=True):
	random.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	if cuda:
		torch.cuda.manual_seed(seed)

def stat_res_from_txt(path):
	files_sort = glob.glob(osp.join(path,"*.txt"))
	if files_sort == []:
		return None
	files_sort = sorted(files_sort)
	acc_list = []
	for fi in files_sort:
		with open(fi,'r') as f:
			acc_ = f.readline().split("\n")[0]
			acc_list.append(float(acc_))

	mean = np.mean(acc_list)
	std = np.std(acc_list)
	max = np.max(acc_list)
	min = np.min(acc_list)
	return (mean,std,max,min)
