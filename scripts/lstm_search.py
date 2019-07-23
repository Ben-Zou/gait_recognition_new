import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pdb

import sys
sys.path.append("../")
from lib.model_utils import get_dm_dict
from lib.utils import set_seed
from lib.models import PlainRNN
from lib.runner import Runner
from lib.runner import OptimizerHook, AccMetricHook
from lib.data import gait_frame_2
from lib.data import gait_frame_dm_xtarget

import argparse

parser = argparse.ArgumentParser(description='Decision Making')
parser.add_argument('--save_dir', type=str, default="test", help='seed value')
parser.add_argument('--seed', type=int, default=5000, help='seed value')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--n_hidden', type=int, default=20, help='num_units in echo state network')
parser.add_argument('--n_output', type=int, default=2, help='number of dm cells')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for adam')
parser.add_argument('--clipval', type=float, default=1.0, help='gradient clipping value')
parser.add_argument('--T', type=int, default=100, help='length of input sequence')
parser.add_argument('--rnn_type', type=str, default="lstm", help='save directory')
parser.add_argument('--trails_per_person', type=int, default=50, help='trails per person used for training')
args = parser.parse_args()

cuda = torch.cuda.is_available()
set_seed(args.seed,cuda)

if cuda:
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

inp_size = 40*40


def set_data():
	## load dataset
	# [5,50,100,40,40]; [5,50,100,40,40]
	# train_data = np.load("../data/5class_50sam_40x40_kinetics_train.npy")
	# test_data = np.load("../data/5class_50sam_40x40_kinetics_val.npy")

	train_data = np.load("../data/5p_40x40_kinetics_train_all.npy")
	test_data = np.load("../data/5p_40x40_kinetics_val_all.npy")

	# train_data = np.load("../data/20p_40x40_30samples_50f_kinetics_train_all.npy")
	# test_data = np.load("../data/20p_40x40_30samples_50f_kinetics_val_all.npy")


	n_trails = args.trails_per_person
	_,tot_n_trails,_,_,_ = train_data.shape
	index = np.random.choice(tot_n_trails,size=n_trails+15)
	train_index = index[:n_trails]
	val_index = index[-15:]

	val_data = train_data[:args.n_output,val_index,:args.T]
	train_data = train_data[:args.n_output,train_index,:args.T]
	test_data = test_data[:,:,:args.T]

	train_size = n_trails*args.n_output
	val_size = 15*args.n_output  # 15 trails for testing.
	test_size = 50*args.n_output

	trainset = gait_frame_2(train_data,T=args.T)
	validset = gait_frame_2(val_data,T=args.T)
	testset = gait_frame_2(test_data,T=args.T)

	# trainset = gait_frame_dm_xtarget(train_data,T=args.T,Tstim=args.T)
	# validset = gait_frame_dm_xtarget(val_data,T=args.T,Tstim=args.T)
	# testset = gait_frame_dm_xtarget(test_data,T=args.T,Tstim=args.T)

	train_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(train_size))
	valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(val_size))
	test_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(test_size))

	# 20*5*8 = 800
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler,num_workers=2)
	validloader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=False, sampler=valid_sampler, num_workers=2)
	testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,sampler=test_sampler ,num_workers=2)

	return trainloader, validloader, testloader

if cuda:
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

model = PlainRNN(inp_size,
			  args.n_hidden,
			  args.n_output,
			  rnn_type=args.rnn_type)


optimizer = optim.Adam(model.parameters(), lr=args.lr)
save_dir='../res/gait_rnn_5way_30trails/{}/'.format(args.save_dir)
os.makedirs(save_dir, exist_ok=True)
runner = Runner(model,
				optimizer)

for seed_i in range(100,3100,100):
	set_seed(seed_i,cuda)
	trainloader,validloader, testloader = set_data()
	runner.model.init_weights()
	runner.model.to(device)
	runner.optimizer = optim.Adam(model.parameters(), lr=args.lr)
	runner.init_runner()
	runner.register_hook(OptimizerHook(grad_clip=args.clipval))
	runner.register_hook(AccMetricHook(dur=1,save_dir=save_dir))

	runner.name = "{},{},{}".format(args.n_hidden, args.n_output,seed_i)

	runner.run([trainloader,validloader],workflow=[('train',1),('val',1)],max_epochs=args.n_epochs)
	best_weights = torch.load(save_dir+"{}.pth".format(runner.name))
	model.load_state_dict(best_weights)
	runner.val(testloader)
	os.remove(save_dir+"{}.pth".format(runner.name))
















##
