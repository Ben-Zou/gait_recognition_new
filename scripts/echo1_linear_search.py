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
from lib.models import Echo1
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
parser.add_argument('--tau', type=float, default=50, help='time constant of echo network')
parser.add_argument('--tau_dm', type=float, default=20, help='time constant of decision making')
parser.add_argument('--n_echo', type=int, default=1000, help='num_units in echo state network')
parser.add_argument('--rho_scale', type=float, default=1.1, help='rho_scale of reservoir network')
parser.add_argument('--num_dm', type=int, default=2, help='number of dm cells')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for adam')
parser.add_argument('--clipval', type=float, default=1.0, help='gradient clipping value')
parser.add_argument('--T', type=int, default=100, help='length of input sequence')
parser.add_argument('--target_mode', type=str, default="x_target", help='save directory')
parser.add_argument('--trails_per_person', type=int, default=30, help='trails per person used for training')
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


	n_trails = args.trails_per_person
	val_data = train_data[:args.num_dm,-15:,:args.T]
	train_data = train_data[:args.num_dm,:n_trails,:args.T]
	test_data = test_data[:,:,:args.T]

	train_size = n_trails*args.num_dm
	val_size = 15*args.num_dm  # 15 trails for testing.
	test_size = 50*args.num_dm

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

dm_dict = get_dm_dict(args.num_dm,
					  tau_dm=args.tau_dm,
					  target_mode=args.target_mode)

## model
echo_dict = dict(tau  = args.tau,
				 dt   = 1,
				 scale  = args.rho_scale,
				 spars_echo = 0.1,
				 scale_echo = 1.,
				 spars_p  = 0.1,
				 init_mode = "mode_a")

model = Echo1(inp_size,
			  args.n_echo,
			  args.num_dm,
			  echo_dict,
			  dm_dict=None)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
save_dir='../res/{}/'.format(args.save_dir)
os.makedirs(save_dir, exist_ok=True)
runner = Runner(model,
				optimizer)

for seed_i in range(100,1100,100):
	set_seed(seed_i,cuda)
	trainloader,validloader, testloader = set_data()
	runner.model.init_weights()
	runner.model.to(device)
	runner.init_runner()
	runner.register_hook(OptimizerHook(grad_clip=args.clipval))
	runner.register_hook(AccMetricHook(dur=1,save_dir=save_dir))

	runner.name = "{},{},{},{},{}".format(args.tau, args.tau_dm, args.n_echo, args.num_dm,seed_i)

	runner.run([trainloader,validloader],workflow=[('train',1),('val',1)],max_epochs=args.n_epochs)
	best_weights = torch.load(save_dir+"{}.pth".format(runner.name))
	model.load_state_dict(best_weights)
	runner.val(testloader)
















##
