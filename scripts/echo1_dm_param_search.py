import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pdb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
from lib.model_utils import get_dm_dict
from lib.utils import set_seed, stat_res_from_txt
from lib.models import Echo1
from lib.runner import Runner
from lib.runner import OptimizerHook, AccMetricHook
from lib.data import gait_frame_2
from lib.data import gait_frame_dm_xtarget

import argparse
import time

parser = argparse.ArgumentParser(description='Decision Making')
parser.add_argument('--save_dir', type=str, default="test_param", help='seed value')
parser.add_argument('--seed', type=int, default=5000, help='seed value')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs')
parser.add_argument('--tau', type=float, default=50, help='time constant of echo network')
parser.add_argument('--tau_dm', type=float, default=20, help='time constant of decision making')
parser.add_argument('--n_echo', type=int, default=1000, help='num_units in echo state network')
parser.add_argument('--rho_scale', type=float, default=1.1, help='rho_scale of reservoir network')
parser.add_argument('--num_dm', type=int, default=5, help='number of dm cells')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for adam')
parser.add_argument('--clipval', type=float, default=1.0, help='gradient clipping value')
parser.add_argument('--T', type=int, default=50, help='length of input sequence')
parser.add_argument('--target_mode', type=str, default="x_target", help='save directory')
parser.add_argument('--trails_per_person', type=int, default=1, help='trails per person used for training')

parser.add_argument('--scale_echo',type=float, default=1.0, help='echo scale factor')
parser.add_argument('--normalize', action='store_true', default=False, help='plot figs or not')
parser.add_argument('--num_valid', type=int, default=15, help='length of input sequence')

parser.add_argument('--tau_begin', type=float, default=5.0, help='search-value')
parser.add_argument('--tau_end', type=float, default=6.0, help='search-value')
parser.add_argument('--tau_step', type=float, default=1.0, help='search-value')
parser.add_argument('--tau_dm_begin', type=float, default=10.0, help='search-value')
parser.add_argument('--tau_dm_end', type=float, default=11.0, help='search-value')
parser.add_argument('--tau_dm_step', type=float, default=1.0, help='search-value')
parser.add_argument('--rho_begin', type=float, default=1.0, help='search-value')
parser.add_argument('--rho_end', type=float, default=2.0, help='search-value')
parser.add_argument('--rho_step', type=float, default=1.0, help='search-value')

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

	# train_data = np.load("../data/5p_40x40_kinetics_train_all.npy")
	# test_data = np.load("../data/5p_40x40_kinetics_val_all.npy")

	train_data = np.load("../data/20p_40x40_30samples_50f_kinetics_train_all.npy")
	test_data = np.load("../data/20p_40x40_30samples_50f_kinetics_val_all.npy")


	n_trails = args.trails_per_person
	val_data = train_data[:args.num_dm,-args.num_valid:,:args.T]
	train_data = train_data[:args.num_dm,:n_trails,:args.T]
	test_data = test_data[:,:,:args.T]

	train_size = n_trails*args.num_dm
	val_size = args.num_valid*args.num_dm  # 15 trails for testing.
	test_size = 50*args.num_dm

	# trainset = gait_frame_2(train_data,T=args.T)
	# validset = gait_frame_2(val_data,T=args.T)
	# testset = gait_frame_2(test_data,T=args.T)

	trainset = gait_frame_dm_xtarget(train_data,T=args.T,Tstim=args.T,normalize=args.normalize)
	validset = gait_frame_dm_xtarget(val_data,T=args.T,Tstim=args.T,normalize=args.normalize)
	testset = gait_frame_dm_xtarget(test_data,T=args.T,Tstim=args.T,normalize=args.normalize)

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
				 scale_echo = args.scale_echo,
				 spars_p  = 0.1,
				 init_mode = "mode_a")

model = Echo1(inp_size,
			  args.n_echo,
			  args.num_dm,
			  echo_dict,
			  dm_dict=dm_dict)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
runner = Runner(model,
				optimizer)

#


"""
search the optimal parameters.

tau: runner.model.simple_echo.alpha = 1/tau
rho: runner.model.simple_echo.scale = rho

tau [tau_begin,tau_end,tau_step]
rho [rho_begin,rho_end,rho_step]
"""

rho_params = np.arange(args.rho_begin,args.rho_end,args.rho_step).tolist()
tau_params = np.arange(args.tau_begin,args.tau_end,args.tau_step).tolist()
tau_dm_params = np.arange(args.tau_dm_begin,args.tau_dm_end,args.tau_dm_step).tolist()

final_mean = []
final_std = []
final_max = []
final_min = []

for rho_i in rho_params:
	for tau_i in tau_params:
		for tau_dm_i in tau_dm_params:
			save_dir='../res/echo1_dm_diffways/{}/rho_{},tau_{},tau_dm_{}/'.format(args.save_dir,rho_i,tau_i,tau_dm_i)
			os.makedirs(save_dir, exist_ok=True)
			for seed_i in range(100,1100,100):
				set_seed(seed_i,cuda)

				## init the network and dataset
				trainloader,validloader, testloader = set_data()

				## init the params
				runner.model.simple_echo.alpha = 1./tau_i #torch.FloatTensor([1./tau_i]).to(device)
				runner.model.simple_echo.scale = rho_i
				runner.model.dm.alpha = 1./tau_dm_i

				runner.model.init_weights()
				runner.model.to(device)
				runner.optimizer = optim.Adam(model.parameters(), lr=args.lr)
				runner.init_runner()
				runner.register_hook(OptimizerHook(grad_clip=args.clipval))
				runner.register_hook(AccMetricHook(dur=1,save_dir=save_dir))

				runner.name = "seed_{}".format(seed_i)

				runner.run([trainloader,validloader],workflow=[('train',1),('val',1)],max_epochs=args.n_epochs)
				best_weights = torch.load(save_dir+"{}.pth".format(runner.name))
				model.load_state_dict(best_weights)
				runner.val(testloader)
				os.remove(save_dir+"{}.pth".format(runner.name))

		# stat results
		mean_i, std_i, max_i, min_i = stat_res_from_txt(save_dir)
		final_mean.append(mean_i)
		final_std.append(std_i)
		final_max.append(max_i)
		final_min.append(min_i)


# #### save final results and plot
# shape = (len(rho_params),len(tau_params),len(tau_dm_params))
# final_mean = np.array(final_mean).reshape(shape)
# final_std = np.array(final_std).reshape(shape)
# final_max = np.array(final_max).reshape(shape)
# final_min = np.array(final_min).reshape(shape)
#
# ###
# save_dir='../res/echo1_dm/{}/final_res'.format(args.save_dir)
# os.makedirs(save_dir, exist_ok=True)
#
# ## save final res
# np.save(os.path.join(save_dir,"final_mean.npy"),final_mean)
# np.save(os.path.join(save_dir,"final_std.npy"),final_std)
# np.save(os.path.join(save_dir,"final_max.npy"),final_max)
# np.save(os.path.join(save_dir,"final_min.npy"),final_min)





##
# plot_data = [final_mean,final_std, final_max, final_min]
# plot_label = ["mean","std","max","min"]
#
# plt.figure(figsize=(10,8))
# for i, (d,l) in enumerate(zip(plot_data, plot_label)):
# 	plt.subplot(2,2,i+1)
# 	plt.imshow(d,aspect='auto')
# 	plt.colorbar()
# 	plt.title(l+"{}".format(np.max(d)))
# plt.savefig(os.path.join(save_dir,"summary.png"))
#















##
