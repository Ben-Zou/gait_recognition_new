from .rnn import LogAct
from .rnn import RecLogAct
import pdb

def get_dm_dict(num_dm,**kwargs):

	if num_dm == 2:
		dm_dict = dict(dt  = 1.,
					   Je = 8.,
					   Jm = -2.,
					   gamma = 0.1,
					   I0 = 0.66,
					   target_mode = kwargs['target_mode'],
					   activation = LogAct(alpha=1.5,beta=4.,gamma=0.1,thr=6.),
					   rec_activation = RecLogAct(alpha=1.5,beta=4.,gamma=0.1,thr=6.),
					   taus = kwargs["tau_dm"])

	if num_dm == 5:
		dm_dict = dict(dt  = 1.,
					   Je = 9.,
					   Jm = -5,
					   gamma = 0.1,
					   I0 = -0.4,
					   target_mode = kwargs['target_mode'],
					   activation = LogAct(alpha=1.5,beta=3.2,gamma=0.1,thr=3.),
					   rec_activation = RecLogAct(alpha=1.5,beta=3.2,gamma=0.1,thr=3.),
					   taus = kwargs["tau_dm"])

	if num_dm == 10:
		dm_dict = dict(dt  = 1.,
					   Je = 18.,
					   Jm = -11.,
					   I0 = 1.11,
					   gamma = 0.1,
					   target_mode = kwargs['target_mode'],
					   activation = LogAct(alpha=1.5,beta=1.,gamma=0.1,thr=1.),
					   rec_activation = RecLogAct(alpha=1.5,beta=1.,gamma=0.1,thr=1.),
					   taus = kwargs["tau_dm"])

	if num_dm == 15:
		dm_dict = dict(dt  = 1.,
					   Je = 20.,
					   Jm = -20.,
					   I0 = 0.4,
					   gamma = 0.1,
					   target_mode = kwargs['target_mode'],
					   activation = LogAct(alpha=1.5,beta=0.8,gamma=0.1,thr=-5.),
					   rec_activation = RecLogAct(alpha=1.5,beta=0.8,gamma=0.1,thr=-5.),
					   taus = kwargs["tau_dm"])


	if num_dm == 20:
		dm_dict = dict(dt  = 1.,
					   Je = 27.,
					   Jm = -27.,
					   I0 = 0.3,
					   gamma = 0.1,
					   target_mode = kwargs['target_mode'],
					   activation = LogAct(alpha=1.5,beta=0.5,gamma=0.1,thr=-9.),
					   rec_activation = RecLogAct(alpha=1.5,beta=0.5,gamma=0.1,thr=-9.),
					   taus = kwargs["tau_dm"])
	return dm_dict
