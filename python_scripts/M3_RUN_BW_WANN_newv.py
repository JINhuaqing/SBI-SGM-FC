#!/usr/bin/env python
# coding: utf-8

# RUN SBI-SGM in alpha, new bounds, new SGM, only three parameters needed
# 
# parameters order is  :tauG,speed,alpha (In second)
# 
# And now, I construct prior from the results with Annealing
# 

# ## Import some pkgs

# In[1]:


import sys
sys.path.append("../mypkg")

import scipy
import itertools

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import trange
from scipy.io import loadmat
from functools import partial
from easydict import EasyDict as edict


# In[2]:


# SBI and torch
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi import analysis
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as sutils

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions.multivariate_normal import MultivariateNormal


# In[3]:


# my own fns
from brain import Brain
from FC_utils import build_fc_freq_m
from constants import RES_ROOT, DATA_ROOT
from utils.misc import load_pkl, save_pkl
from utils.reparam import theta_raw_2out, logistic_np, logistic_torch

import argparse


parser = argparse.ArgumentParser(description='RUN SBIxANN')
parser.add_argument('--band', default="alpha", type=str, help='The freq band')
parser.add_argument('--nepoch', default=100, type=int, help='Emp FC epoch')
parser.add_argument('--noise_sd', default=1.2, type=float, help='Noise sd added')
parser.add_argument('--num_prior_sps', default=1000, type=int, help='Num of samples used for each round')
parser.add_argument('--num_round', default=3, type=int, help='Num of train rounds')
args = parser.parse_args()


# In[ ]:





# ## Some fns

# In[4]:


_minmax_vec = lambda x: (x-np.min(x))/(np.max(x)-np.min(x))


# In[5]:


# transfer vec to a sym mat
def vec_2mat(vec):
    mat = np.zeros((68, 68))
    mat[np.triu_indices(68, k = 1)] = vec
    mat = mat + mat.T
    return mat


# In[ ]:





# ### Some parameters

# In[6]:


# SC
ind_conn_xr = xr.open_dataarray(DATA_ROOT/'individual_connectomes_reordered.nc')
ind_conn = ind_conn_xr.values

# PSD
ind_psd_xr = xr.open_dataarray(DATA_ROOT/'individual_psd_reordered_matlab.nc')
ind_psd = ind_psd_xr.values
fvec = ind_psd_xr["frequencies"].values;


# In[7]:


_paras = edict()
_paras.delta = [2, 3.5]
_paras.theta = [4, 7]
_paras.alpha = [8, 12]
_paras.beta_l = [13, 20]


# In[17]:


paras = edict()

paras.band = args.band
paras.nepoch = args.nepoch
paras.save_prefix = "rawfc2newv"
paras.freqrange =  np.linspace(_paras[paras.band][0], _paras[paras.band][1], 5)
print(paras.freqrange)
paras.fs = 600
paras.num_nodes = 86 # Number of cortical (68) + subcortical nodes
#paras.par_low = np.asarray([0.005,0.005,0.005,5, 0.1,0.001,0.001])
#paras.par_high = np.asarray([0.03, 0.20, 0.03,20,  1,    2,  0.7])
#paras.names = ["Taue", "Taui", "TauC", "Speed", "alpha", "gii", "gei"]
paras.par_low = np.asarray([0.005, 6, 0.1])
paras.par_high = np.asarray([0.03, 18, 1])
paras.names = ["TauC", "Speed", "alpha"]
paras.prior_bds = np.array([paras.par_low, paras.par_high]).T
paras.prior_sd = 3
paras.add_v = 0.01
paras.k = 1

paras.SBI_paras = edict()
paras.SBI_paras.noise_sd = args.noise_sd
paras.SBI_paras.num_prior_sps = args.num_prior_sps
paras.SBI_paras.num_round = args.num_round # 3
paras.SBI_paras.density_model = "nsf"


# In[18]:


# fn for reparemetering
_map_fn_torch = partial(logistic_torch, k=paras.k)
_theta_raw_2out = partial(theta_raw_2out, map_fn=partial(logistic_np, k=paras.k), prior_bds=paras.prior_bds);


# In[ ]:





# ### Load the data

# In[19]:



def _add_v2con(cur_ind_conn):
    cur_ind_conn = cur_ind_conn.copy()
    add_v = np.quantile(cur_ind_conn, 0.99)*paras.add_v # tuning 0.1
    np.fill_diagonal(cur_ind_conn[:34, 34:68], np.diag(cur_ind_conn[:34, 34:68]) + add_v)
    np.fill_diagonal(cur_ind_conn[34:68, :34], np.diag(cur_ind_conn[34:68, :34]) + add_v)
    np.fill_diagonal(cur_ind_conn[68:77, 77:], np.diag(cur_ind_conn[68:77, 77:]) + add_v)
    np.fill_diagonal(cur_ind_conn[77:, 68:77], np.diag(cur_ind_conn[77:, 68:77]) + add_v)
    return cur_ind_conn

if paras.add_v != 0:
    print(f"Add {paras.add_v} on diag")
    ind_conn_adds = [_add_v2con(ind_conn[:, :, ix]) for ix in range(36)]
    ind_conn = np.transpose(np.array(ind_conn_adds), (1, 2, 0))


# In[20]:


# em FC
fc_root = RES_ROOT/"emp_fcs2"
def _get_fc(sub_ix, bd):
    fil = list(fc_root.rglob(f"*{paras.band}*{paras.nepoch}/sub{sub_ix}.pkl"))[0]
    return load_pkl(fil, verbose=False)

fcs = np.array([_get_fc(sub_ix, paras.band) for sub_ix in range(36)]);


# In[ ]:





# ## SBI

# ### Prior

# In[21]:


# get the informative prior
def _get_prior(ind_idx):
    fil = list(RES_ROOT.glob(f"{paras.save_prefix}_ANN_{paras.band}_ep{paras.nepoch}"
                             f"_addv{paras.add_v*100:.0f}/ind{ind_idx}.pkl"))[0];
    ann_res = load_pkl(fil, verbose=False);
    ann_res.ann_res.x
    prior = MultivariateNormal(loc=torch.Tensor(ann_res.ann_res.x), 
                           covariance_matrix=torch.eye(3)*(paras.prior_sd**2))
    return prior


# In[22]:


def simulator(raw_params, brain, noise_sd, prior_bds, freqrange):
    params = []
    for raw_param, prior_bd in zip(raw_params, prior_bds):
        param =  _map_fn_torch(raw_param)*(prior_bd[1]-prior_bd[0]) + prior_bd[0]
        params.append(param)
    params = torch.tensor(params)
    
    params_dict = dict()
    params_dict["tauC"] =  params[0].item()
    params_dict["speed"] =  params[1].item()
    params_dict["alpha"] =  params[2].item()
    modelFC = build_fc_freq_m(brain , params_dict, freqrange)
    modelFC_abs = np.abs(modelFC[:68, :68])
    res = _minmax_vec(modelFC_abs[np.triu_indices(68, k = 1)])
    noise =  np.random.randn(*res.shape)*noise_sd
    return (res+ noise).flatten()


# In[23]:


for cur_ind_idx in range(0, 36):
    print(cur_ind_idx)
    save_fil = f"{paras.save_prefix}_SBIxANNBW_{paras.band}_" +                 f"ep{paras.nepoch}_" +                f"num{paras.SBI_paras.num_prior_sps}_" +                f"density{paras.SBI_paras.density_model}_" +                f"MR{paras.SBI_paras.num_round}_" +                f"noise_sd{paras.SBI_paras.noise_sd*100:.0f}_" +               f"addv{paras.add_v*100:.0f}" +               f"/ind{cur_ind_idx}.pkl"
    if (RES_ROOT/save_fil).exists():
        # thanks to the buggy SCS
        continue
    
    
    # create spectrome brain:
    brain = Brain.Brain()
    brain.add_connectome(DATA_ROOT) # grabs distance matrix
    # re-ordering for DK atlas and normalizing the connectomes:
    brain.reorder_connectome(brain.connectome, brain.distance_matrix)
    brain.connectome =  ind_conn[:, :, cur_ind_idx] # re-assign connectome to individual connectome
    brain.bi_symmetric_c()
    brain.reduce_extreme_dir()
    
    simulator_sp = partial(simulator, 
                           brain=brain, 
                           noise_sd=paras.SBI_paras.noise_sd, 
                           prior_bds=paras.prior_bds, 
                           freqrange=paras.freqrange)
    prior = _get_prior(cur_ind_idx)
    simulator_wrapper, prior = prepare_for_sbi(simulator_sp, prior)
    inference = SNPE(prior=prior, density_estimator=paras.SBI_paras.density_model)
    proposal = prior 
    
    #the observed data
    cur_obs_FC = np.abs(fcs[cur_ind_idx])
    curX = torch.Tensor(_minmax_vec(cur_obs_FC[np.triu_indices(68, k = 1)]))
    #num_spss = [10000, 10000, 5000]
    for ix in range(paras.SBI_paras.num_round):
        theta, x = simulate_for_sbi(simulator_wrapper, proposal,
                                    num_simulations=int(paras.SBI_paras.num_prior_sps),
                                    num_workers=30)
        density_estimator = inference.append_simulations(
                            theta, x, proposal=proposal
                            ).train()
        posterior = inference.build_posterior(density_estimator)
        
        proposal = posterior.set_default_x(curX)
    
    save_pkl(RES_ROOT/save_fil, proposal)


# In[ ]:





# In[ ]:




