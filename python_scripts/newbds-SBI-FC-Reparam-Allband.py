#!/usr/bin/env python
# coding: utf-8

# RUN SBI-SGM in all bands
# 
# parameters order is  :taue,taui,tauG,speed,alpha,gii,gei (In second)

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
import argparse


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
from utils.stable import paras_table_check


# In[ ]:

parser = argparse.ArgumentParser(description='RUN SBI-FC')
parser.add_argument('--noise_sd', default=0.2, type=float, help='the noise sd added to data')
args = parser.parse_args()




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


# In[6]:


def get_mode(x):
    kde_est = scipy.stats.gaussian_kde(x)
    xs = np.linspace(x.min(), x.max(), 500)
    ys = kde_est(xs)
    return xs[np.argmax(ys)]


# In[ ]:

def _filter_unstable(theta_raw, prior_bds, x=None):
    """This fn is to remove unstable SGM parameters
        args: theta_raw: parameters: num of sps x dim
                order: ['Taue', 'Taui', 'TauC', 'Speed', 'alpha', 'gii', 'gei']
    """
    theta = _theta_raw_2out(theta_raw.numpy(), prior_bds)
    stable_idxs = paras_table_check(theta)
    
    # keep stable sps only
    theta_raw_stable = theta_raw[stable_idxs==0]
    if x is not None:
        x_stable = x[stable_idxs==0]
        return theta_raw_stable, x_stable
    else:
        return theta_raw_stable




# ### Some parameters

# In[7]:


# SC
ind_conn_xr = xr.open_dataarray('../data/individual_connectomes_reordered.nc')
ind_conn = ind_conn_xr.values

# PSD
ind_psd_xr = xr.open_dataarray('../data/individual_psd_reordered_matlab.nc')
ind_psd = ind_psd_xr.values
fvec = ind_psd_xr["frequencies"].values


# In[8]:


_paras = edict()
_paras.delta = [2, 3.5]
_paras.theta = [4, 7]
_paras.alpha = [8, 12]
_paras.beta = [13, 30]
_paras.beta_l = [13, 20]
_paras.beta_h = [15, 25]


# In[9]:


#taue,taui,tauG,speed,alpha,gii,gei


# In[10]:


paras = edict()

#paras.fc_types = ["delta", "theta", "alpha"]
paras.fc_types = ["delta", "theta", "alpha", "beta_l"]
paras.freqranges =  [np.linspace(_paras[fc_type][0], _paras[fc_type][1], 5) 
                     for fc_type in paras.fc_types]
print(paras.freqranges)
paras.fs = 600
paras.num_nodes = 86 # Number of cortical (68) + subcortical nodes
paras.par_low = np.asarray([0.005,0.005,0.005,5, 0.1,0.001,0.001])
paras.par_high = np.asarray([0.03, 0.20, 0.03,20,  1,    2,  0.7])
paras.prior_bds = np.array([paras.par_low, paras.par_high]).T
paras.prior_sd = 10
paras.add_v = 0.05

paras.SBI_paras = edict()
paras.SBI_paras.num_prior_sps = int(1e4)
paras.SBI_paras.density_model = "nsf"
paras.SBI_paras.num_round = 2
paras.SBI_paras.noise_sd = args.noise_sd


# In[11]:


# fn for reparemetering
_map_fn_torch = partial(logistic_torch, k=0.1)
_theta_raw_2out = partial(theta_raw_2out, map_fn=partial(logistic_np, k=0.1))


# In[ ]:





# ### Load the data

# In[12]:



def _add_v2con(cur_ind_conn):
    cur_ind_conn = cur_ind_conn.copy()
    add_v = np.max(cur_ind_conn)*paras.add_v # tuning 0.1
    np.fill_diagonal(cur_ind_conn[:34, 34:68], np.diag(cur_ind_conn[:34, 34:68]) + add_v)
    np.fill_diagonal(cur_ind_conn[34:68, :34], np.diag(cur_ind_conn[34:68, :34]) + add_v)
    np.fill_diagonal(cur_ind_conn[68:77, 77:], np.diag(cur_ind_conn[68:77, 77:]) + add_v)
    np.fill_diagonal(cur_ind_conn[77:, 68:77], np.diag(cur_ind_conn[77:, 68:77]) + add_v)
    return cur_ind_conn

if paras.add_v != 0:
    print(f"Add {paras.add_v} on diag")
    ind_conn_adds = [_add_v2con(ind_conn[:, :, ix]) for ix in range(36)]
    ind_conn = np.transpose(np.array(ind_conn_adds), (1, 2, 0))


# In[13]:


# Load true MEG FC time series:
true_FCs = []
for fc_type in paras.fc_types:
    dataPath = DATA_ROOT/f'./epochs3_MEG_FC_{fc_type}_DK_networks_coh.mat'
    data = loadmat(dataPath);
    true_FC = data[f"MEG_{fc_type}_FC_networks_coh"]
    true_FCs.append(true_FC)

true_FC.shape


# In[ ]:





# ## SBI

# ### Prior

# In[14]:


prior = MultivariateNormal(loc=torch.zeros(7), covariance_matrix=torch.eye(7)*(paras.prior_sd**2))


# In[15]:


def simulator(raw_params, brain, noise_sd, prior_bds, freqranges):
    params = []
    for raw_param, prior_bd in zip(raw_params, prior_bds):
        param =  _map_fn_torch(raw_param)*(prior_bd[1]-prior_bd[0]) + prior_bd[0]
        params.append(param)
    params = torch.tensor(params)
    
    params_dict = dict()
    params_dict["tau_e"] =  params[0].item()
    params_dict["tau_i"] =  params[1].item()
    params_dict["tauC"] =  params[2].item()
    params_dict["speed"] =  params[3].item()
    params_dict["alpha"] =  params[4].item()
    params_dict["gii"] =  params[5].item()
    params_dict["gei"] =  params[6].item()
    ress = []
    for freqrange in freqranges:
        modelFC = build_fc_freq_m(brain , params_dict, freqrange)
        modelFC_abs = np.abs(modelFC[:68, :68])
        res = _minmax_vec(modelFC_abs[np.triu_indices(68, k = 1)])
        ress.append(res)
    ress = np.concatenate(ress)
    noise =  np.random.randn(*ress.shape)*noise_sd
    return (ress+ noise).flatten()
    


# In[ ]:


#for cur_ind_idx in range(24, 36):
for cur_ind_idx in range(15, 24):
#for cur_ind_idx in range(3, 12):
    print(cur_ind_idx)
    # create spectrome brain:
    brain = Brain.Brain()
    brain.add_connectome(DATA_ROOT) # grabs distance matrix
    # re-ordering for DK atlas and normalizing the connectomes:
    brain.reorder_connectome(brain.connectome, brain.distance_matrix)
    brain.connectome =  ind_conn[:, :, cur_ind_idx] # re-assign connectome to individual connectome
    brain.bi_symmetric_c()
    brain.reduce_extreme_dir()
    
    simulator_sp = partial(simulator, brain=brain, 
                           noise_sd=paras.SBI_paras.noise_sd, 
                           prior_bds=paras.prior_bds, freqranges=paras.freqranges)
    simulator_wrapper, prior = prepare_for_sbi(simulator_sp, prior)
    inference = SNPE(prior=prior, density_estimator=paras.SBI_paras.density_model)
    proposal = prior 
    
    #the observed data
    res_vecs = []
    for true_FC in true_FCs:
        cur_obs_FC = np.abs(true_FC[:, :, cur_ind_idx])
        res_vec = _minmax_vec(cur_obs_FC[np.triu_indices(68, k = 1)])
        res_vecs.append(res_vec)
    res_vecs = np.concatenate(res_vecs)
    curX = torch.Tensor(res_vecs)
    num_spss = [10000, 1000, 1000]
    for ix in range(paras.SBI_paras.num_round):
        cur_num_sps = num_spss[ix]
        #cur_num_sps = paras.SBI_paras.num_prior_sps
        theta, x = simulate_for_sbi(simulator_wrapper, proposal,
                                    num_simulations=int(cur_num_sps*1.5),
                                    num_workers=20)
        theta_stable, x_stable = _filter_unstable(theta, paras.prior_bds, x)
        theta_stable, x_stable = theta_stable[:cur_num_sps, :], x_stable[:cur_num_sps, :]
        density_estimator = inference.append_simulations(
                            theta_stable, x_stable, proposal=proposal
                            ).train()
        posterior = inference.build_posterior(density_estimator)
        
        
        #update proposal 
        proposal = posterior.set_default_x(curX)
    
    #MR: multi-round
    save_fil = f"epochs3_newbdscorrectNewFC_posteriorMRmulDiffNum_{'-'.join(paras.fc_types)}_" +                f"num{paras.SBI_paras.num_prior_sps}_" +                f"density{paras.SBI_paras.density_model}_" +                f"MR{paras.SBI_paras.num_round}_" +                f"noise_sd{paras.SBI_paras.noise_sd*100:.0f}_" +               f"addv{paras.add_v*100:.0f}" +               f"/ind{cur_ind_idx}.pkl"
        
    save_pkl(RES_ROOT/save_fil, proposal)


# In[ ]:





# In[ ]:




