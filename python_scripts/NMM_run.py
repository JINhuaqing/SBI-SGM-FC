#!/usr/bin/env python
# coding: utf-8

# This notebook is to run NMM with `neurolib` pkg
# 
# I refer to the following files 
# 
# 1. Paper: Cakan_et_al_CC_2023_neurolib_pythonpkg_NMM.pdf
# 
# 2. Web tutorials
# 
#     a. https://neurolib-dev.github.io/examples/example-3-meg-functional-connectivity/
#     
#     b. https://neurolib-dev.github.io/examples/example-2.2-evolution-brain-network-aln-resting-state-fit/
#     
#     c. https://neurolib-dev.github.io/examples/example-0.4-wc-minimal/
# 3. My own notebooks: 
#     MEG_FC_emp.ipynb and Other_method
# 

# In[1]:



# In[2]:


import sys
sys.path.append("../mypkg")

import scipy
import itertools

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import trange
from functools import partial
from easydict import EasyDict as edict
from collections import defaultdict as ddict
from IPython.display import display


# In[3]:


# the main pkg to implemente the NMM
import neurolib
from neurolib.models.wc import WCModel
from neurolib.optimize.evolution import Evolution
from neurolib.utils.signal import Signal 
from neurolib.utils.parameterSpace import ParameterSpace
from neurolib.utils.loadData import Dataset
import neurolib.utils.functions as func

# to calculate coh FC
import mne
from mne_connectivity import spectral_connectivity_epochs
# suppress the log from spec... fn 
mne.set_log_level('ERROR')


# In[4]:


# my own fns
from brain import Brain
from FC_utils import build_fc_freq_m
from constants import RES_ROOT, DATA_ROOT, FIG_ROOT
from utils.misc import load_pkl, save_pkl
from utils.measures import geodesic_dist, reg_R_fn, lin_R_fn, lap_mat_fn

plt.style.use(FIG_ROOT/"base.mplstyle")

import argparse


# In[ ]:


parser = argparse.ArgumentParser(description='RUN NMMC')
parser.add_argument('--band', default="alpha", type=str, help='The freq band')
args = parser.parse_args()



# In[ ]:





# # Prepare

# ## Fns

# In[5]:


_minmax_vec = lambda x: (x-np.min(x))/(np.max(x)-np.min(x));
    
def _model2data(wc):
    """Get time series data from the run model"""
    sim_signal = xr.DataArray(wc.exc[:68, int(1000/wc.params.dt):], 
                              dims=("regions", "time"), 
                              coords={"time": wc.t[int(1000/wc.params.dt):]/1000}, 
                              attrs={'atlas':'DK68'});
    sim_signal = Signal(sim_signal);
    sim_signal.resample(to_frequency=600)
    return np.array(sim_signal.data)

def _get_wc(sc, dmat):
    """Get the NMM model with SC and dmat
       Set the default parameters
    """
    wc = WCModel(Cmat = sc, 
                 Dmat = dmat,
                 seed=0)
    wc.params['duration'] = 61*1000 # 60 seconds -- The unit is in 1 ms
    wc.params['dt'] = 1;
    wc.params['K_gl'] = 6.55
    wc.params['exc_ext'] = 1.58
    wc.params['inh_ext'] = 2.83
    wc.params['sigma_ou'] = 0.02
    return wc



def _get_simu_fc(input_signal, paras={}):
    """
    Calculate functional connectivity from input_signal using the parameters in paras.

    Parameters:
    input_signal (numpy.ndarray): Input signal with shape (n_channels, n_timepont)
    paras (dict): Dictionary containing the following keys and values:
        - nepoch (int): Number of epochs
        - fc_type (str): Functional connectivity method
        - f_skip (int): Frequency skip
        - bd (str): Frequency band
        - faverage (bool): If True, average the connectivity over frequency bands

    Returns:
    numpy.ndarray: Functional connectivity matrix
    """
    def __2matf(mat):
        mat_f = mat + mat.T
        mat_f = mat_f - np.diag(np.diag(mat))
        return mat_f
    _bd_limits = edict()
    _bd_limits.delta = [2, 3.5]
    _bd_limits.theta = [4, 7]
    _bd_limits.alpha = [8, 12]
    _bd_limits.beta_l = [13, 20]
    _fs = 600 # sampling freq

    _paras = edict()
    _paras.bd = "alpha"
    _paras.fc_type = "coh"
    _paras.f_skip = 0
    _paras.nepoch = 10
    _paras.update(paras)
    

    if _paras.nepoch == 1:
        input_signal = input_signal[np.newaxis]
    else:
        input_signal = input_signal.reshape(68, _paras.nepoch, -1).transpose(1, 0, 2)
    
    ts_con = spectral_connectivity_epochs(input_signal,
                                          names=None, 
                                          method=_paras.fc_type, 
                                          indices=None, 
                                          sfreq=_fs, 
                                          mode='multitaper',
                                          fmin=_bd_limits[_paras.bd][0], 
                                          fmax=_bd_limits[_paras.bd][1],
                                          fskip=_paras.f_skip, 
                                          faverage=True, 
                                          tmin=None, 
                                          tmax=None,  
                                          mt_bandwidth=None, 
                                          mt_adaptive=False, 
                                          mt_low_bias=True, 
                                          block_size=1000, 
                                          n_jobs=1, 
                                          verbose=False)
    mat = ts_con.get_data(output='dense').squeeze();
    return __2matf(mat)


# ## Param

# In[6]:


uptri_idxs = np.triu_indices(68, k=1);


# In[ ]:





# ## Load data

# In[7]:


# A fun to load emp FC
fc_root = RES_ROOT/"emp_fcs"
def _get_emp_fc(sub_ix, bd):
    fil = list(fc_root.rglob(f"*{bd}*/sub{sub_ix}.pkl"))[0]
    return np.abs(load_pkl(fil, verbose=False))


# In[8]:


# load SC and dmat, and do some preprocessing
# remove the following idxs in SC and dmat
rm_idxs = [68, 76, 77, 85]

# SC
ind_conn_xr = xr.open_dataarray(DATA_ROOT/'individual_connectomes_reordered.nc')
ind_conn = ind_conn_xr.values;

scs = []
for cur_ind_idx in range(36):
    # create spectrome brain:
    brain = Brain.Brain()
    brain.add_connectome(DATA_ROOT) # grabs distance matrix
    # re-ordering for DK atlas and normalizing the connectomes:
    brain.reorder_connectome(brain.connectome, brain.distance_matrix)
    brain.connectome =  ind_conn[:, :, cur_ind_idx] # re-assign connectome to individual connectome
    brain.bi_symmetric_c()
    brain.reduce_extreme_dir()
    sc = brain.reducedConnectome
    scs.append(sc[:, :])
scs = np.array(scs)
dmat = brain.distance_matrix[:, :];

scs = np.delete(scs, rm_idxs, axis=1)
scs = np.delete(scs, rm_idxs, axis=2)
# based on the paper, we should normalize it (Cakan_et_al_CC_2023_neurolib_pythonpkg_NMM.pdf)
scs_norm = scs/scs.max(axis=(1, 2), keepdims=1);
dmat = np.delete(dmat, rm_idxs, axis=0)
dmat = np.delete(dmat, rm_idxs, axis=1);


# In[ ]:





# # NMM

# In[9]:


# the parameters space, from 
# https://neurolib-dev.github.io/examples/example-3-meg-functional-connectivity/#model-fit
pars = ParameterSpace(['K_gl', 'exc_ext', 'inh_ext', 'sigma_ou'], 
                      [[0.0, 20.0], [0.0, 4.0], [0.0, 4.0], [0.001, 0.5]]);


# In[10]:


# the core fn to run optimization
def evaluate_simulation(traj):
    rid = traj.id
    model = evolution.getModelFromTraj(traj)

    # -------- simulation --------

    model.run()

    # -------- fitness evaluation here --------
    
    simulated_data = _model2data(model);
    
    simulated_fc = _cur_get_simu_fc(simulated_data);
    simulated_vec = _minmax_vec(simulated_fc[uptri_idxs]);
    emp_vec = _minmax_vec(cur_fc[uptri_idxs]);
    score = np.mean((simulated_vec-emp_vec)**2)
    # the output
    results = {
        "simulated_fc": simulated_fc
    }

    return (score,), results


# ## Run

# In[ ]:


cur_ind_idx = 1
cur_bd = args.band
for cur_ind_idx in range(36):
    save_dir = RES_ROOT/f"NMM_{cur_bd}_results"
    if not save_dir.exists():
        save_dir.mkdir()
        
    print(f"It is subject {cur_ind_idx:.0f} of the {cur_bd} band!")
    save_name = f"ind{cur_ind_idx}.dill";
    _cur_get_simu_fc = partial(_get_simu_fc, paras={"bd":cur_bd})
    cur_fc = _get_emp_fc(cur_ind_idx, cur_bd)
    cur_sc = scs_norm[cur_ind_idx]
    
    cur_model = _get_wc(cur_sc, dmat)
    evolution = Evolution(evaluate_simulation, pars, 
                          algorithm = 'nsga2', 
                          weightList = [-1.0], model = cur_model, 
                          POP_INIT_SIZE=128, POP_SIZE=64, NGEN=16, 
                          filename=save_dir/f"ind{cur_ind_idx}_res.hdf")
    
    if not (save_dir/save_name).exists():
        evolution.run(verbose=True, verbose_plotting=False)
        evolution.saveEvolution(save_dir/save_name);
    else:
        evolution = evolution.loadEvolution(save_dir/save_name);


# 

# ## Analysis

# In[22]:


if False:
    evolution.dfPop(outputs=True)


# In[ ]:




