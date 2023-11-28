#!/usr/bin/env python
# coding: utf-8

# RUN SBI-SGM in alpha, new bounds, new SGM, only three parameters needed
# 
# parameters order is  :tauG,speed,alpha (In second)
# 
# Use Annealing
# 

# ## Import some pkgs

# In[1]:

print("Runing my script")

import sys
sys.path.append("../mypkg")

import scipy
import itertools

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import trange, tqdm
from scipy.io import loadmat
from functools import partial
from easydict import EasyDict as edict
from scipy.optimize import dual_annealing


# In[2]:


# my own fns
from brain import Brain
from FC_utils import build_fc_freq_m
from constants import RES_ROOT, DATA_ROOT
from utils.misc import load_pkl, save_pkl
from utils.reparam import theta_raw_2out, logistic_np
from utils.measures import reg_R_fn, lin_R_fn
from joblib import Parallel, delayed




import argparse
# In[ ]:
parser = argparse.ArgumentParser(description='RUN ANN')
parser.add_argument('--include_beta', action="store_true", help='Whether including betal band or not')
parser.add_argument('--nepoch', default=100, type=int, help='Emp FC epoch')
args = parser.parse_args()
# ## Some fns

# In[4]:


_minmax_vec = lambda x: (x-np.min(x))/(np.max(x)-np.min(x));
# transfer vec to a sym mat
def vec_2mat(vec):
    mat = np.zeros((68, 68))
    mat[np.triu_indices(68, k = 1)] = vec
    mat = mat + mat.T
    return mat


# In[ ]:





# ### Some parameters

# In[5]:


# SC
ind_conn_xr = xr.open_dataarray(DATA_ROOT/'individual_connectomes_reordered.nc')
ind_conn = ind_conn_xr.values

# PSD
ind_psd_xr = xr.open_dataarray(DATA_ROOT/'individual_psd_reordered_matlab.nc')
ind_psd = ind_psd_xr.values;


# In[6]:


_paras = edict()
_paras.delta = [2, 3.5]
_paras.theta = [4, 7]
_paras.alpha = [8, 12]
_paras.beta_l = [13, 20]


# In[48]:


paras = edict()

paras.save_prefix = "rawfc2allbd"
if args.include_beta:
    paras.bands = ["delta", "theta", "alpha", "beta_l"]
else:
    paras.bands = ["delta", "theta", "alpha"]
paras.nepoch = args.nepoch
paras.freqranges =  [np.linspace(_paras[band ][0], _paras[band][1], 5) 
                     for band in paras.bands]
print(paras.bands)
#paras.par_low = np.asarray([0.005,0.005,0.005,5, 0.1,0.001,0.001])
#paras.par_high = np.asarray([0.03, 0.20, 0.03,20,  1,    2,  0.7])
#paras.names = ["Taue", "Taui", "TauC", "Speed", "alpha", "gii", "gei"]
paras.par_low = np.asarray([0.005, 5, 0.1])
paras.par_high = np.asarray([0.03, 20, 1])
paras.names = ["TauC", "Speed", "alpha"]
paras.prior_bds = np.array([paras.par_low, paras.par_high]).T
paras.add_v = 0.01
paras.k = 1
if len(paras.bands) == 4:
    paras.ws = [1, 1, 1, 1]
elif len(paras.bands) == 3:
    paras.ws = [1, 1, 1]

paras.bounds = [
    (-10, 10), 
    (-10, 10), 
    (-10, 10), 
]


# In[49]:


# fn for reparemetering
_map_fn_np = partial(logistic_np, k=paras.k)
_theta_raw_2out = partial(theta_raw_2out, map_fn=partial(logistic_np, k=paras.k), prior_bds=paras.prior_bds);


# In[ ]:





# ### Load the data

# In[30]:



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


# In[31]:


# em FC
fc_root = RES_ROOT/"emp_fcs2"
fcss = []
for band in paras.bands:
    def _get_fc(sub_ix, bd):
        fil = list(fc_root.rglob(f"*{bd}*{paras.nepoch}/sub{sub_ix}.pkl"))[0]
        return load_pkl(fil, verbose=False)
    
    fcs = np.array([_get_fc(sub_ix, band) for sub_ix in range(36)]);
    fcss.append(fcs)


# In[ ]:





# ## Annealing

# In[32]:


def simulator(raw_params, brain, prior_bds, freqranges):
    params = _map_fn_np(raw_params)*(prior_bds[:, 1]-prior_bds[:, 0]) + prior_bds[:, 0]
    
    params_dict = dict()
    params_dict["tauC"] =  params[0]
    params_dict["speed"] =  params[1]
    params_dict["alpha"] =  params[2]
    
    ress = []
    modelFCs = []
    for freqrange in freqranges:
        modelFC = build_fc_freq_m(brain, params_dict, freqrange)
        modelFC_abs = np.abs(modelFC[:68, :68])
        res = _minmax_vec(modelFC_abs[np.triu_indices(68, k = 1)])
        ress.append(res)
        modelFCs.append(modelFC)
    return ress, modelFCs


# In[50]:


def _obj_fn(raw_params, empfcs, simulator_sp, ws):
    emp_ress = []
    for empfc in empfcs:
        empfc = np.abs(empfc)
        emp_res = _minmax_vec(empfc[np.triu_indices(68, k = 1)])
        emp_ress.append(emp_res)
    simu_ress = simulator_sp(raw_params)[0] # it is after minmax
    rvs = [-lin_R_fn(simu_res, emp_res)[0] 
          for simu_res, emp_res in zip(simu_ress, emp_ress)]
    #print(np.round(rvs, 3), np.round(np.average(rvs, weights=ws), 3), np.round(raw_params, 3))
    return np.average(rvs, weights=ws)


# In[51]:


brains = []
for sub_idx in range(36):
    brain = Brain.Brain()
    brain.add_connectome(DATA_ROOT) # grabs distance matrix
    # re-ordering for DK atlas and normalizing the connectomes:
    brain.reorder_connectome(brain.connectome, brain.distance_matrix)
    brain.connectome =  ind_conn[:, :, sub_idx] # re-assign connectome to individual connectome
    brain.bi_symmetric_c()
    brain.reduce_extreme_dir()
    brains.append(brain)
    


# In[52]:


def _run_fn(sub_idx):
    # brain
    brain = brains[sub_idx]
    # empfc
    empfcs = [fcs[sub_idx] for fcs in fcss]
    
    simulator_sp = partial(simulator, 
                           brain=brain, 
                           prior_bds=paras.prior_bds, 
                           freqranges=paras.freqranges)
    res = dual_annealing(_obj_fn, 
                         x0=np.array([0, 0, 0]),
                         bounds=paras.bounds, 
                         args=(empfcs, simulator_sp, paras.ws), 
                         maxiter=200,
                         initial_temp=5230.0,
                         seed=24,
                         visit=2.62,
                         no_local_search=False)
    save_res = edict()
    save_res.bestfc = simulator_sp(res.x)[1]
    save_res.ann_res = res
    
    save_fil = f"{paras.save_prefix}_ANN_{'-'.join(paras.bands)}_ep{paras.nepoch}_" +                    f"addv{paras.add_v*100:.0f}" +                   f"/ind{sub_idx}.pkl"
    save_pkl(RES_ROOT/save_fil, save_res)
    return save_res


# In[16]:


with Parallel(n_jobs=36) as parallel:
     _ = parallel(delayed(_run_fn)(sub_idx)  
                  for sub_idx in tqdm(range(36), total=36))


# In[ ]:





