# this file contains parameters for implementing FC-SBI-SGM
from easydict import EasyDict as edict
import numpy as np


paras_cons = edict()
paras_cons.delta = [2, 3.5]
paras_cons.theta = [4, 7]
paras_cons.alpha = [8, 12]
paras_cons.beta = [13, 20]

# Parameter bounds for optimization
v_lower = 3.5-1.8
v_upper = 3.5+1.8
bnds = ((0.005,0.030), (0.005,0.2), (0.005,0.030), (v_lower,v_upper), (0.1,1.0), (0.5,10.0), (0.5,10.0))
#taue,taui,tauG,speed,alpha,gii,gei

paras_cons.fs = 600
paras_cons.num_nodes = 86 # Number of cortical (68) + subcortical nodes
paras_cons.par_low =  np.array([ix[0] for ix in bnds])
paras_cons.par_high = np.array([ix[1] for ix in bnds])
paras_cons.prior_bds = np.array([paras.par_low, paras.par_high]).T

