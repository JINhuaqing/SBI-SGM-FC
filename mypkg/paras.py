# this file contains parameters for implementing FC-SBI-SGM
from easydict import EasyDict as edict


paras = edict()

paras.res_dir = Path("./results")

paras.fc_type = "cohy" #mag, coh
#paras.freqrange = [8, 13] # frequency rage
paras.freqrange =  fvec[np.bitwise_and(fvec>8, fvec<13)]
paras.fs = 600
paras.num_nodes = 86 # Number of cortical (68) + subcortical nodes
paras.uni_SC = False

paras.SBI_paras = edict()
paras.SBI_paras.num_prior_sps = int(1e5)
#paras.SBI_paras.density_model = "nsf"
paras.SBI_paras.density_model = "mdn"
paras.SBI_paras.noise_sd = 0.15
paras.SBI_paras.combine_loss = False
paras.SBI_paras.prefix = "" # ModSC: modified SC, MR: multiple round, None
#paras.SBI_paras.prefix = "ModSC"