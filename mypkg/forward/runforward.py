""" running the ntf over a range of frequencies.
    To note: Compared with previous version of the program, the 
    resulting model, e.g. model_out, is now normalized such that
    all diagonal elements are one (then subtracted)
"""
# from ..forward import network_transfer_macrostable as nt
from . import network_transfer_macrostable as nt
import numpy as np

def run_local_coupling_forward(brain, params, freqs):
    """run_forward. Function for running the forward model over the passed in range of frequencies,
    for the handed set of parameters (which must be passed in as a dictionary)

    Args:
        brain (Brain): An instance of the Brain class.
        params (dict): Dictionary of a setting of parameters for the NTF model.
        freqs (array): Array of freqencies for which the model is to be calculated.

    Returns:
        array: Model values for each frequency, for each region of the brain, ordered as according to HCP
        (as in Brain class ordering).

    """
    
    # eigenvalues = []
    # eigenvectors = []
    # frequency_response = []
    # model_out = []

    # 86 is the total number of nodes
    #    
    model_out = np.zeros((86,86,len(freqs)))
    count = 0 # Consider using enumerate instead

    for freq in freqs:
        w = 2 * np.pi * freq
        freq_model, _, _, _ = nt.network_transfer_local_alpha(
            brain, params, w
        )
        ## Below added for freq_model diagonal normalization purposes
        diagFC = np.diag(freq_model) # used below for the matrix normalization
        diagFC = 1./np.sqrt(diagFC)
        D = np.diag( diagFC )
        # Remove diagonal elements
        freq_model = freq_model - np.diag(np.diag(freq_model))  
        FC_hat = np.matmul( D , freq_model )
        FC_hat = np.matmul( FC_hat , np.matrix.getH(D) )
        #FC_hat = FC_hat - np.diag(np.diag( FC_hat ))
        model_out[:,:,count] = FC_hat
        count += 1
        # frequency_response.append(freq_resp)
        # eigenvalues.append(eig_val)
        # eigenvectors.append(eig_vec)
        # model_out.append(freq_model)
    
    return model_out

