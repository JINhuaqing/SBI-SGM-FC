# Note that I used a new way to do FC (on May 28, 2023)
# U lambda^2 U^H
import numpy as np
from forward import network_transfer_macrostable as nt

def build_fc_freq_m(brain, params , freqrange):

    """
    Input:
    
    brain: brain model
    params: brain parameters
    freqrange: a struct containing the frequency range (bandwidth) of interest, in Hz, with 
    ranges alpha, beta, delta, theta, gamma.

    Output:

    estFC, the mean normalized estimated FC at the given frequency computed 
            over the range given in freqrange.
    """
    estFC = 0
    for cur_freq in freqrange:
        w = 2 * np.pi * cur_freq
        cur_estFC = nt.network_transfer_local_fc_alpha(brain, params, w)
        estFC = cur_estFC/len(freqrange) + estFC

    # Now normalize estFC
    diagFC = np.diag(np.abs(estFC))
    diagFC = 1./np.sqrt(diagFC)
    D = np.diag( diagFC )
    estFC = np.matmul( D , estFC )
    estFC = np.matmul(estFC , np.matrix.getH(D)) # f_ij/\sqrt(f_ii)\sqrt(f_jj)
    estFC = estFC - np.diag(np.diag( estFC ))

    return estFC
