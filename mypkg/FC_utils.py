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
    model_out = 0
    for cur_freq in freqrange:
        cur_model_out, _, _, _ = nt.network_transfer_local_alpha(brain , params, cur_freq)
        model_out = cur_model_out/len(freqrange) + model_out

    # No noise matrix P(\omega) explicitly used here
    estFC = np.matmul( model_out , np.matrix.getH(model_out) )

    # Now normalize estFC
    diagFC = np.diag(np.abs(estFC))
    diagFC = 1./np.sqrt(diagFC)
    D = np.diag( diagFC )
    estFC = np.matmul( D , estFC )
    estFC = np.matmul( estFC , np.matrix.getH(D) ) # f_ij/\sqrt(f_ii)\sqrt(f_jj)
    estFC = estFC - np.diag(np.diag( estFC ))

    return estFC
