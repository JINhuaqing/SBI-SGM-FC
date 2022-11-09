""" 
Reference: "Spectral graph theory of brain oscillations--revisited and improved"

The code computes the frequency domain functional network (normalized) at a given frequency. 
To node, the code does not include the P(\omega) noise matrix, will be added in a future version.
"""

import numpy as np
from ypstruct import *    # Necesssary for using the Matlab-like struct, e.g. myflags.x
from scipy import signal 
from scipy import fftpack
from scipy import stats
# import mne
# import mne_connectivity
# from mne_connectivity import envelope_correlation

# from .runforward import run_local_coupling_forward
# from spectrome.forward import runforward
from spectrome.forward import network_transfer_macrostable as nt

# For Matlab struct use dict, e.g. p.x --> p['x']

def build_fc_freq( brain, params , freqrange ):

    """
    Input:
    
    brain: brain model
    params: brain parameters
    freqrange: a struct containing the frequency range (bandwidth) of interest, in Hz, with 
    ranges alpha, beta, delta, theta, gamma.

    Output:

    estFC, the normalized estimated FC at the given frequency computed as the mean of the range given in freqrange.
    """
    # numAxis = 2 # Axis along which to compute time series

    # Check the dimensionality of mod_fq_resp below, size KxK where K is network number of nodes
    # 3D, see earlier
    # mod_fq_resp, _, _, _ = runforward.run_local_coupling_forward( brain, params, freqs );

    minfreq = freqrange.minfreq
    maxfreq = freqrange.maxfreq

    mean_freq = (minfreq + maxfreq)/2
    model_out, _, _, _ = nt.network_transfer_local_alpha( brain , params, mean_freq )

    # No noise matrix P(\omega) explicitly used here
    estFC = np.matmul( model_out , np.matrix.getH(model_out) )

    # Now normalize estFC
    diagFC = np.diag(estFC)
    diagFC = 1./np.sqrt(diagFC)
    D = np.diag( diagFC )
    estFC = np.matmul( D , estFC )
    estFC = np.matmul( estFC , np.matrix.getH(D) )
    estFC = estFC - np.diag(np.diag( estFC ))

    # White Gaussian noise N(0,stdn) -- P(\omega) in eq 21 - to convolve with timeSeries
    #while myflags.noiseTrue:
    #    convNoise = stdn * np.random.randn( np.shape(timeSeries) )
    #    timeSeries = signal.fftconvolve( timeSeries , convNoise , axis = numAxis )

    # Find Hilbert transform of timeSeries, compute envelope
    #analytic_signal = signal.hilbert( timeSeries , axis = numAxis )
    #env_amp = np.abs( analytic_signal )

    # Compute Pearson FC from the Hilbert envelope
    #modelFC , pval = stats.pearsonr( env_amp , env_amp ) # evaluates 1d arrays only

    # Row-wise (each row contains a time series) Pearson correlation coefficients
    # rowvar: bool, optional
    # If rowvar is True (default), then each row represents a variable, with observations in the columns. 

    return estFC
