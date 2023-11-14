# New way to estimate FC (on May 28, 2023)
# 
"""Module for computing basic quantities from a spectral graph model: the forward model
Makes the calculation for a single frequency only. 
Calculate SGM, but only fit on TauG, alpha, speed
"""

import numpy as np
def network_transfer_local_fc_alpha(brain, parameters, w, is_full=False):
    """Network Transfer Function for spectral graph model.

    Args:
        brain (Brain): specific brain to calculate NTF
        parameters (dict): parameters for ntf. We shall keep this separate from Brain
               for now, as we want to change and update according to fitting.
        w (float): frequency at which to calculate NTF, in angular freq not in Hz (on May 28, 2023)

    Returns:
        fc(numpy asarray):  The FC for the given frequency (w)
    """
    
    C = brain.reducedConnectome
    D = brain.distance_matrix

    speed = parameters["speed"]
    tauC = parameters["tauC"]
    alpha = parameters["alpha"]
    
    # Defining some other parameters used:
    zero_thr = 0.01

    # define sum of degrees for rows and columns for laplacian normalization
    rowdegree = np.transpose(np.sum(C, axis=1))
    coldegree = np.sum(C, axis=0)
    qind = rowdegree + coldegree < 0.2 * np.mean(rowdegree + coldegree)
    rowdegree[qind] = np.inf
    coldegree[qind] = np.inf

    nroi = C.shape[0]
    K = nroi

    Tau = 0.001 * D / speed
    Cc = C * np.exp(-1j * Tau * w)

    # Eigen Decomposition of Complex Laplacian Here
    L1 = np.identity(nroi)
    L2 = np.divide(1, np.sqrt(np.multiply(rowdegree, coldegree)) + np.spacing(1))
    L = L1 - alpha * np.matmul(np.diag(L2), Cc)

    d, v = np.linalg.eig(L)  
    eig_ind = np.argsort(np.abs(d))  # sorting in ascending order and absolute value
    eig_vec = v[:, eig_ind]  # re-indexing eigen vectors according to sorted index
    eig_val = d[eig_ind]  # re-indexing eigen values with same sorted index

    eigenvalues = np.transpose(eig_val[:K])
    eigenvectors = eig_vec[:, :K]

    # Cortical model
    FG = np.divide(1 / tauC ** 2, (1j * w + 1 / tauC) ** 2)


    q1 = (1j * w + 1 / tauC * FG * eigenvalues)
    qthr = zero_thr * np.abs(q1[:]).max()
    magq1 = np.maximum(np.abs(q1), qthr)
    angq1 = np.angle(q1)
    q1 = np.multiply(magq1, np.exp(1j * angq1))
    frequency_response = np.divide(1, np.abs(q1)**2)
    
    fc = eigenvectors @ np.diag(frequency_response) @ np.conjugate(eigenvectors.T)
    fc = np.abs(fc)

    if is_full:
        return fc, eigenvectors, frequency_response, eigenvalues
    else:
        return fc
