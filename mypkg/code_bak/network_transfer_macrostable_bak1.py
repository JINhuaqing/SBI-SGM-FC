"""Module for computing basic quantities from a spectral graph model: the forward model
Makes the calculation for a single frequency only. 
Calculate SGM, but only fit on TauG, alpha, speed
"""

import numpy as np
def network_transfer_local_fc_alpha(brain, parameters, w, diag_ws):
    """Network Transfer Function for spectral graph model.

    Args:
        brain (Brain): specific brain to calculate NTF
        parameters (dict): parameters for ntf. We shall keep this separate from Brain
        for now, as we want to change and update according to fitting.
        frequency (float): frequency at which to calculate NTF

    Returns:
        model_out (numpy asarray):  Each region's frequency response for
        the given frequency (w)
        frequency_response (numpy asarray):
        ev (numpy asarray): Eigen values
        Vv (numpy asarray): Eigen vectors

    """
    # remove the following idxs
    rm_idxs = [68, 76, 77, 85]
    
    C = brain.reducedConnectome
    D = brain.distance_matrix
    
    C = np.delete(C, rm_idxs, axis=0)
    C = np.delete(C, rm_idxs, axis=1)
    D = np.delete(D, rm_idxs, axis=0)
    D = np.delete(D, rm_idxs, axis=1)
    # only take 68
    #C = C[:68, :68]
    #D = D[:68, :68]

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

    eigenvalues = np.transpose(eig_val)
    eigenvectors = eig_vec[:, 0:K]

    # Cortical model
    FG = np.divide(1 / tauC ** 2, (1j * w + 1 / tauC) ** 2)


    q1 = (1j * w + 1 / tauC * FG * eigenvalues)
    qthr = zero_thr * np.abs(q1[:]).max()
    magq1 = np.maximum(np.abs(q1), qthr)
    angq1 = np.angle(q1)
    q1 = np.multiply(magq1, np.exp(1j * angq1))
    frequency_response = np.divide(diag_ws, q1)
    # no square for diag_ws
    # A new way to get FC (on May 24, 2023)
    # new exp FC
    # calculate the diag ws by my own.
    
    model_out = 0
    for k in range(K):
        model_out += (frequency_response[k]) * np.outer(eigenvectors[:, k], np.conjugate(eigenvectors[:, k]))

    # model_out2 = np.linalg.norm(model_out,axis=1)

    return model_out, frequency_response, eigenvalues, eigenvectors
