import numpy as np


def lin_R_fn(x, y):
    """
    For both torch and np
    Calculate the linear correlation coefficient (Lin's R) between x and y.
    
    Args:
    x: torch.Tensor, shape (batch_size, num_features)/(num_feature)
    y: torch.Tensor, shape (batch_size, num_features)/(num_feature)
    
    Returns:
    ccc: torch.Tensor, shape (batch_size,)
    """
    assert x.shape == y.shape, "x and y should have the same shape"
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
    x_bar = x.mean(axis=-1, keepdims=True)
    y_bar = y.mean(axis=-1, keepdims=True)
    num = 2*((x-x_bar)*(y-y_bar)).sum(axis=-1);
    den = (x**2).sum(axis=-1) + (y**2).sum(axis=-1) - (2 * x.shape[-1] * x_bar * y_bar).squeeze()
    ccc = num/den;
    return ccc


def reg_R_fn(x, y):
    """Calculate pearons'r in batch, for both numpy and torch
    Args:
    x: torch.Tensor, shape (batch_size, num_features)/(num_feature)
    y: torch.Tensor, shape (batch_size, num_features)/(num_feature)
    Returns:
    corrs: torch.Tensor, shape (batch_size,)
    """
    assert x.shape == y.shape, "x and y should have the same shape"
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
    x_mean = x.mean(axis=-1, keepdims=True)
    y_mean = y.mean(axis=-1, keepdims=True)
    num = ((x- x_mean)*(y-y_mean)).sum(axis=-1)
    den = np.sqrt(((x- x_mean)**2).sum(axis=-1)*((y-y_mean)**2).sum(axis=-1))
    corrs = num/den
    return corrs

def mat_power(mat, pv=-1):
    """mat is a symmetric positive-semi-definite matrix
    """
    eps = 1e-10
    
    # p-inv
    S, U = np.linalg.eig(mat)
    #assert np.sum(S[np.abs(S)>eps] <0) == 0, "mat should be PSD mat"
    
    S = np.abs(S) # S >=0 for PSD
    S1 = S.copy()
    S1[S<=eps] = 0
    S1[S>eps] = (S[S>eps])**(pv)
    
    mat_r = U @ np.diag(S1) @ U.T
    return mat_r

def geodesic_dist(Q1, Q2):
    """Calculate the geodesic distance between two sys-PSD matrices. 
       Strictly, Q1 and Q2 should be invertible, but I let it can be PSD
       Follows https://github.com/makto-toruk/FC_geodesic/blob/master/utils/distance_FC/distance_FC.py
    """
    eps = 1e-10
    Q1_neg_half = mat_power(Q1, -1/2)
    Q = Q1_neg_half @ Q2 @ Q1_neg_half
    eigvs, _ = np.linalg.eig(Q)
    #assert np.sum(eigvs[np.abs(eigvs)>eps] <0) == 0, "mat should be PSD mat"
    
    eigvs = np.abs(eigvs) # Q is PSD
    eigvs_part = eigvs[eigvs>eps]
    dist = np.sqrt(np.sum(np.log(eigvs_part)**2))
    return dist

def lap_mat_fn(MapC, version=2):
    """
    For version = 2 the resulting matrix may not be exactly symmetric even if
    MapC is symmetric, thus the correction A = (A+A.T)/2

    If version not specified, the program selects the normalized L
    """

    degC = np.diag(np.sum(MapC, axis=1))  # Connectivity strength matrix

    # Unnormalized Laplacian, aka “generator matrix”:
    if version == 1:
        A = degC - MapC
    elif version == 2:
        A = degC - MapC
        # Normalized Laplacian matrix:
        iDelta = np.diag(np.sqrt(1. / (np.diag(degC) + np.finfo(float).eps)))
        A = iDelta @ A @ iDelta
        # If MapC is symmetric then so is A
        if np.array_equal(MapC, MapC.T):
            A = (A + A.T) / 2
    elif version == 3:
        A = np.eye(MapC.shape[0]) - np.linalg.inv(degC) @ MapC

    return A