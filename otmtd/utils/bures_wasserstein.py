import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn
import numpy as np

import ot
from tqdm import tqdm
import itertools
import logging


# -------Code modified from OTDD (https://github.com/microsoft/otdd)--------
# Via Newton-Schulz: based on
# https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py, and
# https://github.com/BorisMuzellec/EllipticalEmbeddings/blob/master/utils.py
def sqrtm_newton_schulz(A, numIters, reg=None, return_error=False, return_inverse=False):
    """ Matrix squareroot based on Newton-Schulz method """
    if A.ndim <= 2:  # Non-batched mode
        A = A.unsqueeze(0)
        batched = False
    else:
        batched = True
    batchSize = A.shape[0]
    dim = A.shape[1]
    # Slightly faster than : A.mul(A).sum((-2,-1)).sqrt()
    normA = (A**2).sum((-2, -1)).sqrt()

    if reg:
        # Renormalize so that the each matrix has a norm lesser than 1/reg,
        # but only normalize when necessary
        normA *= reg
        renorm = torch.ones_like(normA)
        renorm[torch.where(normA > 1.0)] = normA[cp.where(normA > 1.0)]
    else:
        renorm = normA

    Y = A.div(renorm.view(batchSize, 1, 1).expand_as(A))
    I = torch.eye(dim, dim).view(1, dim, dim).repeat(
        batchSize, 1, 1).to(A.device)  # .type(dtype)
    Z = torch.eye(dim, dim).view(1, dim, dim).repeat(
        batchSize, 1, 1).to(A.device)  # .type(dtype)
    for i in range(numIters):
        T = 0.5*(3.0*I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    sA = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    sAinv = Z/torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    if not batched:
        sA = sA[0, :, :]
        sAinv = sAinv[0, :, :]

    if not return_inverse and not return_error:
        return sA
    elif not return_inverse and return_error:
        return sA, compute_error(A, sA)
    elif return_inverse and not return_error:
        return sA, sAinv
    else:
        return sA, sAinv, compute_error(A, sA)


def compute_error(A, sA):
    """ Computes error in approximation """
    normA = torch.sqrt(torch.sum(torch.sum(A * A, dim=1), dim=1))
    error = A - torch.bmm(sA, sA)
    error = torch.sqrt((error * error).sum(dim=1).sum(dim=1)) / normA
    return torch.mean(error)


def cov_bures_distance(Σ1, Σ2, sqrtΣ1=None, inv_sqrtΣ1=None,
                       diagonal_cov=False, commute=False, squared=True, sqrt_method='ns',
                       sqrt_niters=20):
    """
    Returns the second term in Bures-Wasserstein distances:
        tr(Σ1 + Σ2 - 2sqrt((sqrt(Σ1)Σ2sqrt(Σ1))))

    """
    if sqrtΣ1 is None and not diagonal_cov:
        sqrtΣ1 = sqrtm_newton_schulz(Σ1, sqrt_niters)

    if diagonal_cov:
        bures = ((torch.sqrt(Σ1) - torch.sqrt(Σ2))**2).sum(-1)
    elif commute:
        sqrtΣ2 = sqrtm_newton_schulz(Σ2, sqrt_niters)
        bures = ((sqrtΣ1 - sqrtΣ2)**2).sum((-2, -1))
    else:
        cross = sqrtm_newton_schulz(torch.matmul(torch.matmul(
            sqrtΣ1, Σ2), sqrtΣ1), sqrt_niters)
        bures = (Σ1 + Σ2 - 2 * cross).diagonal(dim1=-2, dim2=-1).sum(-1)
    if not squared:
        bures = torch.sqrt(bures)
    return torch.relu(bures)


def wasserstein_gauss_distance(μ_1, μ_2, Σ1, Σ2, sqrtΣ1=None,
                               squared=False, **kwargs):
    """
    Returns 2-Wasserstein Distance between Gaussians:
        W(α, β)^2 = || μ_α - μ_β ||^2 + Bures(Σ_α, Σ_β)^2

    Arguments:
        μ_1 (tensor): mean of first Gaussian
        kwargs (dict): additional arguments for bbures_distance.

    Returns:
        d (tensor): the Wasserstein distance

    """
    mean_diff = ((μ_1 - μ_2)**2).sum(axis=-1)
    cova_diff = cov_bures_distance(
        Σ1, Σ2, sqrtΣ1=sqrtΣ1, squared=True, **kwargs)
    d = torch.relu(mean_diff + cova_diff)
    d = mean_diff + cova_diff
    if not squared:
        d = torch.sqrt(d)
    return d


logger = logging.getLogger(__name__)


def efficient_pwdist_gauss(M1, S1, M2=None, S2=None, sqrtS1=None, sqrtS2=None,
                           symmetric=False, diagonal_cov=False, commute=False,
                           sqrt_method='ns', sqrt_niters=20, sqrt_pref=0,
                           device='cpu', nworkers=1,
                           return_dmeans=False, return_sqrts=False):
    if M2 is None:
        symmetric = True
        M2, S2 = M1, S1

    n1, n2 = len(M1), len(M2)  # Number of clusters in each side
    if symmetric:
        # If tasks are symmetric (same data on both sides) only need combinations
        pairs = list(itertools.combinations(range(n1), 2))
    else:
        # If tasks are assymetric, need n1 x n2 comparisons
        pairs = list(itertools.product(range(n1), range(n2)))

    D = torch.zeros((n1, n2), device=device, dtype=M1.dtype)

    sqrtS = []
    # Note that we need inverses of only one of two datasets.
    # If sqrtS of S1 provided, use those. If S2 provided, flip roles of covs in Bures
    both_sqrt = (sqrtS1 is not None) and (sqrtS2 is not None)
    if (both_sqrt and sqrt_pref == 0) or (sqrtS1 is not None):
        # Either both were provided and S1 (idx=0) is prefered, or only S1 provided
        flip = False
        sqrtS = sqrtS1
    elif sqrtS2 is not None:
        # S1 wasn't provided
        if sqrt_pref == 0:
            logger.warning('sqrt_pref=0 but S1 not provided!')
        flip = True
        sqrtS = sqrtS2  # S2 playes role of S1
    elif len(S1) <= len(S2):  # No precomputed squareroots provided. Compute, but choose smaller of the two!
        flip = False
        S = S1
    else:
        flip = True
        S = S2  # S2 playes role of S1

    if not sqrtS:
        logger.info('Precomputing covariance matrix square roots...')
        for i, Σ in tqdm(enumerate(S), leave=True):
            if diagonal_cov:
                assert Σ.ndim == 1
                sqrtS.append(torch.sqrt(Σ))  # This is actually not needed.
            else:
                sqrtS.append(sqrtm_newton_schulz(Σ, sqrt_niters))

    logger.info('Computing gaussian-to-gaussian wasserstein distances...')
    pbar = tqdm(pairs, leave=True)
    pbar.set_description('Computing label-to-label distances')
    for i, j in pbar:
        if not flip:
            D[i, j] = wasserstein_gauss_distance(M1[i], M2[j], S1[i], S2[j], sqrtS[i],
                                                 diagonal_cov=diagonal_cov,
                                                 commute=commute, squared=False,
                                                 sqrt_method=sqrt_method,
                                                 sqrt_niters=sqrt_niters)
        else:
            D[i, j] = wasserstein_gauss_distance(M2[j], M1[i], S2[j], S1[i], sqrtS[j],
                                                 diagonal_cov=diagonal_cov,
                                                 commute=commute, squared=False,
                                                 sqrt_method=sqrt_method,
                                                 sqrt_niters=sqrt_niters)
        if symmetric:
            D[j, i] = D[i, j]

    if return_dmeans:
        D_means = torch.cdist(M1, M2)  # For viz purposes only
        if return_sqrts:
            return D, D_means, sqrtS
        else:
            return D, D_means
    elif return_sqrts:
        return D, sqrtS
    else:
        return D
