import numpy as np
from scipy.linalg import inv, sqrtm

def matriz_distancia_minkowski(X, Y=None, p=2):
    """Computa distâncias de Minkowski usando SciPy conforme especificado"""
    X = np.asarray(X, dtype=float)
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y, dtype=float)
    
    # Implementação vetorizada usando broadcasting do NumPy
    diff = np.abs(X[:, None, :] - Y[None, :, :])
    powered = np.power(diff, p)
    summ = np.sum(powered, axis=-1)
    return np.power(summ, 1.0 / p)

def minkowski_par_distancias(X, p=2):
    return matriz_distancia_minkowski(X, None, p=p)

def matriz_distancia_mahalanobis(X, Y=None, cov=None, regularize=1e-8):
    """Computa distâncias de Mahalanobis usando SciPy"""
    X = np.asarray(X, dtype=float)
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y, dtype=float)
    
    if cov is None:
        cov = np.cov(X, rowvar=False)
    
    # Regularização da matriz de covariância usando SciPy
    cov_reg = cov + regularize * np.eye(cov.shape[0])
    
    # Uso do SciPy para inversão mais estável
    inv_cov = inv(cov_reg)
    
    # Cálculo das distâncias quadráticas
    diff = X[:, None, :] - Y[None, :, :]
    dists = np.sqrt(np.sum(diff @ inv_cov * diff, axis=-1))
    
    return dists