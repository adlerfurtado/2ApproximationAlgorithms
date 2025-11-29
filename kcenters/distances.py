import numpy as np

def matriz_distancia_minkowski(X, Y=None, p=2):
    """Computa distâncias de Minkowski par-a-par entre as linhas de X e Y.
    Se Y for None, computa a matriz de distâncias de X contra X (NxN).
    Implementado de forma vetorizada com NumPy.
    """
    X = np.asarray(X, dtype=float)
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y, dtype=float)
    diff = np.abs(X[:, None, :] - Y[None, :, :])
    powered = diff ** p
    summ = np.sum(powered, axis=-1)
    return summ ** (1.0 / p)


def minkowski_par_distancias(X, p=2):
    return matriz_distancia_minkowski(X, None, p=p)


def matriz_distancia_mahalanobis(X, Y=None, cov=None, regularize=1e-8):
    """Computa distâncias de Mahalanobis entre pares.
    Se cov for None, estima a matriz de covariância de X.
    Implementado usando produtos matriciais do NumPy.
    """
    X = np.asarray(X, dtype=float)
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y, dtype=float)
    if cov is None:
        cov = np.cov(X, rowvar=False)
    cov = cov + regularize * np.eye(cov.shape[0])
    invcov = np.linalg.inv(cov)
    XI = X @ invcov
    xs = np.sum(XI * X, axis=1)
    YI = Y @ invcov
    ys = np.sum(YI * Y, axis=1)
    cross = XI @ Y.T
    d2 = xs[:, None] + ys[None, :] - 2.0 * cross
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2)