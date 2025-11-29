import numpy as np


def gonzalez_kcenters(X, k, dist_matrix=None):
    """Algoritmo farthest-first (Gonzalez).
    Retorna índices dos centros e o raio da solução.
    Se dist_matrix for fornecida (NxN), usa-a; caso contrário pressupõe distância Euclidiana.
    """
    n = X.shape[0]
    if k <= 0:
        raise ValueError('k deve ser >= 1')
    centers = []
    centers.append(0)
    if dist_matrix is None:
        from kcenters.distances import minkowski_par_distancias
        D = minkowski_par_distancias(X, p=2)
    else:
        D = dist_matrix
    min_dists = D[centers[0], :].copy()
    for _ in range(1, k):
        idx = int(np.argmax(min_dists))
        centers.append(idx)
        min_dists = np.minimum(min_dists, D[idx, :])
    radius = float(np.max(min_dists))
    return centers, radius


def viavel_com_raio_guloso(D, k, R):
    """Procedimento de decisão: é possível cobrir todos os pontos com k centros de raio R?
    Estratégia gulosa: escolher um ponto descoberto, colocar centro nele e marcar como cobertos
    todos os pontos dentro de 2R.
    D: matriz de distâncias completa NxN
    """
    n = D.shape[0]
    uncovered = np.ones(n, dtype=bool)
    centers_count = 0
    while np.any(uncovered):
        if centers_count >= k:
            return False
        i = np.argmax(uncovered)
        centers_count += 1
        covered_by_i = D[i, :] <= 2.0 * R + 1e-12
        uncovered[covered_by_i] = False
    return True


def refinamento_intervalo_kcenters(D, k, eps_fraction=0.05, max_iters=60):
    """Algoritmo de refinamento de intervalo sobre o raio R.
    - D: matriz de distâncias
    - eps_fraction: parar quando (high - low) <= eps_fraction * largura_inicial
    Retorna (centers_indices, raio_real)
    """
    n = D.shape[0]
    low = 0.0
    high = float(np.max(D))
    initial_width = high - low
    if initial_width == 0.0:
        return [0], 0.0
    it = 0
    while it < max_iters and (high - low) > eps_fraction * initial_width:
        mid = 0.5 * (low + high)
        feasible = viavel_com_raio_guloso(D, k, mid)
        if feasible:
            high = mid
        else:
            low = mid
        it += 1
    R = high
    uncovered = np.ones(n, dtype=bool)
    centers = []
    while np.any(uncovered):
        i = int(np.argmax(uncovered))
        centers.append(i)
        to_remove = D[i, :] <= 2.0 * R + 1e-12
        uncovered[to_remove] = False
    min_dists = np.min(D[np.ix_(centers, range(n))], axis=0)
    radius_actual = float(np.max(min_dists))
    return centers, radius_actual
