"""
Distâncias e k-center 2-aproximado.
"""

import numpy as np
from scipy import linalg
import time


def pairwise_minkowski(X, p=2):
    X = np.asarray(X)
    if p == 2:
        # versão otimizada para p=2 (produto escalar): evita calcular diferenças
        sq = np.sum(X**2, axis=1, keepdims=True)
        D2 = sq + sq.T - 2 * (X @ X.T)
        # numéricamente, pequenas negativas podem aparecer; truncar
        D2 = np.maximum(D2, 0.0)
        return np.sqrt(D2)
    else:
        # Para p diferentes de 2 usamos broadcasting para criar a matriz
        # de diferenças e em seguida aplicamos a norma ao longo da dimensão
        # das features.
        dif = X[:, None, :] - X[None, :, :]
        return np.sum(np.abs(dif) ** p, axis=2) ** (1.0 / p)

def covariance_inverse(X, regularize=1e-8):
    X = np.asarray(X)
    # Usamos ddof=0 (máxima verossimilhança) para covariância da amostra; ajustar se for preciso
    cov = np.cov(X, rowvar=False, bias=True)
    # regularização para evitar singularidade
    cov += np.eye(cov.shape[0]) * regularize
    inv = linalg.inv(cov)
    return inv

def pairwise_mahalanobis(X, VI=None):
    X = np.asarray(X)
    if VI is None:
        VI = covariance_inverse(X)
    # Vamos calcular usando forma quadrática: d^2 = (x-y)^T VI (x-y)
    # Expandindo: d^2 = x^T VI x + y^T VI y - 2 x^T VI y
    XV = X @ VI
    q = np.sum(XV * X, axis=1, keepdims=True)  # x^T VI x for each x
    D2 = q + q.T - 2 * (XV @ X.T)
    D2 = np.maximum(D2, 0.0)
    return np.sqrt(D2)


def k_center_farthest_first(dist_matrix, k, random_seed=None):
    n = dist_matrix.shape[0]
    rng = np.random.default_rng(random_seed)
    # escolher primeiro centro (aleatório se fornecemos semente)
    first = int(rng.integers(0, n)) if random_seed is not None else 0
    centers = [first]
    # distâncias até o centro mais próximo (atual)
    dist_to_center = dist_matrix[first].copy()
    for _ in range(1, k):
        # escolher o ponto que maximiza dist to nearest center
        idx = int(np.argmax(dist_to_center))
        centers.append(idx)
        # atualizar dist_to_center
        dist_to_center = np.minimum(dist_to_center, dist_matrix[idx])
    # garantir que os centros são únicos e manter ordem
    centers = list(dict.fromkeys(centers))
    assignments = np.argmin(dist_matrix[:, centers], axis=1)
    # calcular o raio: maior distância entre ponto e seu centro
    radius = float(np.max([dist_matrix[i, centers[assignments[i]]] for i in range(n)]))
    return centers, assignments, radius

def k_center_decision(dist_matrix, k, r):
    n = dist_matrix.shape[0]
    uncovered = np.ones(n, dtype=bool)
    centers = []
    for _ in range(k):
        # se todos cobertos, retorna True
        if not uncovered.any():
            return True, centers
        # pick any uncovered point (first)
        i = np.argmax(uncovered)  # index of first True (works because uncovered is bool)
        centers.append(i)
        # marcar pontos dentro de r como cobertos
        within = dist_matrix[i] <= r + 1e-12
        uncovered = uncovered & (~within)
    return (not uncovered.any()), centers

def k_center_interval_refinement(dist_matrix, k, width_fraction=0.01, max_iter=60):
    n = dist_matrix.shape[0]
    # bounds
    L0 = 0.0
    R0 = float(np.max(dist_matrix))
    L, R = L0, R0
    iter_count = 0
    last_centers = None
    while (R - L) > width_fraction * (R0 - L0) and iter_count < max_iter:
        mid = (L + R) / 2.0
        feasible, centers = k_center_decision(dist_matrix, k, mid)
        if feasible:
            R = mid
            last_centers = centers
        else:
            L = mid
        iter_count += 1
    # If no feasible found during search, run decision on R to produce centers
    if last_centers is None:
        feasible, last_centers = k_center_decision(dist_matrix, k, R)
    centers = list(dict.fromkeys(last_centers))  # unique
    if len(centers) == 0:
        # fallback: use farthest-first to get k centers
        centers, assignments, radius = k_center_farthest_first(dist_matrix, k)
        return centers, assignments, radius
    # compute assignments and radius
    assignments = np.argmin(dist_matrix[:, centers], axis=1)
    radius = float(np.max([dist_matrix[i, centers[assignments[i]]] for i in range(n)]))
    return centers, assignments, radius


def assignments_and_radius_from_centers(dist_matrix, centers):
    assignments = np.argmin(dist_matrix[:, centers], axis=1)
    radius = float(np.max([dist_matrix[i, centers[assignments[i]]] for i in range(dist_matrix.shape[0])]))
    return assignments, radius


def demo_test():
    # cria três grupos em 2D só para ver a saída das funções
    rng = np.random.default_rng(42)
    C1 = rng.normal(loc=[0,0], scale=0.3, size=(50,2))
    C2 = rng.normal(loc=[3,0], scale=0.5, size=(50,2))
    C3 = rng.normal(loc=[0,4], scale=0.7, size=(50,2))
    X = np.vstack([C1, C2, C3])
    k = 3

    print("Dados: N =", X.shape[0], "dim =", X.shape[1])

    # 1) calcular distâncias euclidianas
    t0 = time.time()
    D_euc = pairwise_minkowski(X, p=2)
    t1 = time.time()
    print(f"Distância euclidiana calculada em {t1-t0:.4f}s. max dist = {D_euc.max():.4f}")

    # 2) distância de Mahalanobis (leva em conta covariância)
    t0 = time.time()
    VI = covariance_inverse(X)
    D_mah = pairwise_mahalanobis(X, VI=VI)
    t1 = time.time()
    print(f"Distância Mahalanobis calculada em {t1-t0:.4f}s. max dist = {D_mah.max():.4f}")

    # 3) farthest-first (algoritmo guloso)
    centers_ff, assign_ff, radius_ff = k_center_farthest_first(D_euc, k, random_seed=0)
    print("Farthest-first centers indices:", centers_ff, "radius:", radius_ff)

    # 4) refinamento por intervalo (busca binária sobre o raio)
    centers_int, assign_int, radius_int = k_center_interval_refinement(D_euc, k, width_fraction=0.01)
    print("Interval refinement centers indices:", centers_int, "radius:", radius_int)

    # 5) checar consistência dos assignments calculando o raio direto
    asas, rr = assignments_and_radius_from_centers(D_euc, centers_ff)
    print("Check radius from centers (farthest-first):", rr)
    asas2, rr2 = assignments_and_radius_from_centers(D_euc, centers_int)
    print("Check radius from centers (interval):", rr2)

    return {
        "X": X,
        "D_euc": D_euc,
        "D_mah": D_mah,
        "ff": (centers_ff, assign_ff, radius_ff),
        "int": (centers_int, assign_int, radius_int)
    }

# Executa demo
results = demo_test()

def save_module(path=None):
    import inspect, os
    if path is None:
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tp2_part1_impl_saved.py'))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# Arquivo gerado automaticamente contendo implementações da Parte 1\n")
        f.write(inspect.getsource(pairwise_minkowski) + "\n\n")
        f.write(inspect.getsource(covariance_inverse) + "\n\n")
        f.write(inspect.getsource(pairwise_mahalanobis) + "\n\n")
        f.write(inspect.getsource(k_center_farthest_first) + "\n\n")
        f.write(inspect.getsource(k_center_decision) + "\n\n")
        f.write(inspect.getsource(k_center_interval_refinement) + "\n\n")
        f.write(inspect.getsource(assignments_and_radius_from_centers) + "\n\n")
    print(f"Arquivo com implementações salvo em: {path}")

print("Execução concluída.")

