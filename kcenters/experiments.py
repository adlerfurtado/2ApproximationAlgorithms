import numpy as np
import time
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from kcenters.distances import minkowski_par_distancias, matriz_distancia_mahalanobis
from kcenters.algorithms import gonzalez_kcenters, refinamento_intervalo_kcenters


def calcular_matriz_distancias(X, metric='euclidiana', p=2, cov=None):
    if metric in ('euclidiana', 'minkowski'):
        return minkowski_par_distancias(X, p=p)
    elif metric == 'mahalanobis':
        return matriz_distancia_mahalanobis(X, None, cov=cov)
    else:
        raise ValueError('métrica desconhecida')


def rodar_experimento_unico(X, true_labels, k, metric='euclidiana', p=2, cov=None, n_runs=15, eps_values=None):
    """Roda todos os algoritmos para um único conjunto X.
    Retorna lista de dicionários de resultados.
    """
    results = []
    D = calcular_matriz_distancias(X, metric=metric, p=p, cov=cov)
    if eps_values is None:
        eps_values = [0.01, 0.05, 0.1, 0.15, 0.25]
    for run in range(n_runs):
        t0 = time.perf_counter()
        centers_idx, radius = gonzalez_kcenters(X, k, dist_matrix=D)
        t1 = time.perf_counter()
        labels = atribuir_rotulos_de_centros(D, centers_idx)
        sil = silhouette_score(X, labels) if k > 1 and len(np.unique(labels)) > 1 else float('nan')
        ari = adjusted_rand_score(true_labels, labels) if true_labels is not None else float('nan')
        results.append({'algoritmo': 'gonzalez', 'run': run, 'radius': radius, 'silhouette': sil, 'ari': ari, 'tempo': t1 - t0})
        for eps in eps_values:
            t0 = time.perf_counter()
            centers_idx_int, radius_int = refinamento_intervalo_kcenters(D, k, eps_fraction=eps)
            t1 = time.perf_counter()
            labels_int = atribuir_rotulos_de_centros(D, centers_idx_int)
            sil = silhouette_score(X, labels_int) if k > 1 and len(np.unique(labels_int)) > 1 else float('nan')
            ari = adjusted_rand_score(true_labels, labels_int) if true_labels is not None else float('nan')
            results.append({'algoritmo': 'intervalo', 'eps': eps, 'run': run, 'radius': radius_int, 'silhouette': sil, 'ari': ari, 'tempo': t1 - t0})
        t0 = time.perf_counter()
        km = KMeans(n_clusters=k, n_init=10, random_state=run)
        km.fit(X)
        t1 = time.perf_counter()
        labels_km = km.labels_
        sil = silhouette_score(X, labels_km) if k > 1 and len(np.unique(labels_km)) > 1 else float('nan')
        ari = adjusted_rand_score(true_labels, labels_km) if true_labels is not None else float('nan')
        results.append({'algoritmo': 'kmeans', 'run': run, 'radius': float('nan'), 'silhouette': sil, 'ari': ari, 'tempo': t1 - t0})
    return results


def atribuir_rotulos_de_centros(D, centers_idx):
    sub = D[np.ix_(centers_idx, range(D.shape[0]))]
    labels = np.argmin(sub, axis=0)
    return labels