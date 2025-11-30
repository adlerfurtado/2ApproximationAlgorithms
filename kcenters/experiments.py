import numpy as np
import time
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans

# Importações CORRETAS dos algoritmos
from kcenters.algorithms import gonzalez_kcenters, refinamento_intervalo_kcenters
from kcenters.distances import minkowski_par_distancias, matriz_distancia_mahalanobis


def calcular_matriz_distancias(X, metric='euclidiana', p=2, cov=None):
    """Calcula matriz de distâncias conforme métrica especificada"""
    if metric in ('euclidiana', 'minkowski'):
        return minkowski_par_distancias(X, p=p)
    elif metric == 'mahalanobis':
        return matriz_distancia_mahalanobis(X, cov=cov)
    else:
        raise ValueError(f'Métrica desconhecida: {metric}')


def calcular_raio_kmeans(X, labels):
    """Calcula raio da solução K-Means (máxima distância ao centro do cluster)"""
    labels = np.asarray(labels)
    k = len(np.unique(labels))
    max_dist = 0.0
    
    for cluster_id in range(k):
        mask = labels == cluster_id
        if np.sum(mask) == 0:
            continue
        
        cluster_points = X[mask]
        # Encontrar centro real do K-Means
        center = cluster_points.mean(axis=0)
        dists = np.linalg.norm(cluster_points - center, axis=1)
        cluster_max = np.max(dists)
        max_dist = max(max_dist, cluster_max)
    
    return max_dist


def atribuir_rotulos_de_centros(D, centers_idx):
    """Atribui rótulos baseado nos centros mais próximos"""
    if len(centers_idx) == 0:
        return np.zeros(D.shape[0], dtype=int)
    
    # Extrair distâncias apenas para os centros selecionados
    distances_to_centers = D[centers_idx, :]
    return np.argmin(distances_to_centers, axis=0)


def rodar_experimento_unico(
    X,
    true_labels,
    k,
    metric='euclidiana',
    p=2,
    cov=None,
    n_runs=15,
    eps_values=None,
):
    """Executa experimento completo para um dataset"""
    if eps_values is None:
        eps_values = [0.01, 0.05, 0.10, 0.15, 0.25]
    
    results = []
    
    # Calcular matriz de distâncias uma única vez (conforme especificado)
    print(f"      Calculando matriz de distâncias ({metric})...")
    D = calcular_matriz_distancias(X, metric=metric, p=p, cov=cov)
    
    for run in range(n_runs):
        # Configurar seed para reprodutibilidade
        np.random.seed(run * 42)
        
        # --- Algoritmo de Gonzalez ---
        start_time = time.perf_counter()
        try:
            centers_gonzalez, radius_gonzalez = gonzalez_kcenters(X, k, D)
            labels_gonzalez = atribuir_rotulos_de_centros(D, centers_gonzalez)
            end_time = time.perf_counter()
            
            # Calcular métricas de qualidade
            sil_gonzalez = silhouette_score(X, labels_gonzalez) if len(np.unique(labels_gonzalez)) > 1 else -1
            ari_gonzalez = adjusted_rand_score(true_labels, labels_gonzalez) if true_labels is not None else -1
            
            results.append({
                'algoritmo': 'gonzalez',
                'eps': None,
                'run': run,
                'radius': radius_gonzalez,
                'silhouette': sil_gonzalez,
                'ari': ari_gonzalez,
                'tempo': end_time - start_time,
            })
        except Exception as e:
            print(f"      Erro no Gonzalez - run {run}: {e}")
        
        # --- Algoritmo de Refinamento de Intervalo ---
        for eps in eps_values:
            start_time = time.perf_counter()
            try:
                centers_intervalo, radius_intervalo = refinamento_intervalo_kcenters(D, k, eps_fraction=eps)
                labels_intervalo = atribuir_rotulos_de_centros(D, centers_intervalo)
                end_time = time.perf_counter()
                
                sil_intervalo = silhouette_score(X, labels_intervalo) if len(np.unique(labels_intervalo)) > 1 else -1
                ari_intervalo = adjusted_rand_score(true_labels, labels_intervalo) if true_labels is not None else -1
                
                results.append({
                    'algoritmo': 'intervalo',
                    'eps': eps,
                    'run': run,
                    'radius': radius_intervalo,
                    'silhouette': sil_intervalo,
                    'ari': ari_intervalo,
                    'tempo': end_time - start_time,
                })
            except Exception as e:
                print(f"      Erro no Intervalo (eps={eps}) - run {run}: {e}")
        
        # --- K-Means (Baseline) ---
        start_time = time.perf_counter()
        try:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=run)
            labels_kmeans = kmeans.fit_predict(X)
            end_time = time.perf_counter()
            
            radius_kmeans = calcular_raio_kmeans(X, labels_kmeans)
            sil_kmeans = silhouette_score(X, labels_kmeans) if len(np.unique(labels_kmeans)) > 1 else -1
            ari_kmeans = adjusted_rand_score(true_labels, labels_kmeans) if true_labels is not None else -1
            
            results.append({
                'algoritmo': 'kmeans',
                'eps': None,
                'run': run,
                'radius': radius_kmeans,
                'silhouette': sil_kmeans,
                'ari': ari_kmeans,
                'tempo': end_time - start_time,
            })
        except Exception as e:
            print(f"      Erro no K-Means - run {run}: {e}")
    
    return results