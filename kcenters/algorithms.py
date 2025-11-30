import numpy as np

def gonzalez_kcenters(X, k, dist_matrix=None):
    """Algoritmo de Gonzalez (farthest-first) - 2-aproximado"""
    n = X.shape[0]
    
    if dist_matrix is None:
        from kcenters.distances import minkowski_par_distancias
        D = minkowski_par_distancias(X, p=2)
    else:
        D = dist_matrix
    
    centers = []
    # Escolher primeiro centro aleatoriamente para múltiplas execuções
    first_center = np.random.randint(0, n)
    centers.append(first_center)
    
    # Distâncias mínimas aos centros
    min_dists = D[centers[0], :].copy()
    
    for _ in range(1, k):
        # Escolher ponto mais distante dos centros atuais
        farthest_idx = np.argmax(min_dists)
        centers.append(farthest_idx)
        
        # Atualizar distâncias mínimas
        new_dists = D[farthest_idx, :]
        min_dists = np.minimum(min_dists, new_dists)
    
    radius = float(np.max(min_dists))
    return centers, radius

def viavel_com_raio_guloso(D, k, R):
    """Verifica se é possível cobrir todos pontos com k centros de raio R"""
    n = D.shape[0]
    uncovered = np.ones(n, dtype=bool)
    centers = []
    
    while np.any(uncovered) and len(centers) < k:
        # Escolher ponto descoberto arbitrariamente
        uncovered_indices = np.where(uncovered)[0]
        new_center = uncovered_indices[0]
        centers.append(new_center)
        
        # Marcar pontos cobertos (dentro de 2R do novo centro)
        covered = D[new_center, :] <= 2.0 * R
        uncovered[covered] = False
    
    return len(centers) <= k and not np.any(uncovered)

def refinamento_intervalo_kcenters(D, k, eps_fraction=0.05, max_iters=100):
    """Algoritmo de refinamento de intervalo com parâmetro de largura"""
    n = D.shape[0]
    
    # Encontrar limites inicial
    low = 0.0
    high = np.max(D)
    initial_width = high - low
    
    if initial_width == 0:
        return [0], 0.0
    
    # Busca binária no raio ótimo
    for iteration in range(max_iters):
        mid = (low + high) / 2.0
        
        if viavel_com_raio_guloso(D, k, mid):
            high = mid
        else:
            low = mid
        
        # Critério de parada baseado na largura do intervalo
        if (high - low) <= eps_fraction * initial_width:
            break
    
    # Construir solução final com o raio encontrado
    R = high
    uncovered = np.ones(n, dtype=bool)
    centers = []
    
    while np.any(uncovered) and len(centers) < k:
        uncovered_indices = np.where(uncovered)[0]
        new_center = uncovered_indices[0]
        centers.append(new_center)
        
        covered = D[new_center, :] <= 2.0 * R
        uncovered[covered] = False
    
    # Calcular raio real da solução
    if centers:
        min_dists_to_centers = np.min(D[centers, :], axis=0)
        actual_radius = np.max(min_dists_to_centers)
    else:
        actual_radius = 0.0
    
    return centers, actual_radius
