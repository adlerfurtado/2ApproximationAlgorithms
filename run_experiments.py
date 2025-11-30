import os
import csv
import numpy as np
import pandas as pd
from kcenters.datasets import gerar_colecao_sklearn, gerar_normais_multivariadas
from kcenters.experiments import rodar_experimento_unico

# Importar o carregador UCI
from uci_loader import carregar_datasets_uci

# Configurações conforme especificação
N_RUNS = 15  # 15 execuções por dataset
EPS_VALUES = [0.01, 0.05, 0.10, 0.15, 0.25]  # 1% a 25%

METRICAS = [
    ('euclidiana', 2, None),      # Euclidiana
    ('minkowski', 1, None),       # Manhattan  
    ('mahalanobis', None, None),  # Mahalanobis
]

def rodar_experimentos_completos(output_dir='resultados'):
    """Executa todos os experimentos conforme especificado"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Coletar todos os datasets
    all_datasets = []
    
    print("=" * 60)
    print("INICIANDO EXPERIMENTOS - TP2 ALGORITMOS 2")
    print("=" * 60)
    
    # 1. Datasets sklearn (30 datasets)
    print("\n1. Gerando 30 datasets sklearn...")
    datasets_sklearn = gerar_colecao_sklearn(random_state=42)
    all_datasets.extend(datasets_sklearn)
    print(f"   ✓ Gerados {len(datasets_sklearn)} datasets sklearn")
    
    # 2. Datasets normais multivariadas (10 datasets)
    print("\n2. Gerando 10 datasets normais multivariadas...")
    rng = np.random.RandomState(42)
    for i in range(10):
        k = 3 + (i % 4)  # k entre 3-6
        overlap = 0.1 + 0.08 * i  # sobreposição controlada
        
        X, y = gerar_normais_multivariadas(
            k=k,
            n_por_cluster=150,
            overlap_factor=overlap,
            rng=rng
        )
        
        meta = {
            'tipo': 'mvnorm',
            'name': f'mvnorm_k{k}_ov{overlap:.2f}',
            'k': k,
            'overlap': overlap
        }
        all_datasets.append((X, k, meta, y))
    print(f"   ✓ Gerados 10 datasets multivariados")
    
    # 3. DATASETS UCI REAIS (10 datasets - CONFORME ESPECIFICAÇÃO)
    print("\n3. Carregando 10 datasets UCI...")
    try:
        uci_datasets = carregar_datasets_uci()
        uci_count = 0
        
        for dataset in uci_datasets:
            if len(dataset) == 4:  # (X, y, k, meta)
                X, y, k, meta = dataset
                # Verificar se tem pelo menos 700 exemplos (conforme especificação)
                if X.shape[0] >= 700:
                    all_datasets.append((X, k, meta, y))
                    uci_count += 1
                    print(f"   ✓ {meta.get('name', 'Unknown')} - {X.shape[0]} amostras, k={k}")
                    
                    if uci_count >= 10:  # Limitar a 10 datasets UCI
                        break
                else:
                    print(f"   ✗ {meta.get('name', 'Unknown')} - apenas {X.shape[0]} amostras (mínimo: 700)")
        
        print(f"   ✓ Total datasets UCI carregados: {uci_count}")
        
        if uci_count < 10:
            print(f"   ⚠️  AVISO: Apenas {uci_count} datasets UCI foram carregados (esperado: 10)")
            
    except Exception as e:
        print(f"   ✗ Erro ao carregar datasets UCI: {e}")
        print("   ⚠️  Continuando com datasets sintéticos apenas")
    
    print(f"\nTOTAL DE DATASETS: {len(all_datasets)}")
    print(f"  - {len(datasets_sklearn)} datasets sklearn")
    print(f"  - 10 datasets multivariados") 
    print(f"  - {len([d for d in all_datasets if 'uci' in str(d[2].get('tipo', ''))])} datasets UCI")
    
    # Arquivo de resultados
    csv_path = os.path.join(output_dir, 'resultados_completos.csv')
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Cabeçalho completo
        writer.writerow([
            'dataset_idx', 'dataset_name', 'dataset_tipo', 'n_samples', 'n_features', 'k',
            'metrica', 'p', 'algoritmo', 'eps', 'run',
            'radius', 'silhouette', 'ari', 'tempo'
        ])
        
        # Processar cada dataset
        for idx, (X, k, meta, true_labels) in enumerate(all_datasets):
            dataset_name = meta['name']
            dataset_tipo = meta['tipo']
            
            print(f"\n[{idx+1}/{len(all_datasets)}] Processando: {dataset_name}")
            print(f"   Shape: {X.shape}, k={k}")
            
            for metric_name, p, cov_param in METRICAS:
                print(f"   Métrica: {metric_name}", end="")
                
                # Configurar parâmetros da métrica
                if metric_name == 'mahalanobis':
                    cov = np.cov(X, rowvar=False)
                    p_val = ''
                else:
                    cov = None
                    p_val = p
                
                try:
                    results = rodar_experimento_unico(
                        X=X,
                        true_labels=true_labels,
                        k=k,
                        metric=metric_name,
                        p=p,
                        cov=cov,
                        n_runs=N_RUNS,
                        eps_values=EPS_VALUES
                    )
                    
                    # Salvar resultados
                    for result in results:
                        writer.writerow([
                            idx, dataset_name, dataset_tipo, X.shape[0], X.shape[1], k,
                            metric_name, p_val, result['algoritmo'], result.get('eps', ''),
                            result['run'], result['radius'], result['silhouette'], 
                            result['ari'], result['tempo']
                        ])
                    
                    print(f" - ✓ Concluído ({len(results)} resultados)")
                    
                except Exception as e:
                    print(f" - ✗ Erro: {e}")
                    continue
    
    print(f"\n{'='*60}")
    print("EXPERIMENTOS CONCLUÍDOS!")
    print(f"Resultados salvos em: {csv_path}")
    print(f"{'='*60}")
    
    # Gerar resumos estatísticos
    gerar_resumo_estatistico(csv_path, output_dir)

def gerar_resumo_estatistico(csv_path, output_dir):
    """Gera tabelas resumo com médias e desvios-padrão"""
    df = pd.read_csv(csv_path)
    
    # Agregações para as métricas
    agg_config = {
        'radius': ['mean', 'std'],
        'silhouette': ['mean', 'std'], 
        'ari': ['mean', 'std'],
        'tempo': ['mean', 'std']
    }
    
    # Resumo por dataset e algoritmo
    resumo_dataset = df.groupby(['dataset_name', 'algoritmo', 'metrica', 'eps']).agg(agg_config)
    resumo_dataset.to_csv(os.path.join(output_dir, 'resumo_por_dataset.csv'))
    
    # Resumo por tipo de dataset
    resumo_tipo = df.groupby(['dataset_tipo', 'algoritmo', 'metrica']).agg(agg_config)
    resumo_tipo.to_csv(os.path.join(output_dir, 'resumo_por_tipo.csv'))
    
    # Resumo global
    resumo_global = df.groupby(['algoritmo', 'metrica', 'eps']).agg(agg_config)
    resumo_global.to_csv(os.path.join(output_dir, 'resumo_global.csv'))
    
    print("Resumos estatísticos gerados!")

if __name__ == '__main__':
    rodar_experimentos_completos()