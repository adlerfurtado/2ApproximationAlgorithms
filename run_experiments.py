import os
import json
import csv
import numpy as np
import pandas as pd
from kcenters.datasets import gerar_colecao_sklearn, gerar_normais_multivariadas
from kcenters.experiments import rodar_experimento_unico
from kcenters.distances import minkowski_par_distancias

# Importar função de carregamento UCI (ajuste o import conforme sua estrutura)
# from uci_loader import carregar_datasets_uci

def calcular_raio_kmeans(X, labels):
    """Calcula o raio da solução KMeans (máxima distância de pontos aos centros)"""
    k = len(np.unique(labels))
    max_dist = 0.0
    for cluster_id in range(k):
        mask = labels == cluster_id
        if np.sum(mask) == 0:
            continue
        cluster_points = X[mask]
        center = np.mean(cluster_points, axis=0)
        dists = np.linalg.norm(cluster_points - center, axis=1)
        max_dist = max(max_dist, np.max(dists))
    return max_dist


def rodar_todos_experimentos(output_dir='resultados'):
    """
    Roda todos os experimentos conforme especificado no trabalho.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Configurações
    N_RUNS = 15
    EPS_VALUES = [0.01, 0.05, 0.10, 0.15, 0.25]  # 1%, 5%, 10%, 15%, 25%
    METRICAS = [
        ('euclidiana', 2, None),      # Euclidiana (p=2)
        ('minkowski', 1, None),        # Manhattan (p=1)
        ('mahalanobis', None, None)    # Mahalanobis
    ]
    
    # Coletar todos os datasets
    all_datasets = []
    
    print("=" * 80)
    print("GERANDO DATASETS SINTÉTICOS")
    print("=" * 80)
    
    # 1. Datasets sklearn (30 datasets)
    print("\n1. Gerando datasets sklearn...")
    sklearn_ds = gerar_colecao_sklearn(random_state=42)
    all_datasets.extend(sklearn_ds)
    print(f"   Total: {len(sklearn_ds)} datasets")
    
    # 2. Datasets normais multivariadas (10 datasets)
    print("\n2. Gerando datasets normais multivariadas...")
    for i in range(10):
        k = 3 + (i % 4)  # k varia entre 3 e 6
        overlap = 0.3 + 0.15 * i  # sobreposição crescente
        rng = np.random.RandomState(100 + i)
        X, y = gerar_normais_multivariadas(
            k=k, 
            n_por_cluster=200, 
            overlap_factor=overlap, 
            rng=rng
        )
        meta = {'tipo': 'mvnorm', 'var': i, 'k': k, 'overlap': overlap}
        all_datasets.append((X, k, meta))
    print(f"   Total: 10 datasets")
    
    # 3. Datasets UCI (10 datasets) - DESCOMENTE quando implementar
    # print("\n3. Carregando datasets UCI...")
    # uci_datasets = carregar_datasets_uci()
    # for X, y, k, meta in uci_datasets:
    #     all_datasets.append((X, k, meta, y))  # Note: inclui y para ARI
    # print(f"   Total: {len(uci_datasets)} datasets")
    
    print(f"\n{'=' * 80}")
    print(f"TOTAL DE DATASETS: {len(all_datasets)}")
    print(f"{'=' * 80}\n")
    
    # Preparar arquivo CSV
    csv_path = os.path.join(output_dir, 'resultados_completos.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'dataset_idx', 'dataset_name', 'dataset_tipo', 'n_samples', 'n_features', 'k',
            'metrica', 'p', 'algoritmo', 'eps', 'run',
            'radius', 'silhouette', 'ari', 'tempo'
        ])
        
        # Rodar experimentos
        for ds_idx, dataset_tuple in enumerate(all_datasets):
            # Desempacotar dataset (pode ter 3 ou 4 elementos)
            if len(dataset_tuple) == 3:
                X, k, meta = dataset_tuple
                true_labels = None
            else:
                X, k, meta, true_labels = dataset_tuple
            
            ds_name = meta.get('name', meta.get('tipo', 'unknown'))
            ds_tipo = meta.get('tipo', 'unknown')
            
            print(f"\n[{ds_idx+1}/{len(all_datasets)}] Processando: {ds_name}")
            print(f"  Shape: {X.shape}, k={k}")
            
            # Testar cada métrica
            for metric_name, p, cov_param in METRICAS:
                if metric_name == 'mahalanobis':
                    # Calcular matriz de covariância
                    cov = np.cov(X, rowvar=False)
                    p_val = ''
                else:
                    cov = None
                    p_val = p
                
                print(f"    Métrica: {metric_name} (p={p_val})")
                
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
                    for r in results:
                        writer.writerow([
                            ds_idx,
                            ds_name,
                            ds_tipo,
                            X.shape[0],
                            X.shape[1],
                            k,
                            metric_name,
                            p_val,
                            r.get('algoritmo'),
                            r.get('eps', ''),
                            r.get('run'),
                            r.get('radius'),
                            r.get('silhouette'),
                            r.get('ari'),
                            r.get('tempo')
                        ])
                    
                    print(f"      ✓ Concluído ({len(results)} resultados)")
                    
                except Exception as e:
                    print(f"      ✗ Erro: {e}")
                    continue
    
    print(f"\n{'=' * 80}")
    print(f"EXPERIMENTOS CONCLUÍDOS!")
    print(f"Resultados salvos em: {csv_path}")
    print(f"{'=' * 80}\n")
    
    # Gerar resumo estatístico
    gerar_resumo_estatistico(csv_path, output_dir)


def gerar_resumo_estatistico(csv_path, output_dir):
    """
    Gera tabelas agregadas com médias e desvios-padrão.
    """
    print("\nGerando resumo estatístico...")
    
    df = pd.read_csv(csv_path)
    
    # Agregar por dataset, algoritmo e métrica
    agg_funcs = {
        'radius': ['mean', 'std'],
        'silhouette': ['mean', 'std'],
        'ari': ['mean', 'std'],
        'tempo': ['mean', 'std']
    }
    
    # Tabela 1: Por dataset e algoritmo
    resumo1 = df.groupby(['dataset_name', 'metrica', 'algoritmo', 'eps']).agg(agg_funcs)
    resumo1.to_csv(os.path.join(output_dir, 'resumo_por_dataset.csv'))
    print(f"  ✓ Salvo: resumo_por_dataset.csv")
    
    # Tabela 2: Por tipo de dataset
    resumo2 = df.groupby(['dataset_tipo', 'metrica', 'algoritmo']).agg(agg_funcs)
    resumo2.to_csv(os.path.join(output_dir, 'resumo_por_tipo.csv'))
    print(f"  ✓ Salvo: resumo_por_tipo.csv")
    
    # Tabela 3: Por algoritmo (global)
    resumo3 = df.groupby(['algoritmo', 'metrica', 'eps']).agg(agg_funcs)
    resumo3.to_csv(os.path.join(output_dir, 'resumo_global.csv'))
    print(f"  ✓ Salvo: resumo_global.csv")
    
    # Gerar relatório markdown
    with open(os.path.join(output_dir, 'RESUMO.md'), 'w') as f:
        f.write("# Resumo dos Experimentos\n\n")
        f.write(f"## Estatísticas Gerais\n\n")
        f.write(f"- Total de execuções: {len(df)}\n")
        f.write(f"- Datasets únicos: {df['dataset_name'].nunique()}\n")
        f.write(f"- Algoritmos testados: {df['algoritmo'].nunique()}\n")
        f.write(f"- Métricas testadas: {df['metrica'].nunique()}\n\n")
        
        f.write("## Melhores Resultados por Métrica\n\n")
        for metric in ['silhouette', 'ari']:
            best = df.nlargest(5, metric)[['dataset_name', 'algoritmo', 'metrica', metric]]
            f.write(f"### Top 5 - {metric.upper()}\n\n")
            f.write(best.to_markdown(index=False))
            f.write("\n\n")
    
    print(f"  ✓ Salvo: RESUMO.md\n")


if __name__ == '__main__':
    rodar_todos_experimentos()