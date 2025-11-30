import os
import csv
import numpy as np
import pandas as pd

from kcenters.datasets import gerar_colecao_sklearn, gerar_normais_multivariadas
from kcenters.experiments import rodar_experimento_unico
from uci_loader import carregar_datasets_uci

# vamos reaproveitar a função de resumo já existente
from run_experiments import gerar_resumo_estatistico


# ===================== CONFIGURAÇÕES GERAIS =====================

N_RUNS = 15  # número de execuções por dataset (como no enunciado)
EPS_VALUES = [0.01, 0.05, 0.10, 0.15, 0.25]  # 1%, 5%, 10%, 15%, 25%

METRICAS = [
    ('euclidiana', 2, None),      # Euclidiana (p=2)
    ('minkowski', 1, None),       # Manhattan (p=1)
    ('mahalanobis', None, None),  # Mahalanobis
]


def processar_dataset(idx_global, X, k, meta, true_labels, writer):
    """
    Roda TODOS os algoritmos e métricas para um único dataset
    e grava os resultados no CSV via 'writer'.
    """
    ds_name = meta.get('name', meta.get('tipo', 'unknown'))
    ds_tipo = meta.get('tipo', 'unknown')

    print(f"\n[Dataset #{idx_global}] Processando: {ds_name}")
    print(f"  Shape: {X.shape}, k={k}")

    for metric_name, p, cov_param in METRICAS:
        if metric_name == 'mahalanobis':
            # matriz de covariância para Mahalanobis
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
                eps_values=EPS_VALUES,
            )

            for r in results:
                writer.writerow([
                    idx_global,
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
                    r.get('tempo'),
                ])

            print(f"      ✓ Concluído ({len(results)} resultados)")

        except Exception as e:
            print(f"      ✗ Erro nesta métrica: {e}")
            # segue para a próxima métrica sem derrubar tudo
            continue


def rodar_todos_os_datasets_safe(output_dir='resultados'):
    """
    Roda TODOS os datasets (sklearn + mvnorm + UCI) de forma sequencial,
    processando UM POR VEZ e acumulando tudo em resultados_completos.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'resultados_completos.csv')

    # Se o CSV ainda não existe, cria com cabeçalho; senão, continua em modo append.
    if not os.path.exists(csv_path):
        mode = 'w'
        write_header = True
    else:
        mode = 'a'
        write_header = False

    idx_global = 0  # índice único para cada dataset

    with open(csv_path, mode, newline='') as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                'dataset_idx', 'dataset_name', 'dataset_tipo', 'n_samples', 'n_features', 'k',
                'metrica', 'p', 'algoritmo', 'eps', 'run',
                'radius', 'silhouette', 'ari', 'tempo',
            ])

        print("=" * 80)
        print("RODANDO TODOS OS DATASETS EM MODO SEQUENCIAL (SAFE)")
        print("=" * 80)

        # ===================== 1. DATASETS SKLEARN =====================
        print("\n1. Gerando e processando datasets sklearn...")
        sklearn_ds = gerar_colecao_sklearn(random_state=42)  # lista de (X, k, meta, y)
        for (X, k, meta, y) in sklearn_ds:
            idx_global += 1
            processar_dataset(idx_global, X, k, meta, y, writer)
            # aqui o X/y podem ser coletados pelo GC após cada iteração

        # ===================== 2. DATASETS NORM. MULTIVARIADAS =====================
        print("\n2. Gerando e processando datasets normais multivariadas...")
        for i in range(10):
            k = 3 + (i % 4)          # k varia entre 3 e 6
            overlap = 0.3 + 0.15 * i # sobreposição crescente
            rng = np.random.RandomState(100 + i)
            X, y = gerar_normais_multivariadas(
                k=k,
                n_por_cluster=200,
                overlap_factor=overlap,
                rng=rng,
            )
            meta = {
                'tipo': 'mvnorm',
                'var': i,
                'k': k,
                'overlap': overlap,
                'name': f'mvnorm_{i}_k{k}',
            }
            idx_global += 1
            processar_dataset(idx_global, X, k, meta, y, writer)

        # ===================== 3. DATASETS UCI =====================
        print("\n3. Carregando e processando datasets UCI...")
        uci_datasets = carregar_datasets_uci()  # lista de (X, y, k, meta)
        for (X, y, k, meta) in uci_datasets:
            idx_global += 1
            processar_dataset(idx_global, X, k, meta, y, writer)

    print("\n" + "=" * 80)
    print("TODOS OS DATASETS FORAM PROCESSADOS.")
    print(f"Resultados acumulados em: {csv_path}")
    print("=" * 80 + "\n")

    # Gerar resumos a partir do CSV completo
    gerar_resumo_estatistico(csv_path, output_dir)


if __name__ == "__main__":
    rodar_todos_os_datasets_safe(output_dir='resultados')

