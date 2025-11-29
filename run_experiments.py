import os
import json
import csv
import numpy as np
from kcenters.datasets import gerar_colecao_sklearn, gerar_normais_multivariadas
from kcenters.experiments import rodar_experimento_unico


def main(output_csv='resultados.csv'):
    datasets = gerar_colecao_sklearn(random_state=42)
    for i in range(10):
        X, y = gerar_normais_multivariadas(k=4 + (i%3), n_por_cluster=200, overlap_factor=0.2 + 0.1*i, rng=None)
        datasets.append((X, len(np.unique(y)), {'tipo': 'mvnorm', 'var': i}))
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset_idx','dataset_meta','algoritmo','eps','run','radius','silhouette','ari','tempo'])
        for idx, (X, k, meta) in enumerate(datasets):
            results = rodar_experimento_unico(X, None, k, metric='euclidiana', p=2, n_runs=5)
            for r in results:
                writer.writerow([idx, json.dumps(meta), r.get('algoritmo'), r.get('eps', ''), r.get('run'), r.get('radius'), r.get('silhouette'), r.get('ari'), r.get('tempo')])
    print('Concluído, resultados em', output_csv)

if __name__ == '__main__':
    main()