import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

def carregar_datasets_uci():
    """
    Carrega 10 datasets da UCI com >= 700 exemplos.
    Retorna lista de tuplas (X, y, k, metadata)
    """
    datasets = []
    
    # Lista de IDs sugeridos da UCI (ajuste conforme disponibilidade)
    uci_configs = [
        {'id': 53, 'name': 'Iris', 'min_samples': 150},  # exemplo pequeno
        {'id': 109, 'name': 'Wine', 'min_samples': 178},
        {'id': 17, 'name': 'Breast Cancer Wisconsin', 'min_samples': 569},
        {'id': 186, 'name': 'Wine Quality', 'min_samples': 1599},
        {'id': 19, 'name': 'Car Evaluation', 'min_samples': 1728},
        {'id': 73, 'name': 'Mushroom', 'min_samples': 8124},
        {'id': 2, 'name': 'Adult', 'min_samples': 48842},
        {'id': 144, 'name': 'Statlog (Shuttle)', 'min_samples': 58000},
        {'id': 94, 'name': 'Spambase', 'min_samples': 4601},
        {'id': 15, 'name': 'Breast Cancer', 'min_samples': 286},
    ]
    
    for config in uci_configs:
        try:
            # Fetch dataset
            dataset = fetch_ucirepo(id=config['id'])
            X = dataset.data.features
            y = dataset.data.targets
            
            # Converter para numpy e filtrar apenas colunas numéricas
            X_numeric = X.select_dtypes(include=[np.number]).values
            
            # Processar labels
            if y is not None and len(y) > 0:
                if isinstance(y, pd.DataFrame):
                    y_vals = y.iloc[:, 0].values
                else:
                    y_vals = y.values
                
                # Encode labels
                le = LabelEncoder()
                y_encoded = le.fit_transform(y_vals)
                k = len(np.unique(y_encoded))
            else:
                y_encoded = None
                k = 3  # default
            
            # Filtrar por tamanho mínimo
            if X_numeric.shape[0] >= 700:
                metadata = {
                    'name': config['name'],
                    'n_samples': X_numeric.shape[0],
                    'n_features': X_numeric.shape[1],
                    'tipo': 'uci'
                }
                datasets.append((X_numeric, y_encoded, k, metadata))
                print(f"✓ Carregado: {config['name']} - {X_numeric.shape}")
            
        except Exception as e:
            print(f"✗ Erro ao carregar {config['name']}: {e}")
    
    return datasets


def carregar_datasets_uci_manual(data_dir='data/uci'):
    """
    Alternativa: carregar datasets baixados manualmente.
    Organize os CSVs em data/uci/ com nome do dataset.
    """
    import os
    import glob
    
    datasets = []
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    for filepath in csv_files:
        try:
            df = pd.read_csv(filepath)
            
            # Assumir última coluna como label
            X = df.iloc[:, :-1].select_dtypes(include=[np.number]).values
            y = df.iloc[:, -1].values
            
            # Encode labels
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            k = len(np.unique(y_encoded))
            
            if X.shape[0] >= 700:
                name = os.path.basename(filepath).replace('.csv', '')
                metadata = {
                    'name': name,
                    'n_samples': X.shape[0],
                    'n_features': X.shape[1],
                    'tipo': 'uci'
                }
                datasets.append((X, y_encoded, k, metadata))
                print(f"✓ Carregado: {name} - {X.shape}")
        
        except Exception as e:
            print(f"✗ Erro em {filepath}: {e}")
    
    return datasets