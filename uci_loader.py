import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def carregar_datasets_uci():
    """
    Carrega 10 datasets da UCI com >= 700 exemplos.
    Retorna lista de tuplas (X, y, k, metadata)
    """
    datasets = []
    
    # Primeiro, tenta carregar via ucimlrepo se estiver disponível
    try:
        from ucimlrepo import fetch_ucirepo
        uci_configs = [
            {'id': 186, 'name': 'Wine Quality', 'min_samples': 1599},
            {'id': 19, 'name': 'Car Evaluation', 'min_samples': 1728},
            {'id': 73, 'name': 'Mushroom', 'min_samples': 8124},
            {'id': 2, 'name': 'Adult', 'min_samples': 48842},
            {'id': 144, 'name': 'Statlog (Shuttle)', 'min_samples': 58000},
            {'id': 94, 'name': 'Spambase', 'min_samples': 4601},
            {'id': 80, 'name': 'Optical Recognition of Handwritten Digits', 'min_samples': 5620},
            {'id': 91, 'name': 'Communities and Crime', 'min_samples': 1994},
            {'id': 1, 'name': 'Abalone', 'min_samples': 4177},
            {'id': 292, 'name': 'Ads', 'min_samples': 3279},
        ]
        
        for config in uci_configs:
            if len(datasets) >= 10:
                break
            try:
                dataset = fetch_ucirepo(id=config['id'])
                X = dataset.data.features
                y = dataset.data.targets
                
                # Converter para numpy e filtrar apenas colunas numéricas
                X_numeric = X.select_dtypes(include=[np.number])
                
                # Remover linhas com NaN
                mask = ~X_numeric.isna().any(axis=1)
                X_clean = X_numeric[mask].values
                
                if y is not None:
                    if isinstance(y, pd.DataFrame):
                        y_clean = y[mask].iloc[:, 0].values
                    else:
                        y_clean = y[mask].values
                else:
                    y_clean = None
                
                # Verificar se tem pelo menos 700 exemplos
                if X_clean.shape[0] >= 700:
                    # Processar labels
                    if y_clean is not None and len(y_clean) > 0:
                        le = LabelEncoder()
                        y_encoded = le.fit_transform(y_clean)
                        k = len(np.unique(y_encoded))
                    else:
                        y_encoded = None
                        k = 3  # default
                    
                    metadata = {
                        'name': config['name'],
                        'n_samples': X_clean.shape[0],
                        'n_features': X_clean.shape[1],
                        'tipo': 'uci',
                        'uci_id': config['id']
                    }
                    
                    datasets.append((X_clean, y_encoded, k, metadata))
                    print(f"   ✓ {config['name']} - {X_clean.shape} - k={k}")
                else:
                    print(f"   ✗ {config['name']} tem apenas {X_clean.shape[0]} exemplos (mínimo: 700)")
                
            except Exception as e:
                print(f"   ✗ Erro ao carregar {config['name']}: {e}")
                continue
    except ImportError:
        print("   Biblioteca ucimlrepo não instalada. Tentando carregar datasets manualmente...")
    
    # Se não conseguiu carregar 10 datasets, tenta carregar manualmente
    if len(datasets) < 10:
        datasets_manuais = carregar_datasets_uci_manual()
        datasets.extend(datasets_manuais)
    
    return datasets[:10]  # Retorna no máximo 10

def carregar_datasets_uci_manual(data_dir='data/uci'):
    """
    Alternativa: carregar datasets baixados manualmente da UCI
    """
    import glob
    
    datasets = []
    
    # Criar diretório se não existir
    os.makedirs(data_dir, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    for filepath in csv_files:
        try:
            df = pd.read_csv(filepath)
            
            # Remover colunas não numéricas e linhas com NaN
            df_numeric = df.select_dtypes(include=[np.number]).dropna()
            
            if df_numeric.shape[0] >= 700:
                # Assumir última coluna como label
                X = df_numeric.iloc[:, :-1].values
                y = df_numeric.iloc[:, -1].values
                
                # Encode labels
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                k = len(np.unique(y_encoded))
                
                name = os.path.basename(filepath).replace('.csv', '')
                metadata = {
                    'name': f"UCI_{name}",
                    'n_samples': X.shape[0],
                    'n_features': X.shape[1],
                    'tipo': 'uci_manual'
                }
                
                datasets.append((X, y_encoded, k, metadata))
                print(f"   ✓ Carregado manualmente: {name} - {X.shape}")
        
        except Exception as e:
            print(f"   ✗ Erro em {filepath}: {e}")
    
    return datasets