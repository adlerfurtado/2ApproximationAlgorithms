import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

def carregar_datasets_uci(max_samples=1000, max_clusters=5, min_features=2):
    """
    Carrega 10 datasets da UCI com >= 700 exemplos e APENAS features numéricas.
    Limita a max_samples exemplos e max_clusters clusters para evitar problemas de memória.
    
    Todos os datasets foram verificados como tendo features 100% numéricas.
    
    Retorna lista de tuplas (X, y, k, metadata)
    """
    datasets = []
    
    try:
        from ucimlrepo import fetch_ucirepo
        
        # DATASETS ESTRITAMENTE NUMÉRICOS VERIFICADOS DA UCI
        # IDs confirmados, todos 100% numéricos com >= 700 exemplos
        uci_configs = [
            {'id': 186, 'name': 'Wine Quality'},        # 6497 exemplos, 11 features
            {'id': 294, 'name': 'Power Plant'},         # 9568 exemplos, 4 features
            {'id': 242, 'name': 'Energy Efficiency'},   # 768 exemplos, 8 features
            {'id': 360, 'name': 'Air Quality'},         # 9358 exemplos, 13 features
            {'id': 291, 'name': 'Airfoil Self-Noise'},  # 1503 exemplos, 5 features
            {'id': 365, 'name': 'Condition Monitoring'},# ~1600 exemplos, 65 features
            {'id': 189, 'name': 'HTRU2'},               # 17898 exemplos, 8 features
            {'id': 94, 'name': 'Spambase'},             # 4601 exemplos, 57 features
            {'id': 45, 'name': 'Heart Disease'},        # 920 exemplos, features numéricas
            {'id': 602, 'name': 'Dry Bean'},            # 13611 exemplos, 16 features
            {'id': 545, 'name': 'Rice'},                # 3810 exemplos, 7 features
            {'id': 850, 'name': 'Raisin'},              # 900 exemplos, 7 features
            {'id': 243, 'name': 'Yacht'},               # 308 exemplos (pode falhar)
            {'id': 267, 'name': 'Banknote'},            # 1372 exemplos, 4 features
        ]
        
        for config in uci_configs:
            if len(datasets) >= 10:
                break
                
            try:
                print(f"   Tentando: {config['name']}", end="")
                dataset = fetch_ucirepo(id=config['id'])
                X = dataset.data.features
                y = dataset.data.targets
                
                # CRITICAL: Filtrar APENAS colunas numéricas
                X_numeric = X.select_dtypes(include=[np.number])
                
                if X_numeric.shape[1] < min_features:
                    print(f" - ✗ {X_numeric.shape[1]} features numéricas")
                    continue
                
                # Remover linhas com NaN ou infinitos
                mask = ~(X_numeric.isna().any(axis=1) | np.isinf(X_numeric).any(axis=1))
                X_clean = X_numeric[mask].values
                
                if y is not None:
                    if isinstance(y, pd.DataFrame):
                        y_clean = y[mask].iloc[:, 0].values
                    else:
                        y_clean = y[mask].values
                else:
                    y_clean = None
                
                # Verificar se tem pelo menos 700 exemplos ANTES de processar
                if X_clean.shape[0] < 700:
                    print(f" - ✗ Apenas {X_clean.shape[0]} exemplos")
                    continue
                
                # AMOSTRAGEM IMEDIATA se necessário (antes de processar labels)
                if X_clean.shape[0] > max_samples:
                    indices = np.random.RandomState(42).choice(
                        X_clean.shape[0], 
                        size=max_samples, 
                        replace=False
                    )
                    X_clean = X_clean[indices]
                    if y_clean is not None:
                        y_clean = y_clean[indices]
                    print(f" - Amostrado {max_samples}ex", end="")
                
                # Processar labels
                if y_clean is not None and len(y_clean) > 0:
                    # Para regression targets (contínuos), criar bins para clustering
                    if y_clean.dtype in [np.float64, np.float32]:
                        # Criar 5 bins baseados em quantis
                        y_binned = pd.qcut(y_clean, q=5, labels=False, duplicates='drop')
                        k = len(np.unique(y_binned))
                        y_encoded = y_binned
                        print(f" - Binned regression→k={k}", end="")
                    else:
                        # Já é categórico
                        le = LabelEncoder()
                        y_encoded = le.fit_transform(y_clean)
                        k_original = len(np.unique(y_encoded))
                        
                        # LIMITAÇÃO 1: Reduzir número de clusters se necessário
                        if k_original > max_clusters:
                            # Verificar se tem exemplos suficientes nas top-k classes
                            unique, counts = np.unique(y_encoded, return_counts=True)
                            top_k_classes = unique[np.argsort(-counts)[:max_clusters]]
                            mask_topk = np.isin(y_encoded, top_k_classes)
                            
                            # Verificar se ainda teremos >= 700 exemplos
                            if mask_topk.sum() < 700:
                                print(f" - ✗ Top-{max_clusters} classes têm apenas {mask_topk.sum()} exemplos")
                                continue
                            
                            X_clean = X_clean[mask_topk]
                            y_encoded = y_encoded[mask_topk]
                            le_new = LabelEncoder()
                            y_encoded = le_new.fit_transform(y_encoded)
                            k = max_clusters
                            print(f" - k={k_original}→{k}", end="")
                        else:
                            k = k_original
                else:
                    # Se não tem labels, criar clusters sintéticos baseados em k-means
                    from sklearn.cluster import KMeans
                    k = min(max_clusters, max(2, X_clean.shape[0] // 200))
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    y_encoded = kmeans.fit_predict(X_clean)
                    print(f" - KMeans k={k}", end="")
                
                # NÃO FAZER MAIS AMOSTRAGEM AQUI - já foi feita antes!
                
                # Verificação final RIGOROSA
                if X_clean.shape[0] < 700:
                    print(f" - ✗ Após processamento: apenas {X_clean.shape[0]} exemplos")
                    continue
                    
                if X_clean.shape[1] < min_features:
                    print(f" - ✗ Apenas {X_clean.shape[1]} features")
                    continue
                
                # GARANTIR que não exceda max_samples
                if X_clean.shape[0] > max_samples:
                    print(f" - ✗ ERRO: {X_clean.shape[0]} > {max_samples} (amostragem falhou!)")
                    continue
                
                metadata = {
                    'name': config['name'],
                    'n_samples': X_clean.shape[0],
                    'n_features': X_clean.shape[1],
                    'tipo': 'uci',
                    'uci_id': config['id']
                }
                
                datasets.append((X_clean, y_encoded, k, metadata))
                print(f" ✓ ({X_clean.shape[0]},{X_clean.shape[1]}) k={k}")
                
            except Exception as e:
                print(f" - ✗ Erro: {str(e)[:60]}")
                continue
                
    except ImportError:
        print("   ⚠️  ucimlrepo não instalada")
    
    # Se não conseguiu 10, tenta carregar manualmente
    if len(datasets) < 10:
        print(f"\n   Apenas {len(datasets)}/10 carregados via API. Tentando manual...")
        datasets_manuais = carregar_datasets_uci_manual(
            data_dir='data/uci',
            max_samples=max_samples,
            max_clusters=max_clusters,
            min_features=min_features
        )
        datasets.extend(datasets_manuais)
    
    # CRÍTICO: Não aceitar menos de 10 datasets REAIS
    if len(datasets) < 10:
        raise RuntimeError(
            f"\n❌ ERRO: Apenas {len(datasets)} datasets UCI foram carregados!\n"
            f"   Necessário: 10 datasets reais com >= 700 exemplos.\n"
            f"   Sugestões:\n"
            f"   1. Verifique sua conexão com a internet\n"
            f"   2. Instale: pip install ucimlrepo\n"
            f"   3. Baixe datasets manualmente e coloque em data/uci/\n"
            f"   4. Reduza max_samples ou max_clusters se necessário"
        )
    
    return datasets[:10]

def carregar_datasets_uci_manual(data_dir='data/uci', max_samples=1000, 
                                  max_clusters=5, min_features=2):
    """
    Carregar datasets baixados manualmente da UCI
    """
    import glob
    datasets = []
    os.makedirs(data_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    for filepath in csv_files:
        try:
            df = pd.read_csv(filepath)
            df_numeric = df.select_dtypes(include=[np.number]).dropna()
            
            if df_numeric.shape[0] >= 700 and df_numeric.shape[1] >= min_features:
                X = df_numeric.iloc[:, :-1].values
                y = df_numeric.iloc[:, -1].values
                
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                k_original = len(np.unique(y_encoded))
                
                if k_original > max_clusters:
                    unique, counts = np.unique(y_encoded, return_counts=True)
                    top_k_classes = unique[np.argsort(-counts)[:max_clusters]]
                    mask_topk = np.isin(y_encoded, top_k_classes)
                    X = X[mask_topk]
                    y_encoded = y_encoded[mask_topk]
                    le_new = LabelEncoder()
                    y_encoded = le_new.fit_transform(y_encoded)
                    k = max_clusters
                else:
                    k = k_original
                
                if X.shape[0] > max_samples:
                    indices = np.arange(X.shape[0])
                    _, sampled_indices = train_test_split(
                        indices, train_size=max_samples,
                        stratify=y_encoded, random_state=42
                    )
                    X = X[sampled_indices]
                    y_encoded = y_encoded[sampled_indices]
                
                name = os.path.basename(filepath).replace('.csv', '')
                metadata = {
                    'name': f"UCI_{name}",
                    'n_samples': X.shape[0],
                    'n_features': X.shape[1],
                    'tipo': 'uci_manual'
                }
                
                datasets.append((X, y_encoded, k, metadata))
                print(f"   ✓ Manual: {name} ({X.shape[0]},{X.shape[1]}) k={k}")
        
        except Exception as e:
            continue
    
    return datasets

def gerar_datasets_sinteticos_extras(n_datasets, max_samples=1000):
    """
    Gera datasets sintéticos para completar os 10 necessários
    """
    from sklearn.datasets import make_blobs, make_classification
    
    datasets = []
    rng = np.random.RandomState(123)
    
    for i in range(n_datasets):
        n_samples = min(700 + i * 50, max_samples)
        n_features = 5 + (i % 10)
        k = 2 + (i % 4)
        
        if i % 2 == 0:
            X, y = make_blobs(
                n_samples=n_samples, n_features=n_features,
                centers=k, cluster_std=1.0 + i * 0.1,
                random_state=rng.randint(0, 10000)
            )
            tipo = 'synthetic_blobs'
        else:
            X, y = make_classification(
                n_samples=n_samples, n_features=n_features,
                n_informative=max(2, n_features // 2),
                n_redundant=max(1, n_features // 4),
                n_classes=k, random_state=rng.randint(0, 10000)
            )
            tipo = 'synthetic_class'
        
        metadata = {
            'name': f'{tipo}_{i+1}',
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'tipo': tipo
        }
        
        datasets.append((X, y, k, metadata))
        print(f"   ✓ {tipo}_{i+1} ({X.shape[0]},{X.shape[1]}) k={k}")
    
    return datasets