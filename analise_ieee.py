import pandas as pd
import numpy as np
from scipy import stats
import os

def analise_comparativa_avancada():
    """Análise avançada seguindo o formato do exemplo fornecido"""
    
    print("📊 ANÁLISE COMPARATIVA AVANÇADA - FORMATO IEEE")
    print("=" * 80)
    
    # Carregar dados
    df = pd.read_csv('resultados/resultados_completos.csv')
    
    # Estatísticas básicas
    print("\n1. 📈 ESTATÍSTICAS GERAIS DA EXPERIMENTAÇÃO")
    print("-" * 50)
    print(f"   • Total de execuções: {len(df):,}")
    print(f"   • Datasets: {df['dataset_name'].nunique()}")
    print(f"   • Tipos de datasets: {', '.join(df['dataset_tipo'].unique())}")
    print(f"   • Combinações testadas: {df['algoritmo'].nunique()} algoritmos × {df['metrica'].nunique()} métricas")
    print(f"   • Execuções por combinação: {df['run'].max() + 1}")
    
    # Análise principal: Comparação entre algoritmos
    print("\n2. 🏆 COMPARAÇÃO ENTRE ALGORITMOS (Tabela Principal)")
    print("-" * 60)
    
    comparacao_algoritmos = df.groupby('algoritmo').agg({
        'radius': ['mean', 'std'],
        'silhouette': ['mean', 'std'],
        'ari': ['mean', 'std'],
        'tempo': ['mean', 'std']
    }).round(4)
    
    print("Algoritmo       | Raio Médio  | Silhueta Média | ARI Médio   | Tempo Médio (s)")
    print("-" * 85)
    for algoritmo in ['gonzalez', 'intervalo', 'kmeans']:
        if algoritmo in comparacao_algoritmos.index:
            raio = comparacao_algoritmos.loc[algoritmo, ('radius', 'mean')]
            raio_std = comparacao_algoritmos.loc[algoritmo, ('radius', 'std')]
            silhueta = comparacao_algoritmos.loc[algoritmo, ('silhouette', 'mean')]
            silhueta_std = comparacao_algoritmos.loc[algoritmo, ('silhouette', 'std')]
            ari = comparacao_algoritmos.loc[algoritmo, ('ari', 'mean')]
            ari_std = comparacao_algoritmos.loc[algoritmo, ('ari', 'std')]
            tempo = comparacao_algoritmos.loc[algoritmo, ('tempo', 'mean')]
            tempo_std = comparacao_algoritmos.loc[algoritmo, ('tempo', 'std')]
            
            print(f"{algoritmo:15} | {raio:6.4f} ± {raio_std:.4f} | {silhueta:8.4f} ± {silhueta_std:.4f} | {ari:6.4f} ± {ari_std:.4f} | {tempo:8.4f} ± {tempo_std:.4f}")
    
    # Análise do efeito do ε
    print("\n3. 🎯 EFEITO DO PARÂMETRO ε NO REFINAMENTO")
    print("-" * 60)
    
    refinamento_data = df[df['algoritmo'] == 'intervalo']
    if not refinamento_data.empty:
        eps_effect = refinamento_data.groupby('eps').agg({
            'radius': ['mean', 'std'],
            'silhouette': ['mean', 'std'],
            'tempo': ['mean', 'std']
        }).round(4)
        
        print("ε    | Raio Médio  | Silhueta Média | Tempo Médio (s)")
        print("-" * 55)
        for eps in sorted(eps_effect.index):
            raio = eps_effect.loc[eps, ('radius', 'mean')]
            raio_std = eps_effect.loc[eps, ('radius', 'std')]
            silhueta = eps_effect.loc[eps, ('silhouette', 'mean')]
            silhueta_std = eps_effect.loc[eps, ('silhouette', 'std')]
            tempo = eps_effect.loc[eps, ('tempo', 'mean')]
            tempo_std = eps_effect.loc[eps, ('tempo', 'std')]
            
            print(f"{eps:4} | {raio:8.4f} ± {raio_std:.4f} | {silhueta:10.4f} ± {silhueta_std:.4f} | {tempo:8.4f} ± {tempo_std:.4f}")
    
    # Comparação entre métricas
    print("\n4. 📏 COMPARAÇÃO ENTRE MÉTRICAS DE DISTÂNCIA")
    print("-" * 60)
    
    metric_comparison = df.groupby('metrica').agg({
        'radius': ['mean', 'std'],
        'silhouette': ['mean', 'std'],
        'tempo': ['mean', 'std']
    }).round(4)
    
    print("Métrica    | Raio Médio  | Silhueta Média | Tempo Médio (s)")
    print("-" * 65)
    for metrica in ['euclidiana', 'minkowski', 'mahalanobis']:
        if metrica in metric_comparison.index:
            raio = metric_comparison.loc[metrica, ('radius', 'mean')]
            raio_std = metric_comparison.loc[metrica, ('radius', 'std')]
            silhueta = metric_comparison.loc[metrica, ('silhouette', 'mean')]
            silhueta_std = metric_comparison.loc[metrica, ('silhouette', 'std')]
            tempo = metric_comparison.loc[metrica, ('tempo', 'mean')]
            tempo_std = metric_comparison.loc[metrica, ('tempo', 'std')]
            
            print(f"{metrica:10} | {raio:8.4f} ± {raio_std:.4f} | {silhueta:10.4f} ± {silhueta_std:.4f} | {tempo:8.4f} ± {tempo_std:.4f}")
    
    # Análise por tipo de dataset
    print("\n5. 🎪 DESEMPENHO POR TIPO DE DATASET")
    print("-" * 60)
    
    tipos_analise = ['blobs', 'moons', 'circles', 'mvnorm', 'uci']
    for tipo in tipos_analise:
        if tipo in df['dataset_tipo'].values:
            tipo_data = df[df['dataset_tipo'] == tipo]
            
            # Melhor algoritmo para este tipo
            melhor_algo = tipo_data.groupby('algoritmo')['silhouette'].mean().idxmax()
            melhor_silhueta = tipo_data.groupby('algoritmo')['silhouette'].mean().max()
            
            # Melhor métrica para este tipo
            melhor_metrica = tipo_data.groupby('metrica')['silhouette'].mean().idxmax()
            melhor_silhueta_metrica = tipo_data.groupby('metrica')['silhouette'].mean().max()
            
            print(f"   • {tipo:15}:")
            print(f"        Melhor algoritmo: {melhor_algo} (Silhueta: {melhor_silhueta:.4f})")
            print(f"        Melhor métrica: {melhor_metrica} (Silhueta: {melhor_silhueta_metrica:.4f})")
            print(f"        N° datasets: {tipo_data['dataset_name'].nunique()}")
    
    # Testes de significância estatística
    print("\n6. 📊 SIGNIFICÂNCIA ESTATÍSTICA (Teste t de Student)")
    print("-" * 60)
    
    algoritmos = ['gonzalez', 'intervalo', 'kmeans']
    print("   Comparação de Silhueta entre algoritmos:")
    for i, algo1 in enumerate(algoritmos):
        for algo2 in algoritmos[i+1:]:
            data1 = df[df['algoritmo'] == algo1]['silhouette'].dropna()
            data2 = df[df['algoritmo'] == algo2]['silhouette'].dropna()
            
            if len(data1) > 1 and len(data2) > 1:
                t_stat, p_value = stats.ttest_ind(data1, data2)
                significancia = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
                print(f"      {algo1} vs {algo2}: p = {p_value:.6f} {significancia}")
    
    # Insights principais
    print("\n7. 💡 PRINCIPAIS INSIGHTS PARA O RELATÓRIO")
    print("-" * 60)
    
    # Insight 1: Algoritmo com melhor desempenho geral
    melhor_algo_geral = df.groupby('algoritmo')['silhouette'].mean().idxmax()
    melhor_silhueta_geral = df.groupby('algoritmo')['silhouette'].mean().max()
    print(f"   • Algoritmo com melhor Silhueta geral: {melhor_algo_geral} ({melhor_silhueta_geral:.4f})")
    
    # Insight 2: Algoritmo mais rápido
    algo_mais_rapido = df.groupby('algoritmo')['tempo'].mean().idxmin()
    tempo_mais_rapido = df.groupby('algoritmo')['tempo'].mean().min()
    print(f"   • Algoritmo mais rápido: {algo_mais_rapido} ({tempo_mais_rapido:.4f}s)")
    
    # Insight 3: Melhor métrica
    melhor_metrica_geral = df.groupby('metrica')['silhouette'].mean().idxmax()
    silhueta_melhor_metrica = df.groupby('metrica')['silhouette'].mean().max()
    print(f"   • Melhor métrica geral: {melhor_metrica_geral} ({silhueta_melhor_metrica:.4f})")
    
    # Insight 4: EPS ideal
    if not refinamento_data.empty:
        melhor_eps = refinamento_data.groupby('eps')['silhouette'].mean().idxmax()
        silhueta_melhor_eps = refinamento_data.groupby('eps')['silhouette'].mean().max()
        print(f"   • EPS ideal para refinamento: {melhor_eps} ({silhueta_melhor_eps:.4f})")
    
    return df

def gerar_tabelas_latex():
    """Gera código LaTeX para as tabelas do relatório IEEE"""
    
    df = pd.read_csv('resultados/resultados_completos.csv')
    
    print("\n" + "="*80)
    print("📋 CÓDIGO LATEX PARA AS TABELAS DO RELATÓRIO")
    print("="*80)
    
    # Tabela 1: Comparação entre algoritmos
    print("\n% --- TABELA 1: COMPARAÇÃO ENTRE ALGORITMOS ---")
    print("\\begin{table}[htbp]")
    print("\\caption{Desempenho médio dos algoritmos avaliados}")
    print("\\label{tab:comparacao-algoritmos}")
    print("\\begin{center}")
    print("\\begin{tabular}{|c|c|c|c|c|}")
    print("\\hline")
    print("Algoritmo & Raio Médio & Silhueta Média & ARI Médio & Tempo Médio (s) \\\\")
    print("\\hline")
    
    comparacao = df.groupby('algoritmo').agg({
        'radius': 'mean',
        'silhouette': 'mean', 
        'ari': 'mean',
        'tempo': 'mean'
    }).round(4)
    
    for algoritmo in ['gonzalez', 'intervalo', 'kmeans']:
        if algoritmo in comparacao.index:
            raio = comparacao.loc[algoritmo, 'radius']
            silhueta = comparacao.loc[algoritmo, 'silhouette']
            ari = comparacao.loc[algoritmo, 'ari']
            tempo = comparacao.loc[algoritmo, 'tempo']
            
            # Nomes em português para o relatório
            nome_algo = {
                'gonzalez': 'Gonzalez',
                'intervalo': 'Refinamento',
                'kmeans': 'K-Means'
            }
            
            print(f"{nome_algo[algoritmo]} & {raio} & {silhueta} & {ari} & {tempo} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{center}")
    print("\\end{table}")
    
    # Tabela 2: Efeito do ε
    print("\n% --- TABELA 2: EFEITO DO PARÂMETRO ε ---")
    print("\\begin{table}[htbp]")
    print("\\caption{Impacto do parâmetro $\\varepsilon$ no algoritmo de refinamento}")
    print("\\label{tab:efeito-epsilon}")
    print("\\begin{center}")
    print("\\begin{tabular}{|c|c|c|c|}")
    print("\\hline")
    print("$\\varepsilon$ & Raio Médio & Silhueta Média & Tempo Médio (s) \\\\")
    print("\\hline")
    
    refinamento = df[df['algoritmo'] == 'intervalo']
    if not refinamento.empty:
        eps_effect = refinamento.groupby('eps').agg({
            'radius': 'mean',
            'silhouette': 'mean',
            'tempo': 'mean'
        }).round(4)
        
        for eps in sorted(eps_effect.index):
            raio = eps_effect.loc[eps, 'radius']
            silhueta = eps_effect.loc[eps, 'silhouette']
            tempo = eps_effect.loc[eps, 'tempo']
            
            print(f"{eps} & {raio} & {silhueta} & {tempo} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{center}")
    print("\\end{table}")
    
    # Tabela 3: Comparação entre métricas
    print("\n% --- TABELA 3: COMPARAÇÃO ENTRE MÉTRICAS ---")
    print("\\begin{table}[htbp]")
    print("\\caption{Desempenho por métrica de distância}")
    print("\\label{tab:comparacao-metricas}")
    print("\\begin{center}")
    print("\\begin{tabular}{|c|c|c|c|}")
    print("\\hline")
    print("Métrica & Raio Médio & Silhueta Média & Tempo Médio (s) \\\\")
    print("\\hline")
    
    metric_comparison = df.groupby('metrica').agg({
        'radius': 'mean',
        'silhouette': 'mean',
        'tempo': 'mean'
    }).round(4)
    
    for metrica in ['euclidiana', 'minkowski', 'mahalanobis']:
        if metrica in metric_comparison.index:
            raio = metric_comparison.loc[metrica, 'radius']
            silhueta = metric_comparison.loc[metrica, 'silhouette']
            tempo = metric_comparison.loc[metrica, 'tempo']
            
            # Nomes em português
            nome_metrica = {
                'euclidiana': 'Euclidiana',
                'minkowski': 'Manhattan',
                'mahalanobis': 'Mahalanobis'
            }
            
            print(f"{nome_metrica[metrica]} & {raio} & {silhueta} & {tempo} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{center}")
    print("\\end{table}")

if __name__ == '__main__':
    print("🎓 ANÁLISE PARA RELATÓRIO IEEE - TP2 ALGORITMOS 2")
    print("=" * 80)
    
    df = analise_comparativa_avancada()
    gerar_tabelas_latex()
    
    print(f"\n{'='*80}")
    print("✅ ANÁLISE CONCLUÍDA! Use os dados acima para completar seu relatório.")
    print(f"{'='*80}")