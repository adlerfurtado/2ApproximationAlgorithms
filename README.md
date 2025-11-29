Este projeto implementa e compara algoritmos de agrupamento (clustering):

Algoritmo de Gonzalez (Farthest-First): Algoritmo 2-aproximado guloso
Refinamento de Intervalo: Algoritmo 2-aproximado com busca binária
K-Means: Algoritmo clássico (baseline)

Métricas de Distância Implementadas

Minkowski (p=1: Manhattan, p=2: Euclidiana)
Mahalanobis

Avaliação

Raio da solução (para K-Centros)
Silhueta (qualidade de agrupamento)
Índice de Rand Ajustado (ARI) (comparação com labels verdadeiros)
Tempo de execução

Instalação

1. Clone o repositório

git clone https://github.com/adlerfurtado/2ApproximationAlgorithms/
cd 2ApproximationAlgorithms/

2. Crie um ambiente virtual (recomendado)

python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate

3. Instale as dependências

pip install -r requirements.txt

Como Executar

python run_experiments.py

Saída: 

resultados/resultados_completos.csv - Todos os resultados detalhados

resultados/resumo_por_dataset.csv - Médias por dataset

resultados/resumo_por_tipo.csv - Médias por tipo de dataset

resultados/resumo_global.csv - Médias globais

resultados/RESUMO.md - Relatório em Markdown

Datasets Utilizados 

Sintéticos (Implementados)

Sklearn (30 datasets)

Blobs (5 variações)
Moons (5 variações)
Circles (5 variações)
Varied (5 variações)
Anisotropic (5 variações)
Uniform (5 variações)


Normais Multivariadas (10 datasets)

Clusters gaussianos com sobreposição controlada
Formas elípticas e circulares
