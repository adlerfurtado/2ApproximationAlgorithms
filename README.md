# 2-Approximation Algorithms for Clustering

Este projeto implementa e compara diferentes algoritmos de **agrupamento (clustering)**, com foco em K-Centros e K-Means, incluindo métodos aproximados com garantias teóricas.

---

## Algoritmos implementados

### **1. Gonzalez (Farthest-First)**
- Algoritmo guloso **2-aproximado** para **K-Centros**
- Seleciona sempre o ponto mais distante do conjunto atual de centros
- Simples, rápido e com garantia teórica

### **2. Refinamento de intervalo (Binary Search Refinement)**
- Algoritmo **2-aproximado** para **K-Centros**
- Usa busca binária no raio para encontrar uma solução viável
- Mais estável e preciso em certos cenários

### **3. K-Means**
- Algoritmo clássico (baseline)
- Comparado com os aproximados em qualidade e performance

---

## Métricas de Distância Implementadas

### **Minkowski**
- Suporta qualquer valor de `p`
- Casos importantes:
  - `p = 1` → Distância Manhattan  
  - `p = 2` → Distância Euclidiana  

### **Mahalanobis**
- Considera correlações entre atributos  
- Ideal para clusters alongados / elípticos

---

## Avaliação

| Métrica | Descrição |
|--------|-----------|
| **Raio da solução** | Usada para K-Centros |
| **Silhueta** | Mede a qualidade dos clusters |
| **ARI (Adjusted Rand Index)** | Compara com labels verdadeiros |
| **Tempo de execução** | Medida de desempenho |

---

## Instalação

### **1. Clone o repositório**
```bash
git clone https://github.com/adlerfurtado/2ApproximationAlgorithms/
cd 2ApproximationAlgorithms/
```
### **2. Crie um ambiente virtual (recomendado)**
```bash
python -m venv venv
```
Linux/Mac
```bash
source venv/bin/activate
```
Windows
```bash
venv\Scripts\activate
```
### **3. Instale as dependências**
```bash
pip install -r requirements.txt
```

## Como executar
```bash
python run_experiments.py
```

## Saída dos experimentos
Os resultados são gerados em resultados/:
- resultados_completos.csv — Todos os resultados detalhados
- resumo_por_dataset.csv — Médias agrupadas por dataset
- resumo_por_tipo.csv — Médias por tipo de dataset
- resumo_global.csv — Métricas agregadas
- RESUMO.md — Relatório consolidado em Markdown

## Datasets utilizados
### **Sintéticos (via sklearn — 30 datasets)**
- Blobs (5 variações)
- Moons (5 variações)
- Circles (5 variações)
- Varied (5 variações — variâncias diferentes)
- Anisotropic (5 variações — deformados por transformação linear)
- Uniform (5 variações)

### **Normais Multivariadas (10 datasets)**
- Clusters gaussianos com:
- Overlap controlado
- Diferentes matrizes de covariância
- Formatos circulares e elípticos
