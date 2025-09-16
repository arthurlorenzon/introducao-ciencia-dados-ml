# Exploração do Dataset de Detecção de Fraudes
#
# Dataset: Payment Card Fraud Detection (Kaggle)
# Classes:
# - 0 = Transação legítima
# - 1 = Fraude

#%% 1. Imports e carregamento
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

print("Explorando Dataset de Fraudes")
print("="*30)

# Configurações de visualização
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Caminho do dataset
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
csv_path = os.path.join(DATA_DIR, "luxury_cosmetics_fraud_analysis_2025.csv")

if not os.path.exists(csv_path):
    print(f"⚠️ Dataset não encontrado. Coloque o CSV em: {csv_path}")
    exit()

# Carregando dados
df = pd.read_csv(csv_path)
print("Dados carregados!")

#%% 2. Informações básicas
print("\nINFORMAÇÕES BÁSICAS")
print("="*30)

print("Formato:", df.shape)
print("Colunas:", list(df.columns))

print("\nPrimeiras linhas:")
print(df.head())

print("\nInfo do dataset:")
print(df.info())

print("\nValores faltantes:")
print(df.isnull().sum())

#%% 3. Distribuição da variável alvo (fraude vs legítima)
print("\nDISTRIBUIÇÃO DAS CLASSES")
print("="*30)

if "is_fraud" in df.columns:
    target_col = "is_fraud"
else:
    target_col = df.columns[-1]  # pega a última coluna caso o nome seja diferente

print(df[target_col].value_counts())

# Gráfico simples
plt.figure(figsize=(6, 4))
df[target_col].value_counts().plot(kind="bar", color=["green", "red"])
plt.title("Distribuição de Transações (Legítimas vs Fraudes)")
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.xticks([0, 1], ["Legítima", "Fraude"], rotation=0)
plt.show()

#%% 4. Estatísticas descritivas
print("\nESTATÍSTICAS DESCRITIVAS")
print("="*30)

print("Resumo geral:")
print(df.describe())

print("\nResumo por classe:")
print(df.groupby(target_col).describe())

#%% 5. Visualização das features numéricas
print("\nVISUALIZAÇÃO DAS FEATURES NUMÉRICAS")
print("="*30)

features_numericas = df.select_dtypes(include=[np.number]).columns.drop(target_col)

# Histogramas
n_features = len(features_numericas)
n_cols = 2
n_rows = math.ceil(n_features / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
axes = axes.flatten()

for i, feature in enumerate(features_numericas):
    sns.histplot(df[feature], bins=30, ax=axes[i], kde=True)
    axes[i].set_title(feature)

# Remove subplots vazios
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

#%% 6. Boxplots das features por classe
print("\nBOXPLOTS POR CLASSE")
print("="*30)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
axes = axes.flatten()

for i, feature in enumerate(features_numericas):
    sns.boxplot(x=target_col, y=feature, data=df, ax=axes[i])
    axes[i].set_title(f"{feature} por classe")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

#%% 7. Matriz de correlação
print("\nCORRELAÇÕES")
print("="*30)

correlacao = df[features_numericas].corr()
print("Matriz de correlação:")
print(correlacao.round(3))

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlacao, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlação")
plt.show()

#%% 8. Scatter plot simples entre duas features mais importantes
print("\nSCATTER PLOTS")
print("="*30)

# Pegamos as duas primeiras features numéricas para visualização
if len(features_numericas) >= 2:
    plt.figure(figsize=(8, 6))
    colors = df[target_col].map({0: "green", 1: "red"})
    plt.scatter(df[features_numericas[0]], df[features_numericas[1]], c=colors, alpha=0.6)
    plt.xlabel(features_numericas[0])
    plt.ylabel(features_numericas[1])
    plt.title(f"{features_numericas[0]} vs {features_numericas[1]}")
    plt.show()

#%% 9. Resumo final
print("\nRESUMO")
print("="*30)

print(f"Total de amostras: {len(df)}")
print(f"Número de features: {len(features_numericas)}")
print(f"Número de classes: {df[target_col].nunique()}")
print(f"Classes balanceadas: {df[target_col].value_counts().min() == df[target_col].value_counts().max()}")

print("\nPrincipais observações:")
print("- Dataset desbalanceado (provavelmente muito mais transações legítimas que fraudes)")
print("- Possui múltiplas features numéricas para análise")
print("- Correlações podem indicar variáveis mais importantes")
print("- Próximo passo: aplicar técnicas de balanceamento para treino de modelos")

print("\nExploração concluída!")
