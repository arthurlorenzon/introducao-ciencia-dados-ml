# fraud_decision_tree.py
# =====================================
# Classificação de Fraudes usando Árvore de Decisão
# Dataset: Payment Card Fraud Detection (Kaggle)
# =====================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Árvore de Decisão - Detecção de Fraudes")
print("=" * 40)

# 1. Carregar dataset
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
csv_path = os.path.join(DATA_DIR, "luxury_cosmetics_fraud_analysis_2025.csv")

if not os.path.exists(csv_path):
    print(f"⚠️ Dataset não encontrado. Coloque o CSV em: {csv_path}")
    exit()

df = pd.read_csv(csv_path)
print("✅ Dados carregados!")
print("Formato:", df.shape)

# Definir target
target_col = "is_fraud" if "is_fraud" in df.columns else df.columns[-1]
features_numericas = df.select_dtypes(include=[np.number]).columns.drop(target_col)

X = df[features_numericas].values
y = df[target_col].values

# 2. Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Treino:", X_train.shape, "Teste:", X_test.shape)

# 3. Configuração da Árvore de Decisão
print("\n⚙️ Configurando modelo de Árvore de Decisão...")

tree_model = DecisionTreeClassifier(
    criterion="gini",       # ou "entropy"
    max_depth=5,            # profundidade máxima da árvore
    min_samples_split=10,   # mínimo de amostras para dividir um nó
    min_samples_leaf=5,     # mínimo de amostras em uma folha
    random_state=42
)

# Treinamento
print("\n🚀 Treinando modelo...")
tree_model.fit(X_train, y_train)

# 4. Predição e avaliação
print("\n📊 Avaliando no conjunto de teste...")
y_pred = tree_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Acurácia no teste: {acc:.4f}")

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

print("\n✅ Execução concluída!")