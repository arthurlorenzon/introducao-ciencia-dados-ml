# fraud_knn.py
# =====================================
# Classificação de Fraudes usando KNN (K-Nearest Neighbors)
# Dataset: Payment Card Fraud Detection (Kaggle)
# =====================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("KNN - Detecção de Fraudes")
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

# Normalização (essencial para KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Configuração do KNN
print("\n⚙️ Configurando modelo KNN...")

knn_model = KNeighborsClassifier(
    n_neighbors=5,        # número de vizinhos
    metric="minkowski",   # métrica de distância (padrão = euclidiana)
    p=2,                  # p=2 → distância euclidiana
    weights="uniform",    # todos os vizinhos com o mesmo peso
    n_jobs=-1             # usa todos os núcleos da CPU
)

# Treinamento
print("\n🚀 Treinando modelo KNN...")
knn_model.fit(X_train, y_train)

# 4. Predição e avaliação
print("\n📊 Avaliando no conjunto de teste...")
y_pred = knn_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Acurácia no teste: {acc:.4f}")

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

print("\n✅ Execução concluída!")