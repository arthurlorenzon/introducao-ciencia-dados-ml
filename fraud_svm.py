```python
# fraud_svm.py
# =====================================
# Classificação de Fraudes usando SVM (Support Vector Machine)
# Dataset: Payment Card Fraud Detection (Kaggle)
# =====================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("SVM - Detecção de Fraudes")
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

# Normalização (essencial para SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Configuração do modelo SVM
print("\n⚙️ Configurando SVM com hiperparâmetros padrão...")

svm_model = SVC(
    kernel="rbf",       # kernel radial (mais usado)
    C=1.0,              # penalidade de erro
    gamma="scale",      # ajuste automático de gamma
    probability=True,   # permite calcular probabilidades
    random_state=42
)

# Treinamento
print("\n🚀 Treinando modelo SVM...")
svm_model.fit(X_train, y_train)

# 4. Predição e avaliação
print("\n📊 Avaliando no conjunto de teste...")
y_pred = svm_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Acurácia no teste: {acc:.4f}")

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

print("\n✅ Execução concluída!")
```
