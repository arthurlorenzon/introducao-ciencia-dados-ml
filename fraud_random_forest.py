# =====================================
# Classificação de Fraudes usando Random Forest
# Dataset: Payment Card Fraud Detection (Kaggle)
# =====================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Random Forest - Detecção de Fraudes")
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

# 3. Configuração do Random Forest
print("\n⚙️ Configurando modelo Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,        # número de árvores
    criterion="gini",        # função de impureza (pode ser 'entropy')
    max_depth=None,          # sem limite de profundidade (as árvores crescem até o fim)
    min_samples_split=2,     # mínimo de amostras para dividir um nó
    min_samples_leaf=1,      # mínimo de amostras por folha
    max_features="sqrt",     # número de features consideradas em cada divisão
    bootstrap=True,          # usa amostragem com reposição
    random_state=42,
    n_jobs=-1                # usa todos os núcleos da CPU
)

# Treinamento
print("\n🚀 Treinando modelo Random Forest...")
rf_model.fit(X_train, y_train)

# 4. Predição e avaliação
print("\n📊 Avaliando no conjunto de teste...")
y_pred = rf_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Acurácia no teste: {acc:.4f}")

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# 5. Importância das features
print("\n🔎 Importância das variáveis:")
feature_importances = pd.Series(rf_model.feature_importances_, index=features_numericas)
print(feature_importances.sort_values(ascending=False).head(10))

print("\n✅ Execução concluída!")