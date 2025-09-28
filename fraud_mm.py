#%% 10. Preparação para Machine Learning
print("\nPREPARAÇÃO PARA ML")
print("="*30)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Features (X) e alvo (y)
X = df[features_numericas].values
y = df[target_col].values

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalização
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Treino:", X_train.shape, "Teste:", X_test.shape)

#%% 11. Definição da Rede Neural
print("\nCONFIGURANDO REDE NEURAL")
print("="*30)

model = Sequential([
    Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")  # saída binária
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

#%% 12. Treinamento
print("\nTREINANDO MODELO")
print("="*30)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

#%% 13. Avaliação no conjunto de teste
print("\nAVALIAÇÃO NO TESTE")
print("="*30)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Acurácia no teste: {acc:.4f}")

#%% 14. Predições
print("\nPREDIÇÕES")
print("="*30)

y_pred = (model.predict(X_test) > 0.5).astype("int")

from sklearn.metrics import classification_report, confusion_matrix

print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
