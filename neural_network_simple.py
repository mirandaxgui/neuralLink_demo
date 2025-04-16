# Recriando o script com melhorias para melhor desempenho da rede neural

improved_script = '''\
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# 1. Simular dados fictícios (aumentado para 1000 amostras)
np.random.seed(42)
n_samples = 1000

# [vidas, exames, cursos, tempo_contrato (meses), tentativas_anteriores]
X = np.random.randint(low=1, high=100, size=(n_samples, 5))

# Regra fictícia para gerar Y (aceite ou não)
y = ((X[:, 1] > 30) & (X[:, 3] > 12) & (X[:, 4] > 3)).astype(int)

# 2. Normalizar os dados
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 3. Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Construir o modelo (camada extra adicionada)
model = Sequential()
model.add(Dense(32, input_dim=5, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # saída binária

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. Treinar o modelo (mais épocas)
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2, verbose=1)

# 6. Avaliar o modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Acurácia do modelo: {accuracy:.2%}")

# 7. Curva de aprendizado
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Curva de Aprendizado')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. Testes manuais com novos clientes simulados
cliente_positivo = np.array([[20, 40, 3, 24, 5]])  # Esperado: aceita (≈ 1)
cliente_negativo = np.array([[10, 20, 1, 6, 1]])   # Esperado: não aceita (≈ 0)

cliente_positivo_norm = scaler.transform(cliente_positivo)
cliente_negativo_norm = scaler.transform(cliente_negativo)

pred_positivo = model.predict(cliente_positivo_norm)[0][0]
pred_negativo = model.predict(cliente_negativo_norm)[0][0]

print("\\n--- Teste Manual de Clientes ---")
print(f"Cliente [20, 40, 3, 24, 5] → Probabilidade de ACEITAR: {pred_positivo:.2f} → Esperado: 1")
print(f"Cliente [10, 20, 1, 6, 1] → Probabilidade de ACEITAR: {pred_negativo:.2f} → Esperado: 0")
'''

# Salvar em novo arquivo .py e gerar .zip
file_path = "/mnt/data/rede_neural_farm_demo_melhorado.py"
with open(file_path, "w") as f:
    f.write(improved_script)

file_path
