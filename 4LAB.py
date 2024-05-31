import numpy as np
import tensorflow as tf
from tensorflow import keras

# Генерация данных
N = 3
K = 1
X_train = np.random.randn(N, N)  # Генерация входных данных
y_train = np.zeros(N)  # Генерация выходных данных, начально заполняем нулями
y_train[:K] = 1  # Первые K событий считаем положительными

# Определение модели нейронной сети
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(N,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')  # Используем среднеквадратичную ошибку

# Обучение модели
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Прогнозирование
X_test = np.random.randn(N, N)  # Генерация тестовых данных
predictions = model.predict(X_test)
print(predictions)