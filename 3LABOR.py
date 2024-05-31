import tensorflow as tf
import numpy as np

X = np.array([[5, 5], [5, 11], [11, 5], [11, 11]])
y = np.array([[5], [5], [5], [11]])

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(2, input_dim=2, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, y, epochs=1000)
Test=np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Проверка результатов
print(model.predict(Test))