import keras
import tensorflow as tf
from markdown_it.rules_core import inline
from tensorflow import keras
import matplotlib.pyplot as plt


import numpy as np


(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()

print(len(X_train))
print(len(X_test))

print(X_train[0].shape)

print(X_train[0])



print(X_train.shape)

X_train = X_train /255
X_test = X_test /255


'''Pasar la matriz a una unica dimension para pasarlo por parametro a la red neuronal'''
X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)

print(X_train_flattened.shape)
print(X_test_flattened.shape)

'''Creamos una variable modelo en la que guardamos la red neuronal, compuesta de 10 neuronas de salida, 784 neuronas de entrada y una funcion sigmoidal de activacion'''
model = keras.Sequential([
    keras.layers.Dense(600, activation="relu"),
    keras.layers.Dense(10, input_shape=(784,), activation = 'sigmoid')
])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(X_train_flattened, y_train, epochs=5)


model.evaluate(X_test_flattened, y_test)

plt.matshow(X_test[7])
plt.show()


y_predicted = model.predict(X_test_flattened)

print(np.argmax(y_predicted[7]))