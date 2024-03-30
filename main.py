import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
from keras import layers
from keras import models, Sequential
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

myModel = tf.keras.models.Sequential([
    layers.Input(x_train.shape[1:]),
    layers.Flatten(),
    layers.Dense(64, activation='elu'),
    layers.Dense(64, activation='elu'),
    layers.Dense(10, activation='softmax')
])

myModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
myModel.summary()

myModel_history = myModel.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

plt.plot(myModel_history.history['loss'], label='train')
plt.plot(myModel_history.history['val_loss'], label='val')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.plot(myModel_history.history['accuracy'], label='train')
plt.plot(myModel_history.history['val_accuracy'], label='val')
plt.ylabel('accuracy')
plt.legend()
plt.show()
