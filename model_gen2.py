import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Activation
import numpy as np 
import matplotlib.pyplot as plt

def generate_model(X_train, y_train):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(2, activation="sigmoid"))

    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=10, validation_split=0.1)

X = np.load("X.npy")
y = np.load("y.npy")

X = X/255.0

X_train = X[:25]
X_test = X[25:]

y_train = y[:25]
y_test = y[25:]

generate_model(X, y)
