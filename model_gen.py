import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
import numpy as np 
import matplotlib.pyplot as plt
import pickle

def generate_model(X_train, y_train):
    model = Sequential()

    model.add(Flatten())
    model.add(Dense(64, input_shape=X.shape[1:], activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(2, activation="sigmoid"))

    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=10, validation_split=0.1)

    pickle.dump("model.pickle", model)

X = np.load("X.npy")
y = np.load("y.npy")

X = X/255.0

X_train = X[:25]
X_test = X[25:]

y_train = y[:25]
y_test = y[25:]

generate_model(X_train, y_train)
