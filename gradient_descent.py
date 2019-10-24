import numpy as np


#Using assumptions from previous sections
def initialize(dim):
    b = 0
    w = np.zeros(shape = (dim, 1))
    return w, b


def sigmoid(s):
    return ((1 / (1 + np.exp(-s))))


def gradient_descent(w, b, X, Y, epochs, learn_rate):
    for i in range(epochs):
        z = np.dot(w.T, X) + b
        A = sigmoid(z)
        cost = -(1/m)*[np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))]
        dz = A - Y
        dw = (1/m) * np.dot(X, dz.T)
        db = (1/m) * np.sum(dz)
        w = w - learn_rate * dw
        b = b - learn_rate * db