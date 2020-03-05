import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from common import as_numpy, confusion_dataframe

class NeuralNetwork:
    def __init__(self):
        self.inputs, self.hidden, self.outputs = 2,2,1
        self.hidden_weights = np.random.uniform(size=(self.inputs,self.hidden))
        self.output_weights = np.random.uniform(size=(self.hidden,self.outputs))
        self.bias = np.ones(size=(self.hidden,1))
        self.lr = 1

    def xor_net(self, x1, x2):
        pass

    def mse(self, y_pred, y_true):
        return np.sum((y_pred-y_true)**2)/len(y_pred)

    def grdmse(self, /):
        pass

    def gradient_descent(self, weights):
        return weights - self.lr*self.grdmse(weights)

if __name__ == "__main__":
    X_train = np.array([[0,0],[0,1],[1,0],[1,1]])
    y_train = np.array([[0],[1],[1],[0]])

    model = NeuralNetwork()
    for i in range(X_train):
        model.xor_net(X_train[i], y_train[i])