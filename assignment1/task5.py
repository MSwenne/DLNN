import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.X = [[0,0], [0,1], [1,0], [1,1]]
        self.y = [[0],   [1],   [1],   [0]]
        self.lr = 0.1
        self.inputs = 2
        self.hidden = 2
        self.outputs = 1

    def xor_net(self, x1, x2, weights):
        bias = 1
        X = np.array([x1, x2, bias])
        hidden = np.dot(weights[0:2], X.T).T
        hidden = self.sigmoid(hidden)
        hidden = np.append(hidden, bias)
        output = np.dot(weights[2], hidden.T).T
        output = self.sigmoid(output)
        print("input:  [", x1, x2, "]\toutput: ", int(output > 0.5))
        return output

    def mse(self, weights):
        mse = np.sum([(self.xor_net(x[0],x[1],weights)-self.y[i])**2 for i, x in enumerate(self.X)])/len(self.X)
        print("mse: ", mse)
        return mse

    def grdmse(self, weights):
        print("HERE:",self.mse(weights))
        return np.gradient(self.mse(weights),weights)

    def gradient_descent(self, weights):
        return weights - self.lr*self.grdmse(weights)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    weights = np.random.uniform(size=(3,3))
    model = NeuralNetwork()
    for _ in range(10):
        weights = model.gradient_descent(weights)
        print("weights: ", weights)