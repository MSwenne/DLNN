import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.X = np.array([ [0,0],
                            [1,1],
                            [1,0],
                            [1,1]])
        self.y = np.array([ [0],
                            [1],
                            [1],
                            [0]])
        self.learning_rate = 0.01
        self.inputs = 2
        self.hidden = 2
        self.outputs = 1

    def xor_net(self, x1, x2, weights):
        (i1h1, i2h1, bh1,
        i1h2, i2h2, bh2,
        h1o, h2o, bo) = weights
        bias = 1

        h1 = x1*i1h1 + x2*i2h1 + bias*bh1
        h2 = x1*i1h2 + x2*i2h2 + bias*bh2
        h1 = self.sigmoid(h1)
        h2 = self.sigmoid(h2)

        out = h1*h1o + h2*h2o + bias*bo
        out = self.sigmoid(out)

        return out

    def mse(self, y, y_pred):
        return (y - y_pred)**2

    def grdmse(self, y, y_pred, weights):
        return 2*(y - y_pred)

    def gradient_descent(self, y, y_pred, weights):
        return weights + self.learning_rate * self.grdmse(y, y_pred, weights)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    weights = np.random.uniform(size=(9,1))
    model = NeuralNetwork()
    X = [[0,0], [0,1], [1,0], [1,1]]
    y = [0, 1, 1, 0]
    for i in range(100):
        # print("iteration: ", i)
        for i, (x1, x2) in enumerate(X):
            y_pred = model.xor_net(x1, x2, weights)
            weights = model.gradient_descent(y[i], y_pred, weights)
            print(x1, " ",x2, " -> ",y_pred)
        # print("mse: ",model.mse(weights))
    print("weights: ", weights)