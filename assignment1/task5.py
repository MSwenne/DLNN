import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.learning_rate = 0.1

    def xor_net(self, x1, x2, weights):
        (i1h1, i2h1, bh1,
        i1h2, i2h2, bh2,
        h1o, h2o, bo) = weights
        bias = 1

        h1_pre = i1h1*x1 + i2h1*x2 + bh1*bias
        h1 = self.sigmoid(h1_pre)
        h2_pre = i1h2*x1 + i2h2*x2 + bh2*bias
        h2 = self.sigmoid(h2_pre)
        out_pre = h1o*h1 + h2o*h2 + bo*bias
        out = self.sigmoid(out_pre)

        return h1, h2, out

    def mse(self, weights):
        x1 = np.array([0,0,1,1]).T
        x2 = np.array([0,1,0,1]).T
        y  = np.array([0,1,1,0]).T
        _, _, y_pred = self.xor_net(x1, x2, weights)

        for i in range(len(x1)):
            print(f"[{x1.T[i]},{x2.T[i]}] -> {y_pred[i]}")

        return np.mean(y - y_pred)**2

    def grdmse(self, weights):
        x1 = np.array([0,0,1,1]).T
        x2 = np.array([0,1,0,1]).T
        y  = np.array([0,1,1,0]).T
        bias = np.array([1,1,1,1]).T

        h1, h2, out = self.xor_net(x1, x2, weights)
        hidden = np.array((h1, h2, bias)).T
        inputs = np.array((x1, x2, bias)).T

        delta3 = (y - out).T
        delta2 = np.tensordot(delta3, weights[3:6], axes=0).T
        delta1 = np.tensordot(delta3, weights[0:3], axes=0).T

        w_grad3 = np.matmul(delta3, self.sigmoid_diff(hidden))
        w_grad2 = np.matmul(delta2, self.sigmoid_diff(inputs))
        w_grad1 = np.matmul(delta1, self.sigmoid_diff(inputs))

        return np.concatenate((np.mean(w_grad1,axis=0)[0], np.mean(w_grad2,axis=0)[0], w_grad3))

    def gradient_descent(self, weights):
        return weights - self.learning_rate * np.array([self.grdmse(weights)]).T

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_diff(self, x):
        return x * (1.0 - x)

if __name__ == "__main__":
    weights = np.random.uniform(size=(9,1))
    model = NeuralNetwork()
    # print("mse",model.mse(weights))
    for i in range(10000):
        weights = model.gradient_descent(weights)
    print("mse",model.mse(weights))
        # print("weights: ", weights)
