import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.X = [[0,0,1,1], [0,1,0,1]]
        self.y = [0,1,1,0]
        self.lr = 0.01
        self.inputs = 2
        self.hidden = 2
        self.outputs = 1

    def xor_net(self, x1, x2, weights):
        bias = [1,1,1,1]
        X = np.array([x1, x2, bias])
        hidden = np.dot(X.T, weights[0:2].T).T
        hidden = self.sigmoid(hidden)
        hidden = np.array([hidden[0],hidden[1],bias]).T
        output = np.dot(weights[2].T, hidden.T).T
        output = self.sigmoid(output)
        for i, b in enumerate(output):
            print("\toutput: [",x1[i],x2[1],"]\t",b," -> ",int(b > 0.5))
        return output

    def mse(self, weights):
        estimate = self.xor_net(self.X[0],self.X[1],weights)
        mse = np.sum((self.y-estimate)**2)
        return mse

    def grdmse(self, weights):
        estimate = self.xor_net(self.X[0],self.X[1],weights)
        return 2*np.sum(self.y-estimate)*(-weights)

    def gradient_descent(self, weights):
        return weights + self.lr*self.grdmse(weights)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    weights = np.random.uniform(size=(3,3))
    model = NeuralNetwork()
    for i in range(100):
        # print("iteration: ", i)
        weights = model.gradient_descent(weights)
        print("mse: ",model.mse(weights))
    print("weights: ", weights)