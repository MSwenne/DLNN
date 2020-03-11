import numpy as np

class NeuralNetwork:
    def __init__(self):
        
        self.X = np.array([[0,0], [1, 0], [0, 1], [1, 1]])
                  
        self.y = np.array([0,1,1,0]).T[:,np.newaxis]
        self.lr = 1
        self.inputs = 2
        self.hidden = 1
        self.outputs = 1
        self.bias = np.ones(self.X.shape[0])

    def forward(self, w1, w2):
    
        layer_one_in = np.c_[ self.bias, self.X ]
        layer_one_out = np.dot(layer_one_in, w1)
        layer_two_in = np.c_[ self.bias, self.sigmoid(layer_one_out) ]
        layer_output = np.dot(layer_two_in, w2)
        
        return self.sigmoid(layer_output), layer_one_out, layer_two_in, layer_one_in


    def grd_update(self, weights):
        w1 = weights[:,:2]
        w2 = weights[:,-1]
        w2 = w2.reshape(w2.shape[0],1)
        final_layer_output, hidden_layer_output, layer_two_in, layer_one_in = self.forward(w1, w2)

        
        final_layer_error = self.y - final_layer_output
        hidden_layer_error = np.dot(final_layer_error, w2[1:,:].T) * self.sigmoid_grad(hidden_layer_output)

        # Updating weights of the last layer.
        w2 +=  self.lr * np.dot(layer_two_in.T, final_layer_error)
        # Updating weights of the hidden layer.
        w1 += self.lr * np.dot(layer_one_in.T, hidden_layer_error) 
        
        return np.c_[ w1, w2 ]


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_grad(self, x):
        sigmoid = self.sigmoid(x)
        return sigmoid * (1- sigmoid)

if __name__ == "__main__":
    weights = np.random.uniform(size=(3,3))
    model = NeuralNetwork()
    for i in range(1000):
        weights = model.grd_update(weights)

    res, _, _, _ = model.forward(weights[:,:2], weights[:,-1])
    print(res)