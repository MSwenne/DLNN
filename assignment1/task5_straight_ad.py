import numpy as np

def xor_not(x1, x2, weights, activation):
    (wx1h1, wx2h1, bh1,
     wx1h2, wx2h2, bh2,
     wh1o, wh2o, bo) = weights

    h1pa = wx1h1 * x1 + wx2h1 * x2 + bh1
    h1 = activation(h1pa)
    h2pa = wx1h2 * x1 + wx2h2 * x2 + bh2
    h2 = activation(h2pa)
    opa = wh1o * h1 + wh2o * h2
    o = activation(opa)
    return o

def grad_xor_not(x1, x2, y, weights, activation, activation_diff):
    (wx1h1, wx2h1, bh1,
     wx1h2, wx2h2, bh2,
     wh1o, wh2o, bo) = weights

    # Forward propagation.
    h1pa = wx1h1 * x1 + wx2h1 * x2 + bh1
    h1 = activation(h1pa)
    h2pa = wx1h2 * x1 + wx2h2 * x2 + bh2
    h2 = activation(h2pa)
    opa = wh1o * h1 + wh2o * h2
    o = activation(opa)

    # Backprop.
    E = (y - o)**2
    dEdo = 2*(y - o)

    dEdopa = dEdo * activation_diff(opa)
    dEdwh1o = dEdopa * h1
    dEdwh2o = dEdopa * h2



    dEdopa = dEdo * dodopa

    return np.array([
        dEwx1h1, dEwx2h1, dEbh1,
        dEwx1h2, dEwx2h2, dEbh2,
        dEwh1o, dEwh2o, dEbo
    ])



def mse(weights, activation):
    x1 = np.array([0, 0, 1, 1]).T
    x2 = np.array([0, 1, 0, 1]).T
    target = np.array([0, 1, 1, 0]).T
    return np.mean((target - xor_not(x1, x2, weights, activation))**2)

def grdmse(weights, activation, activation_diff):
    x1 = np.array([0, 0, 1, 1]).T
    x2 = np.array([0, 1, 0, 1]).T
    target = np.array([0, 1, 1, 0]).T
    return np.mean(grad_xor_not(x1, x2, weights, activation_diff), axis=0)


def relu(x):
    return (x >= 0) * x

def relu_diff(x):
    return x >= 0

print(mse([0, 4, 2, 2, 35, 1, 2,4, 0], relu))




