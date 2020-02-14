import matplotlib.pyplot as plt
from numpy import pi
from math import sin
import numpy as np
import random

N = [9, 15, 100]
dim = 2

y = lambda x: 0.5 + 0.4*np.sin(2*np.pi*x) 

for n in N:
    x_dense = np.linspace(0, 1, 10000)

    x_train = np.linspace(0, 1, n)
    y_train = y(x_train) + np.random.normal(0, 0.05, n)
    
    x_test = np.linspace(0, 1, n)
    y_test = y(x_test) +  np.random.normal(0, 0.05, n)

    # w = [random.random() for i in range(dim+1)]
    # for i in range(n):
    #     y = w[0] + w[1]*x[i]

    fig, axs = plt.subplots(2, 2)
    for ax, degree in [(axs[0,0], 0), (axs[0,1], 1), (axs[1,0], 3), (axs[1,1], 9)]:
        coeffs = np.polyfit(x_train, y_train, degree)
        poly = np.poly1d(coeffs)

        ax.set_title(f'Degree {degree}')
        ax.plot(x_dense, y(x_dense), '-g')
        ax.plot(x_dense, poly(x_dense), '-r')
        ax.plot(x_train, y_train, 'ob', fillstyle='none')
        
    for ax in axs.flat:
        ax.set(xlabel='x', ylabel='t')
    fig.tight_layout()
    # plt.plot(x_test, y_test, '.r')
plt.show()
