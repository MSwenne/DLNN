
import numpy as np
from math import pi
from math import sqrt
from numpy.random import uniform
from itertools import combinations
import matplotlib.pyplot as plt


def double_factorial(n):

    if n <= 1:
        return 1
    if n == 2:
        return n
    else:
        return n * double_factorial(n-2)

class N_UnitSphere:
    def __init__(self, n):
        self.dimensions = n
        self.side_length = 1
        self.corners = 2**self.dimensions

    def longest_diagonal(self):
        diagonal = 0
        for i in range(self.dimensions):
            diagonal = sqrt(diagonal**2 + 1**2)

        return diagonal

    def epsilon_skin_volume(self, alpha):
        return 1 - (1 - 2*alpha)**self.dimensions

    def pairwise_uniform_distances(self, number_points=1000, histogram=True):
        # initialize
        p = number_points*(number_points-1)//2
        distances = [0]*p
        points = uniform(0, self.side_length, (number_points, self.dimensions))

        i = 0
        for comb in combinations(points, 2):
            distances[i] = sqrt(sum((comb[0] - comb[1])**2))
            i += 1

        if histogram:
            plt.hist(distances, bins=10)
            plt.show()


        return distances

class N_Ball:
    def __init__(self, center, radius, n):
        self.center = center
        self.radius = radius
        self.dimension = n

    def volume(self):
        return (pi*(2**(self.dimension-1))*(self.radius ** self.dimension)) / double_factorial(self.dimension)

        self.radius = radius
        self.dimension = n

def main(iterations = 100):

    k = 1
    for i in range(1,iterations+1):
        sphere = N_UnitSphere(i)
        ball = N_Ball((.5, .5), .5, i)
        print("------------------")
        print(str(i) + " dimensional Sphere and Ball.")
        print("Number of corners Sphere: " + str(sphere.corners))
        print("Longest diagonal of the Sphere: " + str(sphere.longest_diagonal()))
        print("Volume of the Ball: " + str(ball.volume()))
        print("0.01-Skin of the sphere: " + str(sphere.epsilon_skin_volume(0.01)))

        if i == 2**k:
            sphere.pairwise_uniform_distances(1000, False)
            k += 1

if __name__ == '__main__':
    number_of_iterations = 10
    main(number_of_iterations)

