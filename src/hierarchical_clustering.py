from numpy import *
from src.nn_cluster import NNCluster
import ctypes
import math


class HierarchicalClustering:

    lib = ctypes.cdll.LoadLibrary('../lib/librms.so')
    function = lib.radius_min_circle
    function.restype = ctypes.c_double

    def __init__(self, points, epsilon, gamma):
        self.epsilon = epsilon
        self.points = points
        self.n, self.dimension = points.shape
        self.nnc = NNCluster(points, epsilon, gamma)
        self.max_dist = 2 * self.function(self.n, self.dimension, ctypes.c_void_p(points.ctypes.data))
        self.min_dist = self.min_distance()
        self.beta = math.ceil(math.log((self.max_dist/self.min_dist) * self.n))
        self.merges = list()

    def min_distance(self):
        min_dis = float("inf")
        for i in range(self.n):
            _, dist, __ = self.nnc.query(self.points[i], 1)
            min_dis = min(dist, min_dis)
        return min_dis

    @staticmethod
    def __merge(mu_a, mu_b, size_a, size_b):
        den = size_a + size_b
        coeff_a = float(size_a)/den
        coeff_b = float(size_b)/den
        return coeff_a * mu_a + coeff_b * mu_b, size_a + size_b

    # here we want to implement the main algorithm
    def build_hierarchy(self):
        for i in range(self.beta):
            merge_value = math.pow(1 + self.epsilon, i)
            for j in range(self.points.size):
                nn, dist = self.nnc.query(self.points[j], 1)  # I want this one to return a size
                while dist < merge_value:
                    nu, size_nu = self.__merge(nn, 1, self.points[j], 1)
                    self.merges.append((nn, self.points[j]))
                    nn, dist = self.nnc.query(nu, size_nu)  # I want this one to return a size
                    delete(self.points, i)
            if self.points.size == 1:
                return 0
        return 0
