from numpy import *
from src.nn_cluster import NNCluster

class HierarchicalClustering:

    def __init__(self, points, epsilon, gamma):
        self.points = points
        self.n, self.dimension = points.shape
        self.nnc = NNCluster(points, epsilon, gamma)

    @staticmethod
    def __merge(mu_a, mu_b, size_a, size_b):
        den = size_a + size_b
        coeff_a = float(size_a)/den
        coeff_b = float(size_b)/den
        return coeff_a * mu_a + coeff_b * mu_b

    # here we want to implement the main algorithm
    def build_hierarchy(self):
