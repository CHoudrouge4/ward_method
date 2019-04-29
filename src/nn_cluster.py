import math
import numpy
from pyflann import *
from numpy.random import *

# flann = FLANN()
# params = flann.build_index(dataset, algorithm='autotuned', target_precision = 0.9, log_level = "info");
# print params
# result, dists = flann.nn_index(testset, 5, checks=params['checks']);

class NNCluster:

    """
        This class is to compute the nearest neighbour cluster
    """

    def __init__(self, points, epsilon, method='hnsw', space='cosinesimil'):
        dataset = rand(100, 128)
        self.size, self.dimension = points.shape
        self.epsilon = epsilon
        self.number_of_data_structure = int(math.ceil(math.log(self.size, 1 + epsilon)))
        self.nn_data_structure = list()
        self.method = method
        self.space = space
        # empty = np.empty(self.dimension)
        for i in range(self.number_of_data_structure):
            self.nn_data_structure.append(FLANN())
            self.nn_data_structure[i].build_index(dataset)

    # ind.add_points(dataset, 2.0)
    # pyf = pyf.build_index(empty, algorithm='autotuned', target_precision = 0.9, log_level ="info")
    # self.nn_data_structure[0].add_points(points)

    @staticmethod
    def __distance(self, size_a, size_b, dist):
        coef = size_a * size_b
        coef = float(coef)/(size_a + size_b)
        return coef * dist

    def query(self, q_cluster, q_size):
        result = numpy.empty(self.dimension)
        min_distance = float("inf")
        for i in range(len(self.nn_data_structure)):
            tmp, dist = self.nn_data_structure[i].nn_index(q_cluster)
            size_tmp = (1 + self.epsilon) ** i
            distance = self.__distance(q_size, size_tmp, dist)
            if dist[0] < min_distance:
                min_distance = distance
                result = tmp
        return result, min_distance

    def add(self, cluster, size):
        i = math.ceil(math.log(size, 1 + self.epsilon))
        self.nn_data_structure[i].add_points([cluster])

    def delete(self, cluster, size):
        i = math.ceil(math.log(size, 1 + self.epsilon))
        self.nn_data_structure[i].remove_point([cluster])
