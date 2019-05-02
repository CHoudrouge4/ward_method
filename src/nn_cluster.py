import math
import numpy
from pyflann import *
# flann = FLANN()
# params = flann.build_index(dataset, algorithm='autotuned', target_precision = 0.9, log_level = "info");
# print params
# result, dists = flann.nn_index(testset, 5, checks=params['checks']);


class NNCluster:

    """
        This class is to compute the nearest neighbour cluster
    """

    def __init__(self, points, epsilon, gamma=0.9, method='hnsw', space='cosinesimil'):
        self.size, self.dimension = points.shape
        self.epsilon = epsilon
        self.number_of_data_structure = int(math.ceil(math.log(self.size, 1 + epsilon)))
        self.nn_data_structure = list()
        self.method = method
        self.space = space
        self.points = points
        self.gamma = gamma
        # empty = np.empty(self.dimension)
        for i in range(self.number_of_data_structure):
            self.nn_data_structure.append(FLANN())
        self.nn_data_structure[0].build_index(points, algorithm='autotuned', target_precision=gamma)
        self.built = numpy.zeros(len(self.nn_data_structure), bool)
        assert self.built.size == len(self.nn_data_structure)
        self.built[0] = True

    # ind.add_points(dataset, 2.0)
    # pyf = pyf.build_index(empty, algorithm='autotuned', target_precision = 0.9, log_level ="info")
    # self.nn_data_structure[0].add_points(points)

    @staticmethod
    def __distance(size_a, size_b, dist):
        coef = size_a * size_b
        coef = float(coef)/(size_a + size_b)
        return coef * dist

    def query(self, q_cluster, q_size):
        # print("dimension", self.dimension)
        result = numpy.zeros(self.dimension)
        min_distance = float("inf")
        result_size = 1
        for i in range(len(self.nn_data_structure)):
            if not self.built[i]:
                continue
            tmp, dist = self.nn_data_structure[i].nn_index(q_cluster)
            size_tmp = math.floor((1 + self.epsilon) ** i)
            distance = self.__distance(q_size, size_tmp, dist[0])
            # print("the tmp result is ", self.points[tmp[0]], " ", dist[0])
            assert tmp[0] >= 0 < self.points.shape[0]
            if min_distance > distance:
                # print("inside the if statement")
                min_distance = distance
                result = self.points[tmp[0]]
                result_size = i
            assert result.size == self.dimension
        return result, min_distance, result_size

    def add_cluster(self, cluster, size):
        i = math.floor(math.log(size, 1 + self.epsilon))
        if not self.built[i]:
            self.nn_data_structure[i].build_index(cluster)
        else:
            self.nn_data_structure[i].add_points(cluster)

    def delete_cluster(self, idx, size):
        i = math.floor(math.log(size, 1 + self.epsilon))
        self.nn_data_structure[i].remove_point(idx)
