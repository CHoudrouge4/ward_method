import nmslib
import math
import numpy as np
import pyflann as pyf

#flann = FLANN()
#params = flann.build_index(dataset, algorithm='autotuned', target_precision = 0.9, log_level = "info");
#print params
#result, dists = flann.nn_index(testset, 5, checks=params['checks']);

class nnCluster:

    """
        This class is to compute the nearest neighbour cluster
    """

    def __init__(self, size, points, epsilon, method = 'hnsw', space = 'cosinesimil'):
        self.size, self.dimension = points.shape
        self.epsilon = epsilon
        self.number_of_data_structure = int(math.ceil(math.log(size, 1 + epsilon)))
        for i in range(self.number_of_data_structure):
            flann = flan.build_index([], algorithm='autotuned', target_precision = 0.9, log_level "info")
            self.nn_data_structure.append(flann)
        self.nn_data_structure[0].add_points(points)

    def __distance(self, mu_a, size_a, mu_b, size_b, dist):
        coef = size_a * size_b
        coef = float(coef)/(size_a + size_b)
        return coef * dist

    def query(self, q_cluster, q_size): pass
        result = np.empty(self.dimension)
        min_distance = float("inf")
        for i in range(len(self.nn_data_structure)):
            tmp, dist = ds[i].nn_index(q_cluster)
            size_tmp = (1 + self.epsilon) ** i
            distance = __distance(q_cluster, q_size, tmp, size_tmp, dist)
            if dist[0] < min_distance:
                min_distance = distance
                result = tmp
        return result, min_distance

    def add(self, cluster, size):
        index = math.ceil(math.log(size ,1 + self.epsilon))
        self.nn_data_structure[index].add.points([cluster])

    def delete(self, cluster, size):
        index = math.ceil(math.log(size ,1 + self.epsilon))
        self.nn_data_structure[index].add.remove_point([cluster])
