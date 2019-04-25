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

    def __distance(self, mu_a, size_a, mu_b, size_b):
        coef = size_a * size_b
        coef = float(coef)/(size_a + size_b)
        return coef * math.sqrt(np.sum((mu_a - mu_b)**2))

    def query(self, cluster): pass


    def add(self, cluster, size):
        index = math.ceil(math.log(size ,1 + self.epsilon))
        self.nn_data_structure[index].add.points([cluster])

    def delete(self, cluster, size):
        index = math.ceil(math.log(size ,1 + self.epsilon))
        self.nn_data_structure[index].add.remove_point([cluster])
