import nmslib
import math
import numpy as np
import pyflann as pyf

class nnCluster:

    #data_structure = nmslib.init(method='hnsw', space='cosinesimil')

    def __init__(self, size, epsilon, method = 'hnsw', space = 'cosinesimil'):
        self.size = size
        self.epsilon = epsilon
        self.number_of_data_structure = int(math.ceil(math.log(size, 1 + epsilon)))
        for i in range(self.number_of_data_structure):
#            self.nn_data_structure.append


#self.nn_data_structure = nmslib.init(method, space)

    def query(cluster): pass

    def add(cluster):
        self.nn_data_structure.addDataPoint(None, cluster)

    def delete(cluster): pass

    def size(): pass
