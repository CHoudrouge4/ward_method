import nmslib
import numpy as np
from numpy.random import *
from pyflann import *

dataset = rand(10000, 128)
testset = rand(1, 128)

def print_name():
   print 'hello wards'

class HierarchicalClustering:
    def __init__(self, points):
      self.points = points
      self.n, self.dimension = points.shape

    def __merge(self, mu_a, mu_b, size_a, size_b):
        den = size_a + size_b
        coeff_a = float(size_a)/den
        coeff_b = float(size_b)/den
        return  coeff_a * mu_a + coeff_b * mu_b

    # here we want to implement the main algorithm
    def build_hierarchy():
        pass

#data = numpy.random.randn(10000, 100).astype(numpy.float32)
#a , _ = data.shape
#print a
#hc = HierarchicalClustering(data)

flann = FLANN()
params = flann.build_index(dataset, algorithm='autotuned', target_precision = 0.9, log_level = "info");
#print params
result, dists = flann.nn_index(testset, 1, checks=params['checks']);

print result, "distance ", dists
