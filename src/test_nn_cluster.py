from numpy import *
from src.nn_cluster import NNCluster, print_hello
from numpy.random import *

dataset = rand(10000, 128)
epsilon = 0.5

points = array([[0.0, 0.1], [0.1, 0.1]])

nnc = NNCluster(points, epsilon)
print('the number of data structure ', nnc.number_of_data_structure)
cluster = array([[1.3, 0.0]])

nnc.add_cluster(cluster, 2)
