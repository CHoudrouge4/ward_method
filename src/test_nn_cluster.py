from numpy import *
from src.nn_cluster import NNCluster
from numpy.random import *

epsilon = 0.5

points = array([[0.0, 0.1], [0.1, 0.1]])

nnc = NNCluster(points, epsilon)
print('the number of data structure ', nnc.number_of_data_structure)
cluster = array([[0.3, 0.0]])

q = array([0.0, 0.0])
print(q)

result, dist = nnc.query(q, 1)
print (result, " ",  dist)


nnc.add_cluster(cluster, 2)
nnc.delete_cluster(0, 1)