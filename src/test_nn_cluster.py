import nn_cluster
from nn_cluster import nnCluster
from numpy.random import *

dataset = rand(10000, 128)
epsilon = 0.5

nnc = nnCluster(dataset, epsilon)
