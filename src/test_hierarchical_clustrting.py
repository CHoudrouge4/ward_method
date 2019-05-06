from numpy import *
from src.hierarchical_clustering import HierarchicalClustering
from numpy.random import *

data = rand(5, 2)
epsilon = 0.5
gamma = 0.9

hc = HierarchicalClustering(data, epsilon, gamma)
