from numpy import *
from src.nn_cluster import NNCluster
from numpy.random import *
import random

def brute_force(points, query):
    d_min = float("inf")
    _, dimension = data.shape
    res = zeros(dimension)
    for p in points:
        d = linalg.norm(p - query)
        if d < d_min:
            d_min = d
            res = p
    return res


def test_accuracy(points, q_size, nbiterations, epsilon, gamma):
    nnc = NNCluster(points, epsilon, gamma)
    _, dimension = points.shape
    for i in range(nbiterations):
        q = rand(1, dimension)
        result, _ = nnc.query(q, q_size)
        res = brute_force(data, q)
        assert (res == result).all()


def test_add(points, nbiterations, epsilon, gamma):
    nnc = NNCluster(points, epsilon, gamma)
    point_size, dimension = points.shape
    for i in range(nbiterations):
        p = rand(1, dimension)
        p_size = random.randint(1, point_size)
        nnc.add_cluster(p, p_size)

data_size = randint(10, 100)
data_dimension = randint(1, 200)
data = rand(data_size, data_dimension)
epsilon = random.uniform(0.1, 1)
number_buckets = int(math.ceil(math.log(data_size, 1 + epsilon)))
gamma = 1
query_size = 1 #random.randint(1, number_buckets)
nb_iterations = random.randint(100, 10000)
# test_accuracy(data, query_size, nb_iterations, epsilon, gamma)
test_add(data, nb_iterations, epsilon, gamma)
