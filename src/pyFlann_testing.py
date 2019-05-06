from pyflann import  *
from numpy.random import  *
import numpy

data = rand(5, 2)

count = 0
flann = FLANN()
params = flann.build_index(data, algorithm='autotuned', target_precision=0.9, log_level="info")
print(data)
res, dis = flann.nn_index(data[0])
print("first query for 0 ", res, " ", dis)

print("remove ", res[0])
flann.remove_point(res[0])
count = count + 1

res, dis = flann.nn_index(data[0])
print("query for 0 ", res, " ", dis)

print("add zero")
flann.add_points(data[0])

res, dis = flann.nn_index(data[0])
print("query for 0 ", res, " ", dis)
print("remove ", res[0])
flann.remove_point( 5 - 1) # substract the number of |add - remove|

res, dis = flann.nn_index(data[0])
print("query for 0 ", res, " ", dis)


"""
data = rand(5, 2)
testset = rand(1, 2)
print (data)

flann = FLANN()
params = flann.build_index(data)

# flann = FLANN()
# params = flann.build_index(dataset, algorithm='autotuned', target_precision = 0.9, log_level = "info");
# print params
#
result, dists = flann.nn_index(testset, 1, checks=params['checks']);
print('result ', result)
print('querry', testset)
print(data[result[0]])

d_min = float("inf")
for p in data:
    res = numpy.zeros(2)
    d = numpy.linalg.norm(p - testset)
    if d < d_min:
        d_min = d
        result = p

print('brute force', result)
"""