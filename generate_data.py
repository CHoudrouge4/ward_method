from sklearn.datasets.samples_generator import make_blobs
from random import *
from numpy import *

range_from = -10
range_to = 10

# c is the number of cluster
# d is the dimension
def generate_random_clusters(c, d):
    centers = []
    for i in range(c):
        p = []
        for j in range(d):
            p.append(randint(range_from, range_to))
        centers.append(p)
    return centers

#Drawing a chart for our generated dataset
# import matplotlib.pyplot as plt
#set colors for the clusters
# colors = ['r','g','b','c','k','y','m']
# c = []
# for i in y:
#     c.append(colors[i])
# Plot the training points
# plt.scatter(X[:, 0], X[:, 1], c= c)
# plt.gray()
# plt.xlabel('X axis')
# plt.ylabel('Y axis')
# plt.show()

"""
    we can loop over the dimension, then over the number of points
"""
number_of_data = 10
max_d = 100
max_n = 10000
# for d in range(2, max_d, 5):
#     for n in range(1000, max_n, 10):
#         for i in range(number_of_data):
#             nb_center = randint(5, 100)
#             centers = generate_random_clusters(nb_center, d);
#             X, y = make_blobs(n_samples=n, n_features=d, centers=centers, cluster_std=0.8, center_box=(1, 10.0), shuffle=True, random_state=0)
#             file_name = './data/data' + str(n) + '_' + str(i) + '_' + str(d) + '.in'
#             head = str(n) + ' ' + str(d) + ' ' + str(nb_center)
#             savetxt(file_name, X, delimiter=' ', newline='\n', comments='', header=head)

nb_center = 100
centers = generate_random_clusters(nb_center, 10)
X, y = make_blobs(n_samples=max_n, n_features=10, centers=centers, cluster_std=0.8, center_box=(1, 10.0), shuffle=True, random_state=0)
file_name = './data/data' + str(max_n) + '_' + str(0) + '_' + str(100) + '.in'
n , d = X.shape
head = str(n) + ' ' + str(d) + ' ' + str(nb_center)
savetxt(file_name, X, delimiter=' ', newline='\n', comments='', header=head)
