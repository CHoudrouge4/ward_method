from sklearn.cluster import AgglomerativeClustering
import numpy as np
from numpy.random import *
import time
from datetime import datetime

X = rand(20000,2)


n_clusters = 2
ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',
                                connectivity=None)
start = time.time()
ward.fit(X)
end = time.time()
print(end - start)
