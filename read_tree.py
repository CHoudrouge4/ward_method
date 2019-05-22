from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from numpy import *
import time

def read_file(filename):
    f = open(filename, "r")
    lines = f.readlines()
    n = len(lines)+1
    nb_clust = n
    #print(n)
    clusters = {3*i*n+1: i for i in range(n)}
    T = [[i,-1] for i in range(n)]
    for l in lines:
        words = l.split(";")
        res, c1, c2 = words[:3]
        idres_str = res.split(",")
        idres = int(idres_str[0])*3*n + int(idres_str[1])

        idc1_str = c1.split(",")
        idc1 = int(idc1_str[0])*3*n + int(idc1_str[1])

        idc2_str = c2.split(",")
        idc2 = int(idc2_str[0])*3*n + int(idc2_str[1])

        #print(idres_str, idc2_str, idc1_str)
        #print(idres, idc1, idc2)

        clusters[idres] = nb_clust
        T.append([clusters[idc1], clusters[idc2]])
        nb_clust+=1

    return T


def get_cluster(tree, index):
    if tree[index][0] == index:
        return [index]

    return get_cluster(tree,
                       tree[index][0])+get_cluster(tree,
                                                   tree[index][1])

def clusters(tree, k):
    nums = []
    i = 1
    while len(nums) != k:
        current = len(tree)-i
        if current in nums:
            nums.remove(current)
        c1 = tree[-i][0]
        c2 = tree[-i][1]
        nums.append(c1)
        if c1 != current:
            nums.append(c2)
        i+=1

    return [get_cluster(tree, nums[i]) for i in range(k)]

def get_dataset(name):
    from sklearn.preprocessing import scale
    data = []
    if name == "cancer":
        from sklearn.datasets import load_breast_cancer
        dataset = load_breast_cancer()
    elif name == "digits":
        from sklearn.datasets import load_digits
        dataset = load_digits()
    elif name == "iris":
        from sklearn.datasets import load_iris
        dataset = load_iris()
    elif name == "boston":
        from sklearn.datasets import load_boston
        dataset = load_boston()
    elif name == "KDD":
        from sklearn.datasets import fetch_kddcup99
        dataset = fetch_kddcup99(subset='SF')
        data = dataset.data[:2000, [0,2,3]]
    else:
        print("Unknown name of dataset")
        exit(-1)


    labels = dataset.target
    if data == []:
        data = scale(dataset.data)
        n_samples, n_features = data.shape
        n_elements = len(unique(labels))
    return data, n_elements, labels, len(set(labels))


# example
# t = [[0, -2], [1, -1], [2, -1], [3, -1], [4, -1], [0, 1], [2, 3], [5, 6], [7, 4]]

# print(clusters(t, 3))
from sklearn.datasets import load_iris
from sklearn.metrics.cluster import normalized_mutual_info_score

def convert(clusters, n):
    clustering_vect = [0]*n
    for i in range(len(clusters)):
        for p in clusters[i]:
            clustering_vect[p] = i
    return clustering_vect


#epsilon = 200
# numebr_of_trees = 16
# number_of_visited_leafs = 128
# #print("epsilon ", "0." + str(epsilon), numebr_of_trees, number_of_visited_leafs)
#
# #eps = [25, 50, 75, 85, 95, 200, 400];
# eps = [400]
# data_sets = ["iris", "cancer", "digits", "boston"]
# #data_sets = ["digits"]
# for e in eps:
#     output_file = 'result' + str(e) + '_' + str(numebr_of_trees) + '_' + str(number_of_visited_leafs) + '.txt'
#     with open(output_file, 'w') as file:
#         file.write('epsilon 0.' + str(e) + ' ' + str(numebr_of_trees) + ' ' + str(number_of_visited_leafs) + '\n')
#         for name in data_sets:
#             data, n, labels, k = get_dataset(name)
#             file_name = './data/' + name + str(e) + '_' + str(numebr_of_trees) + '_' + str(number_of_visited_leafs) + ".out"
#             T = read_file(file_name)
#             clust = clusters(T, k)
#             file.write('Algo ' + str(normalized_mutual_info_score(convert(clust, len(labels)), labels)) + '\n')
#
#         for name in data_sets:
#             data, n, labels, k = get_dataset(name)
#             ward = AgglomerativeClustering(n_clusters=k, linkage='ward', connectivity=None)
#             clustering = ward.fit(data)
#             clust = clustering.labels_
#             file.write('Ward ' + str(normalized_mutual_info_score(clust, labels)) + '\n')
#
#         for name in data_sets:
#             data, n, labels, k = get_dataset(name)
#             ward = AgglomerativeClustering(n_clusters=k, linkage='average', connectivity=None)
#             clustering = ward.fit(data)
#             clust = clustering.labels_
#             file.write('Average ' + str(normalized_mutual_info_score(clust, labels)) + '\n')
#
#         for name in data_sets:
#             data, n, labels, k = get_dataset(name)
#             ward = AgglomerativeClustering(n_clusters=k, linkage='single', connectivity=None)
#             clustering = ward.fit(data)
#             clust = clustering.labels_
#             file.write('Single ' + str(normalized_mutual_info_score(clust, labels)) + '\n')



def readFILE(file_name):
    with open(file_name) as f:
        content = f.readline()
        content = content.split(' ')
        n = int(content[0])
        d = int(content[1])
        #k = int(content[2])
        X = zeros((n, d))
        for i in range(n):
            content = f.readline()
            content = content.split(' ')
            for j in range(d):
                X[i, j] = float(content[j])
        return X



  #
  # std::vector<int> trees = {4 , 16};
  # std::vector<int> leaves = {5, 128};
  # std::vector<float> epsilons = {0.5, 1, 7};

trees = [4, 16]
leaves = [5, 128]
epsilons = [50, 200, 500]
d = 2
k = 10
ward = AgglomerativeClustering(n_clusters=k, linkage='ward', connectivity=None)
with open('ward_accuracy1.txt', 'w') as f:
    for e in epsilons:
        for t in trees:
            for l in leaves:
                for i in range(1000, 10000, 500):
                    for j in range(10):
                        data_file = './data/data' + str(i) + '_' + str(j) + '_' + str(d) + '_' + str(k) + '.in'
                        data = readFILE(data_file)
                        start = time.time()
                        clustering = ward.fit(data)
                        end = time.time()
                        labels = clustering.labels_
                        res_file = 'data' + str(i) + '_' + str(j) + '_' + str(d) + '_' + str(k) + '_' + str(e) + '_' + str(t) + '_' + str(l) + '.out'
                        T = read_file(res_file)
                        clust = clusters(T, k);
                        acc = normalized_mutual_info_score(convert(clust, len(labels)), labels)
                        f.write(str(end - start) + ' ' + str(acc) + ' ')
                    f.write('\n')

#
# data = readFILE("./data/data10000_50_50.in")
# clustering = ward.fit(data)
# #random.shuffle(data)
# labels = clustering.labels_
# file_name = './data/data10000_50_50200_8_200.out'
# T = read_file(file_name)
# clust = clusters(T, 100)
# print(normalized_mutual_info_score(convert(clust, len(labels)), labels))
