
def read_file(filename):
    f = open(filename, "r")
    lines = f.readlines()
    n = len(lines)+1
    nb_clust = n
    print(n)
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


# example
# t = [[0, -2], [1, -1], [2, -1], [3, -1], [4, -1], [0, 1], [2, 3], [5, 6], [7, 4]]



t = read_file("output.in")

print(clusters(t, 3))
