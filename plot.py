import matplotlib.pyplot as plt

def get_data(file_name):
    content = []
    with open(file_name) as f:
        content = f.readlines() # read file line by line
    content = [x.strip() for x in content]
    return content

def get_info(content):
    sizes = []
    algo = []
    ward = []
    for e in range(len(content)):
        c = content[e].split()
        if e % 3 == 0:
            sizes.append(int(c[4]))
        if e % 3 == 1:
            algo.append(float(c[1]))
        if e % 3 == 2:
            ward.append(float(c[1]))
    return sizes, algo, ward

content = get_data('result100_25_128.txt')
sizes, algo, ward = get_info(content)

plt.plot(sizes, algo, label = 'Ward-approx')
plt.plot(sizes, ward, label = 'Ward')
plt.title('Accuracy of Ward-approx (e = 10, T = 25, L = 128) vs Ward')
plt.legend(loc = 'best')
plt.xlabel('Number of Points')
plt.ylabel('Accuracy NMI')
plt.show()
