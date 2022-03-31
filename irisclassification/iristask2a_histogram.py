import numpy as np
import matplotlib.pyplot as plt

# Reading the iris.data file, and extracting the data to a list
file = open("iris.data", "r")
data_arr=[]
for line in file:
    if line!="\n":
        data_subarr = line.split(",")
        data_subarr[0]=float(data_subarr[0])
        data_subarr[1]=float(data_subarr[1])
        data_subarr[2]=float(data_subarr[2])
        data_subarr[3]=float(data_subarr[3])
        data_subarr[4]=data_subarr[4].replace('\n','')
        # print(data_subarr)
        data_arr.append(data_subarr)
file.close()
# print(data_arr)
data_arr_np=np.array(data_arr)


sepal_length_arr=[]
sepal_width_arr=[]
petal_length_arr=[]
petal_width_arr=[]

for iris in data_arr:
    sepal_length_arr.append(iris[0])
    sepal_width_arr.append(iris[1])
    petal_length_arr.append(iris[2])
    petal_width_arr.append(iris[3])


# print(sepal_length_arr)
# print(sepal_width_arr)
# print(petal_length_arr)
# print(petal_width_arr)


# Produces a histogram of the sepal length for each class
plt.style.use('seaborn-deep')
sepal_length_n, sepal_length_bins, sepal_length_patches = plt.hist([sepal_length_arr[0:49],sepal_length_arr[50:99], sepal_length_arr[100:149]], 25, stacked=True, label=['Setosa','Versicolor','Virginica'], density=True, alpha=0.55, edgecolor='black', linewidth=1.2)

plt.tight_layout()
plt.legend(prop={'size':16})
plt.xticks(size = 14)
plt.yticks(size = 14)
plt.xlabel('Sepal length [cm]', fontsize=16)
plt.ylabel('Probability density', fontsize=16)
plt.title('Histogram (stacked) of the sepal length for each flower', fontsize=22)
plt.grid(True)
plt.savefig('task2a_sepal_length_savefig.png', bbox_inches='tight')
plt.show()


# Produces a histogram of the sepal width for each class
plt.style.use('seaborn-deep')
sepal_width_n, sepal_width_bins, sepal_width_patches = plt.hist([sepal_width_arr[0:49],sepal_width_arr[50:99], sepal_width_arr[100:149]], 25, stacked=True, label=['Setosa','Versicolor','Virginica'], density=True, alpha=0.55, edgecolor='black', linewidth=1.2)

plt.tight_layout()
plt.legend(prop={'size':16})
plt.xticks(size = 14)
plt.yticks(size = 14)
plt.xlabel('Sepal width [cm]', fontsize=16)
plt.ylabel('Probability density', fontsize=16)
plt.title('Histogram (stacked) of the sepal width for each flower', fontsize=22)
plt.grid(True)
plt.savefig('task2a_sepal_width_savefig.png', bbox_inches='tight')
plt.show()


# Produces a histogram of the petal length for each class
plt.style.use('seaborn-deep')
petal_length_n, petal_length_bins, petal_length_patches = plt.hist([petal_length_arr[0:49],petal_length_arr[50:99], petal_length_arr[100:149]], 25, stacked=True, label=['Setosa','Versicolor','Virginica'], density=True, alpha=0.55, edgecolor='black', linewidth=1.2)

plt.tight_layout()
plt.legend(prop={'size':16})
plt.xticks(size = 14)
plt.yticks(size = 14)
plt.xlabel('Petal length [cm]', fontsize=16)
plt.ylabel('Probability density', fontsize=16)
plt.title('Histogram (stacked) of the petal length for each flower', fontsize=22)
plt.grid(True)
plt.savefig('task2a_petal_length_savefig.png', bbox_inches='tight')
plt.show()


# Produces a histogram of the petal width for each class
plt.style.use('seaborn-deep')
petal_width_n, petal_width_bins, petal_width_patches = plt.hist([petal_width_arr[0:49],petal_width_arr[50:99], petal_width_arr[100:149]], 25, stacked=True, label=['Setosa','Versicolor','Virginica'], density=True, alpha=0.55, edgecolor='black', linewidth=1.2)

plt.tight_layout()
plt.legend(prop={'size':16})
plt.xticks(size = 14)
plt.yticks(size = 14)
plt.xlabel('Petal width [cm]', fontsize=16)
plt.ylabel('Probability density', fontsize=16)
plt.title('Histogram (stacked) of the petal width for each flower', fontsize=22)
plt.grid(True)
plt.savefig('task2a_petal_width_savefig.png', bbox_inches='tight')
plt.show()

