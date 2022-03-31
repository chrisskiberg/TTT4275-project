import numpy as np
import matplotlib.pyplot as plt
# Reading from the iris.data file, and extracting the data to a list
# https://www.machinelearningplus.com/plots/matplotlib-histogram-python-examples/
# https://www.nature.com/srep/author-instructions/submission-guidelines#general-figure

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

# Selecting training and testing sets
data_setosa_training_arr=data_arr[0:29]
data_versicolor_training_arr=data_arr[50:79]
data_virginica_training_arr=data_arr[100:129]
# print(data_setosa_training_arr)
# print(data_versicolor_training_arr)
# print(data_virginica_training_arr)

data_setosa_testing_arr=data_arr[30:49]
data_versicolor_testing_arr=data_arr[80:99]
data_virginica_testing_arr=data_arr[130:149]
# print(data_setosa_testing_arr)
# print(data_versicolor_testing_arr)
# print(data_virginica_testing_arr)

#Number of Attributes: 4 numeric, predictive attributes and the class\n    :Attribute Information:\n        - sepal length in cm\n        - sepal width in cm\n        - petal length in cm\n        
# - petal width in cm\n        - class:\n                - Iris-Setosa\n                - Iris-Versicolour\n                - Iris-Virginica\n                \n    
# sepal length in cm\n        - sepal width in cm\n        - petal length in cm\n         - petal width in cm


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

# for setosa in data_arr[0:49]:
# for versicolor in data_arr[50:99]:
# for virginica in data_arr[100:149]:

# # Compute frequency and bins
# frequency, bins = np.histogram(sepal_length_arr, bins=50)

# # Pretty Print
# for b, f in zip(bins[1:], frequency):
#     print(round(b, 1), ' '.join(np.repeat('*', f)))

# https://www.originlab.com/doc/Tutorials/Customize-Color-Transparancy


# the histogram of the sepal length for each class
plt.style.use('seaborn-deep')
sepal_length_n, sepal_length_bins, sepal_length_patches = plt.hist([sepal_length_arr[0:49],sepal_length_arr[50:99], sepal_length_arr[100:149]], 25, stacked=True, label=['Setosa','Versicolor','Virginica'], density=True, alpha=0.55, edgecolor='black', linewidth=1.2)
# sepal_length_setosa_n, sepal_length_setosa_bins, sepal_length_setosa_patches = plt.hist(sepal_length_arr[0:49], 10, stacked=True, label='Setosa', density=True, facecolor='navy', alpha=0.55)
# sepal_length_versicolor_n, sepal_length_versicolor_bins, sepal_length_versicolor_patches = plt.hist(sepal_length_arr[50:99], 10, stacked=True, label='Versicolor', density=True, facecolor='orange', alpha=0.55)
# sepal_length_virginica_n, sepal_length_virginica_bins, sepal_length_virginica_patches = plt.hist(sepal_length_arr[100:149], 10, stacked=True, label='Virginica', density=True, facecolor='seagreen', alpha=0.55)

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

# the histogram of the sepal width for each class
plt.style.use('seaborn-deep')
sepal_width_n, sepal_width_bins, sepal_width_patches = plt.hist([sepal_width_arr[0:49],sepal_width_arr[50:99], sepal_width_arr[100:149]], 25, stacked=True, label=['Setosa','Versicolor','Virginica'], density=True, alpha=0.55, edgecolor='black', linewidth=1.2)
# sepal_width_setosa_n, sepal_width_setosa_bins, sepal_width_setosa_patches = plt.hist(sepal_width_arr[0:49], 10, stacked=True, label='Setosa', density=True, facecolor='navy', alpha=0.55)
# sepal_width_versicolor_n, sepal_width_versicolor_bins, sepal_width_versicolor_patches = plt.hist(sepal_width_arr[50:99], 10, stacked=True, label='Versicolor', density=True, facecolor='orange', alpha=0.55)
# sepal_width_virginica_n, sepal_width_virginica_bins, sepal_width_virginica_patches = plt.hist(sepal_width_arr[100:149], 10, stacked=True, label='Virginica', density=True, facecolor='seagreen', alpha=0.55)

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

# the histogram of the petal length for each class
plt.style.use('seaborn-deep')
petal_length_n, petal_length_bins, petal_length_patches = plt.hist([petal_length_arr[0:49],petal_length_arr[50:99], petal_length_arr[100:149]], 25, stacked=True, label=['Setosa','Versicolor','Virginica'], density=True, alpha=0.55, edgecolor='black', linewidth=1.2)
# petal_length_setosa_n, petal_length_setosa_bins, petal_length_setosa_patches = plt.hist(petal_length_arr[0:49], 10, stacked=True, label='Setosa', density=True, facecolor='navy', alpha=0.55)
# petal_length_versicolor_n, petal_length_versicolor_bins, petal_length_versicolor_patches = plt.hist(petal_length_arr[50:99], 10, stacked=True, label='Versicolor', density=True, facecolor='orange', alpha=0.55)
# petal_length_virginica_n, petal_length_virginica_bins, petal_length_virginica_patches = plt.hist(petal_length_arr[100:149], 10, stacked=True, label='Virginica', density=True, facecolor='seagreen', alpha=0.55)

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



# the histogram of the petal width for each class
plt.style.use('seaborn-deep')
petal_width_n, petal_width_bins, petal_width_patches = plt.hist([petal_width_arr[0:49],petal_width_arr[50:99], petal_width_arr[100:149]], 25, stacked=True, label=['Setosa','Versicolor','Virginica'], density=True, alpha=0.55, edgecolor='black', linewidth=1.2)
# petal_width_setosa_n, petal_width_setosa_bins, petal_width_setosa_patches = plt.hist(petal_width_arr[0:49], 10, stacked=True, label='Setosa', density=True, facecolor='navy', alpha=0.55)
# petal_width_versicolor_n, petal_width_versicolor_bins, petal_width_versicolor_patches = plt.hist(petal_width_arr[50:99], 10, stacked=True, label='Versicolor', density=True, facecolor='orange', alpha=0.55)
# petal_width_virginica_n, petal_width_virginica_bins, petal_width_virginica_patches = plt.hist(petal_width_arr[100:149], 10, stacked=True, label='Virginica', density=True, facecolor='seagreen', alpha=0.55)

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

