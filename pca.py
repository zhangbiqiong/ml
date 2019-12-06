#coding=utf-8


##用sklearn的PCA
from sklearn.decomposition import PCA
import numpy as np
X = np.array([
    [32.0,33.0],
    [36.0,44.0],
    [16.0,4.0],
    [16.0,44.0],
    [3.0,14.0],
    [22.0,59.0],
    [36.0,54.0],
    [56.0,4.0],
    [56.0,14.0],
    [9.0,14.0],
    [32.0,3.0],
    [36.0,4.0],
    [1.0,4.0],
    [16.0,4.0],
    [33.0,14.0],
    [23.0,59.0],
    [33.0,54.0],
    [56.0,34.0],
    [56.0,34.0],
    [9.0,34.0],

    [72.0,73.0],
    [76.0,74.0],
    [86.0,84.0],
    [86.0,64.0],
    [86.0,74.0],
    [92.0,73.0],
    [60.0,74.0],
    [86.0,64.0],
    [96.0,64.0],
    [86.0,60.0],

    [92.0,73.0],
    [96.0,74.0],
    [89.0,84.0],
    [89.0,64.0],
    [89.0,79.0],
    [92.0,79.0],
    [60.0,79.0],
    [86.0,69.0],
    [96.0,66.0],
    [86.0,66.0],
    [92.0,66.0],
    [96.0,77.0],
    [89.0,77.0],
    [77.0,64.0],
    [77.0,79.0],
    [77.0,79.0],
    [77.0,79.0],
    [86.0,88.0],
    [96.0,88.0],
    [86.0,88.0],   
])

pca=PCA(n_components=1) 
pca.fit (X)
print(pca.transform(X))


import matplotlib.pyplot as plt

# dataSet=np.asarray(dataSet)
x_values = X[:,0]
y_values = X[:,1]
plt.scatter(x_values,y_values)
plt.show()
