#coding=utf-8
'''
Created on 2018年7月17日

@author: Administrator
'''
# k-means 算法python实现

import numpy as np

def distEclud(vecA, vecB):  #定义一个欧式距离的函数  
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

# print('----test_distEclud-----')
# vecA, vecB = np.array([2,1]), np.array([3,1])
# distance = distEclud(vecA, vecB)
# print(distance) # 1.0 计算两点之间的距离

def randCent(dataSet, k):  #第一个中心点初始化
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros([k, n]))  #创建 k 行 n列的全为0 的矩阵
    for j in range(n):
        minj = np.min(dataSet[:,j]) #获得第j 列的最小值
        rangej = float(np.max(dataSet[:,j]) - minj)     #得到最大值与最小值之间的范围
        #获得输出为 K 行 1 列的数据，并且使其在数据集范围内
        centroids[:,j] = np.mat(minj + rangej * np.random.rand(k, 1))   
    return centroids


# print('----test_randCent-----')
# dataSet1 = np.array([[1,2],[3,6],[8,10],[12,23],[10,11],[13,18]])
# print(dataSet1[1,:])
# r = randCent(dataSet1, 2)
# print(r)
# [[ 8.83544015 16.75467081]
#  [ 2.85688493  4.4799291 ]]

np.random.seed(666) #定义一个随机种子
rand_num = np.random.rand(3,1)  #输出为3行1 列,随机数在 0 到 1 之间
print(rand_num)
# [[0.70043712]
#  [0.84418664]
#  [0.67651434]]
test = np.mat(np.zeros([3,2]))  #此处的 zeros 函数内的矩阵形式需要加中括号 []
print(test)  
# [[0. 0.]    #打印出 3行 2列的矩阵
#  [0. 0.]
#  [0. 0.]] 

#参数： dataSet 样本点， K 簇的个数
#disMeans 距离， 默认使用欧式距离， createCent 初始中心点的选取
def KMeans(dataSet, k, distMeans= distEclud, createCent= randCent):
    m = np.shape(dataSet)[0]    #得到行数，即为样本数
    clusterAssement = np.mat(np.zeros([m,2]))   #创建 m 行 2 列的矩阵
    centroids = createCent(dataSet, k)      #初始化 k 个中心点
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf   #初始设置值为无穷大
            minIndex = -1
            for j in range(k):
                #  j循环，先计算 k个中心点到1 个样本的距离，在进行i循环，计算得到k个中心点到全部样本点的距离
                distJ = distMeans(centroids[j,:], dataSet[i,:])
                if distJ <  minDist:
                    minDist = distJ #更新 最小的距离
                    minIndex = j 
            if clusterAssement[i,0] != minIndex:    #如果中心点不变化的时候， 则终止循环
                clusterChanged = True 
            clusterAssement[i,:] = minIndex, minDist**2 #将 index，k值中心点 和  最小距离存入到数组中
        print(centroids)
        
        #更换中心点的位置
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssement[:,0].A == cent)[0]] #分别找到属于k类的数据
            centroids[cent,:] = np.mean(ptsInClust, axis = 0)   #得到更新后的中心点
    return centroids, clusterAssement 
                          

# print('------test-----')
# demo_a = np.array([[1,0],[0,2],[0,0]])
# non_a = np.nonzero(demo_a)
# print(demo_a)
# # [[1 0]
# #  [0 2]
# #  [0 0]]
# print(non_a)
# # 输出的第一行为 行数， 第二行为列数,意思为   1行1列的数 和2行2列的数 是非0数
# # (array([0, 1], dtype=int64), array([0, 1], dtype=int64))

# demo_a1 = np.array([1,2,0,0,1]) #当只有一行时
# non_a1 = np.nonzero(demo_a1)
# print(non_a1)   # (array([0, 1, 4], dtype=int64),)  

# a1 = np.inf > 100000
# print(a1)       #  True   inf 是无穷大

print('---------- test KMeans ---------')
dataSet = np.mat([
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
    [96.0,69.0],
    [86.0,69.0],
                    ])
print(dataSet)
center, cluster = KMeans(dataSet, 2)
print('--center--')
print(center)
# [[-1.05990877 -2.0619207 ]
#  [-0.03469197  2.95415497]]
print('--cluster--')
# print(cluster)



import matplotlib.pyplot as plt

dataSet=np.asarray(dataSet)
x_values = dataSet[:,0]
y_values = dataSet[:,1]

center=np.asarray(center)
c_x_values = center[:,0]
c_y_values = center[:,1]

plt.scatter(x_values,y_values)
plt.scatter(c_x_values,c_y_values)

plt.show()
