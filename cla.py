import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# 学习资料 https://www.cnblogs.com/lsqin/p/9342935.html
# 学习资料 https://blog.csdn.net/csqazwsxedc/article/details/69690655

X =  [  # 训练集
        [1.0, 5.0, 9.0],
        [1.0, 11.0, 21.0],
        [1.0, 60.0, 70.0],
        [1.0, 7.0, 2.0],
        [1.0, 8.0, 3.0],
        [1.0, 11.0, 90.0],
        [1.0, 1.0, 1.0],
        [1.0, 11.0, 2.0],
        [1.0, 1.0, 2.0],
        [1.0, 51.0, 9.0],
        [1.0, 11.0, 11.0],
        [1.0, 60.0, 10.0],
        [1.0, 7.0, 21.0],
        [1.0, 8.0, 31.0],
        [1.0, 12.0, 90.0],
        [1.0, 12.0, 1.0],
        [1.0, 12.0, 2.0],
        [1.0, 1.0, 22.0],
        [1.0, 11.0, 2.0],
        [1.0, 11.0, 22.0],


        [1.0, 120.0, 211.0],
        [1.0, 142.0, 232.0],
        [1.0, 1341.0, 233.0],
        [1.0, 120.0, 211.0],
        [1.0, 155.0, 120.0],
        [1.0, 111.0, 355.0],
        [1.0, 122.0, 221.0],
        [1.0, 115.0, 122.0],
        [1.0, 111.0, 352.0],
        [1.0, 131.0, 345.0],
        [1.0, 130.0, 211.0],
        [1.0, 132.0, 232.0],
        [1.0, 1341.0, 233.0],
        [1.0, 130.0, 211.0],
        [1.0, 135.0, 120.0],
        [1.0, 131.0, 355.0],
        [1.0, 122.0, 231.0],
        [1.0, 115.0, 132.0],
        [1.0, 111.0, 332.0],
        [1.0, 131.0, 335.0],
    ]


y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# 线性归一化处理
# X[:, 1] = (X[:, 1] - (X[:, 1].min())) / (X[:, 1].max() - (X[:, 1].min()))
# X[:, 2] = (X[:, 2] - (X[:, 2].min())) / (X[:, 2].max() - (X[:, 2].min()))

# print((X[:,0].max()))
# print(X)

# print("学习率:" + str(lr))
# print("记录数:" + str(row))

ax_x = []
ax_y = []


def sigmoid(inX):  #sigmoid函数
    return 1.0/(1+np.exp(-inX))

# def gradient(X, y, theta):
#     return (X.T.dot(sigmoid(X.dot(theta)) - y)) / len(y)

# def j(X, y, theta):
#     return -(y.dot(np.log(sigmoid(X.dot(theta)))) + (1-y).dot(1-np.log(sigmoid(X.dot(theta))))) / len(y)


# def gra(j):
#     gra = 0.0
#     for num in range(0, row):
#         f = W[0]*X[num][0]+W[1]*X[num][1]+W[2]*X[num][2]
#         hhat = sigmoid(f)
#         gra += (hhat-Y[num])*X[num][j]
#     return gra




def gradAscent(dataMat, labelMat): #梯度上升求最优参数
    dataMatrix=np.mat(dataMat) #将读取的数据转换为矩阵
    classLabels=np.mat(labelMat).transpose() #将读取的数据转换为矩阵
    m,n = np.shape(dataMatrix)
    alpha = 0.001  #设置梯度的阀值，该值越大梯度上升幅度越大
    maxCycles = 5000 #设置迭代的次数，一般看实际数据进行设定，有些可能200次就够了
    weights = np.ones((n,1)) #设置初始的参数，并都赋默认值为1。注意这里权重以矩阵形式表示三个参数。
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (classLabels - h)     #求导后差值
        weights = weights + alpha * dataMatrix.transpose()* error #迭代更新权重
        # print(weights)
    return weights


weights=gradAscent(X,y)

PX=[[1.0,10.0,10.0]]
print (sigmoid(np.mat(PX)*weights)>0.5)

# model = LogisticRegression(verbose=1)
# model.fit(X, y)
# print (theta)
# print('逻辑回归模型:\n',model)


# PX=np.array([1.0,387.0,309.0])
# print(model.predict(PX.reshape(1, -1)))


# print(graAscent(X,y))
# best_theta = gd(X, y, theta)
#print(best_theta)

# px=np.array([1.0,2000.0,20000.0])
# print(model(px,best_theta))
# print(sigmoid(px.dot(best_theta)))

# PX=np.array([1.0,213.0,213.0])
# print(sigmoid(PX.T.dot(best_theta).sum()))

# for num in range(1, 10 + 1):
#     W[0]=W[0] - lr * gra(0)
#     W[1]=W[1] - lr * gra(1)
#     W[2]=W[2] - lr * gra(2)
#     C=mycost()
#     print("第" + str(num) + "次学习完")
#     print(W)
#     print(C)
#     ax_x.append(num)
#     ax_y.append(C)

# fig=plt.figure()
# ax=fig.add_subplot()
# ax.set(
#     xlim = [0, 500],
#     ylim = [0, 5000000],
#     title = "An Example Axes",
#     ylabel = "Y-Axis",
#     xlabel = "X-Axis",

# )
# ax.plot(ax_x, ax_y)
# plt.show()
