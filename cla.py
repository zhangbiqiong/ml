import numpy as np
import matplotlib.pyplot as plt

# 学习资料 https://www.jianshu.com/p/c53509ed9b52
# y=w0x0+w1x1+w2x2
theta = np.array([0.0, 0.0, 0.0])  # 权重，也就是训练的目标 构造依据 [0.1,2,3]
X = np.array(
    [  # 训练集
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
)


y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# 线性归一化处理
# X[:, 1] = (X[:, 1] - (X[:, 1].min())) / (X[:, 1].max() - (X[:, 1].min()))
# X[:, 2] = (X[:, 2] - (X[:, 2].min())) / (X[:, 2].max() - (X[:, 2].min()))

# print((X[:,0].max()))
print(X)

# print("学习率:" + str(lr))
# print("记录数:" + str(row))

ax_x = []
ax_y = []


def model(x,theta):
    return sigmoid(np.dot(x,theta.T))

def sigmoid(t):
    return 1. / (1. + np.exp(-t))


def gradient(X, y, theta):
    return (X.T.dot(sigmoid(X.dot(theta)) - y)) / len(y)
    # return ((sigmoid(X.dot(theta)) - y)* X[:, j]).sum()
    # return (X.T.dot(sigmoid(X.dot(theta)) - y)) / len(y)
    # grad=np.zeros(theta.shape)
    # error=(model(X,theta)-y).ravel()
    # for j in range(len(theta.ravel())):
    #     term=np.multiply(error,X[:,j])
    #     grad[0,j]=np.sum(term)/len(X)
    # return grad[0,j]


def j(X, y, theta):
    return -(y.dot(np.log(sigmoid(X.dot(theta)))) + (1-y).dot(1-np.log(sigmoid(X.dot(theta))))) / len(y)


# def gra(j):
#     gra = 0.0
#     for num in range(0, row):
#         f = W[0]*X[num][0]+W[1]*X[num][1]+W[2]*X[num][2]
#         hhat = sigmoid(f)
#         gra += (hhat-Y[num])*X[num][j]
#     return gra


def gd(X, y, theta, alpha=0.0001, n_iters=100):
    cur_iters = 0
    while cur_iters < n_iters:
        theta = theta - alpha*gradient(X, y, theta)
        # grad = gradient(X, y, theta,j)
        # theta[0]=theta[0] -alpha* gradient(X, y, theta,0)
        # theta[1]=theta[1] -alpha* gradient(X, y, theta,1)
        # theta[2]=theta[2] -alpha* gradient(X, y, theta,2)
        print(theta)
        print(j(X, y, theta))
        cur_iters += 1
    return theta


best_theta = gd(X, y, theta)
#print(best_theta)

px=np.array([1.0,2000.0,20000.0])
print(model(px,best_theta))
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
