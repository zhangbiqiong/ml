import numpy
from numpy import *
import numpy as np
import matplotlib.pyplot as plt

# y=w0x0+w1x1+w2x2
W = array([0.0, 0.0, 0.0])  # 权重，也就是训练的目标 构造依据 [0.1,2,3]
X = array(
    [  # 训练集
        [1.0, 5.0, 9.0],
        [1.0, 60.0, 10.0],
        [1.0, 7.0, 2.0],
        [1.0, 8.0, 3.0],
        [1.0, 11.0, 90.0],
        [1.0, 155.0, 1.0],
        [1.0, 1.0, 21.0],
        [1.0, 1.0, 355.0],
        [1.0, 11.0, 2.0],
        [1.0, 1.0, 2.0],
        [1.0, 120.0, 2.0],
        [1.0, 1.0, 22.0],
        [1.0, 111.0, 2.0],
        [1.0, 120.0, 211.0],
        [1.0, 11.0, 221.0],
    ]
)
lr = 0.8
row = X[:, 0].size
Y = np.dot(X, array([10.0, 20.0, 30.0])) + np.random.randint(-1, 1, size=(1, row))


# 线性归一化处理
X[:, 1] = (X[:, 1] - (X[:, 1].min())) / (X[:, 1].max() - (X[:, 1].min()))
X[:, 2] = (X[:, 2] - (X[:, 2].min())) / (X[:, 2].max() - (X[:, 2].min()))


# print((X[:,0].max()))
print(X)


print("学习率:" + str(lr))
print("记录数:" + str(row))

ax_x = []
ax_y = []

for num in range(1, 500 + 1):
    W[0] = W[0] - lr * (((np.dot(X, W) - Y)).sum() / row)
    W[1] = W[1] - lr * (((np.dot(X, W) - Y) * X[:, 1]).sum() / row)
    W[2] = W[2] - lr * (((np.dot(X, W) - Y) * X[:, 2]).sum() / row)
    C = ((np.dot(X, W) - Y) ** 2).sum() / (2 * row)
    print("第" + str(num) + "次学习完")
    print(W)
    print(C)
    ax_x.append(num)
    ax_y.append(C)

fig = plt.figure()
ax = fig.add_subplot()
ax.set(
    xlim=[0, 500],
    ylim=[0, 5000000],
    title="An Example Axes",
    ylabel="Y-Axis",
    xlabel="X-Axis",
)
ax.plot(ax_x, ax_y)
plt.show()

