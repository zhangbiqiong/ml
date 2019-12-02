from mxnet import gluon
from mxnet.gluon import loss as gloss
from mxnet import init
from mxnet.gluon import nn
from mxnet.gluon import data as gdata
from mxnet import autograd, nd

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

features = nd.random.normal(scale=80, shape=(num_examples, num_inputs))

labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
for n in range(num_examples):
    if features[n].sum() < 0:
        labels[n] = 0
    elif features[n].sum() < 100:
        labels[n] = 1
    else:
        labels[n] = 2


batch_size = 100
# 将训练数据的特征和标签组合
dataset = gdata.ArrayDataset(features, labels)
# 随机读取小批量
train_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
test_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

for X, y in train_iter:
    print(X, y)
    break

# sigmoid relu
net = nn.Sequential()
net.add(nn.Dense(1024, activation="sigmoid"), nn.Dropout(0.1),nn.Dense(3))

net.initialize(init.Normal(sigma=0.5))


loss = gloss.SoftmaxCELoss()  # 平方损失又称L2范数损失
# trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.015})
trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.015, 'wd': 1})


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y = y.astype("float32")
        acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()
        n += y.size
    return acc_sum / n




# 本函数已保存在d2lzh包中方便以后使用
def train_ch3(
    net,
    train_iter,
    test_iter,
    loss,
    num_epochs,
    batch_size,
    params=None,
    lr=None,
    trainer=None,
):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            trainer.step(batch_size)  # “softmax回归的简洁实现”一节将用到
            y = y.astype("float32")
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net)
        print(
            "epoch %d, loss %.4f, train acc %.3f, test acc %.3f"
            % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc)
        )

print("init acc %.3f" % evaluate_accuracy(test_iter, net))
num_epochs = 5
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)


def test(X):
    y_hat=net(nd.array(X))
    print(y_hat.argmax(axis=1))

test([[-1,-2]])
test([[-11,1]])
test([[-12,2]])
test([[-34,11]])
test([[11,-33]])
print("---------------------------")
test([[-9,61]])
test([[22,12]])
test([[11,8]])
test([[51,-8]])
test([[25,19]])
print("---------------------------")
test([[59,61]])
test([[72,12]])
test([[91,28]])
test([[51,58]])
test([[65,69]])

# print(net(nd.array([[-10, -1]]))[0].argmax(axis=1))
# print(net(nd.array([[22, 22]]))[0].argmax(axis=1))
# print(net(nd.array([[81, 99]]))[0].argmax(axis=1))
# print(net(nd.array([[-10, -1]])))
# print(net(nd.array([[-1, -1]])))
# print(net(nd.array([[22, 22]])))
# print(net(nd.array([[22, 2]])))
# print(net(nd.array([[77, 77]])))
# print(net(nd.array([[177, 77]])))
