from mxnet import gluon
from mxnet.gluon import loss as gloss
from mxnet import init
from mxnet.gluon import nn
from mxnet.gluon import data as gdata
from mxnet import autograd, nd

num_inputs = 2
num_examples = 2000
true_w = [2, -3.4]
true_b = 4.2

features = nd.random.normal(scale=80, shape=(num_examples, num_inputs))
labels = nd.random.normal(scale=80, shape=(num_examples, num_inputs))
# labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
for n in range(num_examples):
    if(features[n].sum() > 100):
        labels[n][0] = 0.0
        labels[n][1] = 1.0
    else:
        labels[n][0] = 1.0
        labels[n][1] = 0.0
# labels += nd.random.normal(scale=0.01, shape=labels.shape)


batch_size = 200
# 将训练数据的特征和标签组合
dataset = gdata.ArrayDataset(features, labels)
# 随机读取小批量
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

for X, y in data_iter:
    print(X, y)
    break


net = nn.Sequential()
net.add(nn.Dense(2560, activation='sigmoid'),
        nn.Dense(2))

net.initialize(init.Normal(sigma=0.5))

# net = nn.Sequential()
# net.add(nn.Dense(1))
# net.initialize(init.Normal(sigma=0.01))
loss = gloss.L2Loss()  # 平方损失又称L2范数损失
# loss = gloss.SoftmaxCELoss()  # 平方损失又称L2范数损失
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.002})

num_epochs = 300
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))


# dense = net[0]
# print(true_w)
# print(dense.weight.data())
# print(true_b)
# print(dense.bias.data())
# true_b, dense.bias.data()

print(net(nd.array([[110.0, 111.9]])))
