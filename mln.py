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
# features = nd.random.randint(10, 80, shape=(num_examples, num_inputs))
# labels = nd.random.normal(scale=80, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
for n in range(num_examples):
    if(features[n].sum() < 0):
        labels[n] = 0
    elif(features[n].sum() < 100):
        labels[n] = 1
    else:
        labels[n] = 2
# labels += nd.random.normal(scale=0.01, shape=labels.shape)


batch_size = 200
# 将训练数据的特征和标签组合
dataset = gdata.ArrayDataset(features, labels)
# 随机读取小批量
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

for X, y in data_iter:
    print(X, y)
    break

# sigmoid relu
net = nn.Sequential()
net.add(nn.Dense(512, activation='sigmoid'),
        nn.Dense(3))

net.initialize(init.Normal(sigma=0.5))

# net = nn.Sequential()
# net.add(nn.Dense(1))
# net.initialize(init.Normal(sigma=0.01))
# loss = gloss.L2Loss()  # 平方损失又称L2范数损失
loss = gloss.SoftmaxCELoss()  # 平方损失又称L2范数损失
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})

num_epochs = 30
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))

print(net(nd.array([[-10, -1]])))
print(net(nd.array([[-1, -1]])))
print(net(nd.array([[22, 22]])))
print(net(nd.array([[22, 2]])))
print(net(nd.array([[77, 77]])))
print(net(nd.array([[177, 77]])))
