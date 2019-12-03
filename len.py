import sys
import time

import mxnet as mx
from IPython import display
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn
from pylab import mpl, plt


batch_size = 256

def load_data_fashion_mnist(batch_size):
    
    transformer = gdata.vision.transforms.ToTensor()
    if sys.platform.startswith("win"):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4

    mnist_train = gdata.vision.FashionMNIST(train=True)
    mnist_test = gdata.vision.FashionMNIST(train=False)

    train_iter = gdata.DataLoader(
        mnist_train.transform_first(transformer),
        batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_iter = gdata.DataLoader(
        mnist_test.transform_first(transformer),
        batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_iter, test_iter

# sigmoid relu

net = nn.Sequential()
net.add(
    nn.Conv2D(channels=6, kernel_size=5, activation="sigmoid"),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Conv2D(channels=16, kernel_size=5, activation="sigmoid"),
    nn.MaxPool2D(pool_size=2, strides=2),
    # Dense会默认将(批量大小, 通道, 高, 宽)形状的输入转换成
    # (批量大小, 通道 * 高 * 宽)形状的输入
    nn.Dense(120, activation="sigmoid"),
    nn.Dense(84, activation="sigmoid"),
    nn.Dense(10),
)

X = nd.random.uniform(shape=(1, 1, 28, 28))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, "output shape:\t", X.shape)



train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)


def evaluate_accuracy(data_iter, net, ctx):
    acc_sum, n = nd.array([0], ctx=ctx), 0
    for X, y in data_iter:
        # 如果ctx代表GPU及相应的显存，将数据复制到显存上
        X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype("float32")
        acc_sum += (net(X).argmax(axis=1) == y).sum()
        n += y.size
    return acc_sum.asscalar() / n


def train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs):
    print("training on", ctx)
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            trainer.step(batch_size)
            y = y.astype("float32")
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print(
            "epoch %d, loss %.4f, train acc %.3f, test acc %.3f, "
            "time %.1f sec"
            % (
                epoch + 1,
                train_l_sum / n,
                train_acc_sum / n,
                test_acc,
                time.time() - start,
            )
        )


def try_gpu():  # 本函数已保存在d2lzh包中方便以后使用
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx


ctx = try_gpu()


lr, num_epochs = 0.9, 1
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": lr})
train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)


def get_fashion_mnist_labels(labels):
    text_labels = [
        "t-shirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle boot",
    ]
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):

    # plt.set_matplotlib_formats('svg')
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


def printpic():
    transformer = gdata.vision.transforms.ToTensor()
    mnist_train = gdata.vision.FashionMNIST(train=True)
    X, y = mnist_train[110:116]
    X1=transformer(X)
    y_hat = net(X1).argmax(axis=1)
    show_fashion_mnist(X, get_fashion_mnist_labels(y_hat.asnumpy()))


printpic()
