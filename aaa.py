import d2lzh as d2l
import math
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn, rnn
import time
import random

def load_data_alarm():
    """Load the Jay Chou lyric data set (available in the Chinese book)."""
    # with zipfile.ZipFile('../data/jaychou_lyrics.txt.zip') as zin:
    #     with zin.open('jaychou_lyrics.txt') as f:
    #         corpus_chars = f.read().decode('utf-8')
    # corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    # corpus_chars = corpus_chars[0:10000]
    ALARM_MAX = 9
    lines = []
    for n in range(2000):

        word0 = random.randint(1, ALARM_MAX)
        word1 = random.randint(1, ALARM_MAX)
        word2 = random.randint(1, ALARM_MAX)
        word3 = random.randint(1, ALARM_MAX)
        word4 = random.randint(1, ALARM_MAX)
        word5 = random.randint(1, ALARM_MAX)
        word6 = random.randint(1, ALARM_MAX)
        word7 = random.randint(1, ALARM_MAX)
        word8 = random.randint(1, ALARM_MAX)
        word9 = random.randint(1, ALARM_MAX)
        word10 = random.randint(1, ALARM_MAX)
        word11 = random.randint(1, ALARM_MAX)
        word12 = random.randint(1, ALARM_MAX)
        word13 = random.randint(1, ALARM_MAX)
        word14 = random.randint(1, ALARM_MAX)
        word15 = random.randint(1, ALARM_MAX)
        word16 = random.randint(1, ALARM_MAX)
        word17 = random.randint(1, ALARM_MAX)
        word18 = random.randint(1, ALARM_MAX)
        word19 = random.randint(1, ALARM_MAX)

        line = [
            str(word0),
            str(word1),
            str(word2),
            str(word3),
            str(word4),
            str(word5),
            str(word6),
            str(word7),
            str(word8),
            str(word9),
            str(word10),
            str(word11),
            str(word12),
            str(word13),
            str(word14),
            str(word15),
            str(word16),
            str(word17),
            str(word18),
            str(word19),
        ]
        if line.count("2") != 0 and line.count("4") != 0:
            if line.index("2") > line.index("4"):
                line[line.index("2")] = "4"
                line[line.index("4")] = "2"
                print(line)
        # if(line.count('11')!=0 and line.count('33')!=0):
        #     if(line.index('11')>line.index('33')):
        #         line[line.index('11')]='33'
        #         line[line.index('33')]='11'
        #         print(line)

        lines += line

    idx_to_char = list(set(lines))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in lines]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


# (corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_alarm()
num_hiddens = 256
rnn_layer = rnn.RNN(num_hiddens)
rnn_layer.initialize()

batch_size = 2
state = rnn_layer.begin_state(batch_size=batch_size)
print(state[0].shape)

num_steps = 35
X = nd.random.uniform(shape=(num_steps, batch_size, vocab_size))
Y, state_new = rnn_layer(X, state)
print(Y.shape)
print(len(state_new))
print(state_new[0].shape)

# 本类已保存在d2lzh包中方便以后使用
class RNNModel(nn.Block):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        # 将输入转置成(num_steps, batch_size)后获取one-hot向量表示
        X = nd.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        output = self.dense(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


# 本函数已保存在d2lzh包中方便以后使用
def predict_rnn_gluon(
    prefix, num_chars, model, vocab_size, ctx, idx_to_char, char_to_idx
):
    # 使用model的成员函数来初始化隐藏状态
    state = model.begin_state(batch_size=1, ctx=ctx)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = nd.array([output[-1]], ctx=ctx).reshape((1, 1))
        (Y, state) = model(X, state)  # 前向计算不需要传入模型参数
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(axis=1).asscalar()))
    return ",".join([idx_to_char[i] for i in output])


ctx = d2l.try_gpu()
model = RNNModel(rnn_layer, vocab_size)
model.initialize(force_reinit=True, ctx=ctx)
print(predict_rnn_gluon('22', 10, model, vocab_size, ctx, idx_to_char, char_to_idx))


# 本函数已保存在d2lzh包中方便以后使用
def train_and_predict_rnn_gluon(
    model,
    num_hiddens,
    vocab_size,
    ctx,
    corpus_indices,
    idx_to_char,
    char_to_idx,
    num_epochs,
    num_steps,
    lr,
    clipping_theta,
    batch_size,
    pred_period,
    pred_len,
    prefixes,
):
    loss = gloss.SoftmaxCrossEntropyLoss()
    model.initialize(ctx=ctx, force_reinit=True, init=init.Normal(0.01))
    trainer = gluon.Trainer(
        model.collect_params(), "sgd", {"learning_rate": lr, "momentum": 0, "wd": 0}
    )

    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = d2l.data_iter_consecutive(
            corpus_indices, batch_size, num_steps, ctx
        )
        state = model.begin_state(batch_size=batch_size, ctx=ctx)
        for X, Y in data_iter:
            for s in state:
                s.detach()
            with autograd.record():
                (output, state) = model(X, state)
                y = Y.T.reshape((-1,))
                l = loss(output, y).mean()
            l.backward()
            # 梯度裁剪
            params = [p.data() for p in model.collect_params().values()]
            d2l.grad_clipping(params, clipping_theta, ctx)
            trainer.step(1)  # 因为已经误差取过均值，梯度不用再做平均
            l_sum += l.asscalar() * y.size
            n += y.size

        if (epoch + 1) % pred_period == 0:
            print(
                "epoch %d, perplexity %f, time %.2f sec"
                % (epoch + 1, math.exp(l_sum / n), time.time() - start)
            )
            for prefix in prefixes:
                print(
                    " -",
                    predict_rnn_gluon(
                        prefix,
                        pred_len,
                        model,
                        vocab_size,
                        ctx,
                        idx_to_char,
                        char_to_idx,
                    ),
                )


num_epochs, batch_size, lr, clipping_theta = 50, 20, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 20, ["2", "4"]
train_and_predict_rnn_gluon(
    model,
    num_hiddens,
    vocab_size,
    ctx,
    corpus_indices,
    idx_to_char,
    char_to_idx,
    num_epochs,
    num_steps,
    lr,
    clipping_theta,
    batch_size,
    pred_period,
    pred_len,
    prefixes,
)
print('--------------------------------------------------------------------------------------------------------')
pre2=predict_rnn_gluon("2", 20, model, vocab_size, ctx, idx_to_char, char_to_idx)
print(pre2)
print(pre2.count('4'))
