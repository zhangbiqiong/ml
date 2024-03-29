# import time
import numpy as np
# import mxnet as mx
from gensim.models import Word2Vec
import random

row = 50000
col = 50
negativeword = set()

def create_data():
    doc = np.random.randint(11, 99, (row, col))
    for n in range(row - 1):
        line = doc[n]
        listline = list(line)
        if listline.count(22) != 0 and listline.count(44) != 0:
            if listline.index(22) > listline.index(44):
                line[listline.index(22)] = 44
                line[listline.index(44)] = 22

    datafilter = []
    for n in range(row - 1):
        line = doc[n]
        listline = list(line)
        if listline.count(22) != 0:
            datafilter.append(listline[: listline.index(22)])
            # for word in listline[:listline.index(22)]:
            if len(negativeword) < 5:
                negativeword.add(str(listline[0]))
            datafilter.append(listline[listline.index(22) :])
    return datafilter

data = create_data()
# print('-----------------------------------------------------------')
# print(data)

def convert2str(data):
    newdata = []
    for line in data:
        if line.count(44)>0 and random.randint(1,30)>1:
            continue
        newline = []
        for word in line:
            newword = str(word)
            newline.append(newword)
        newdata.append(newline)
    return newdata


newdata = convert2str(data)
# print(newdata)
np.savetxt('a.txt',newdata,'%s')
# np.array(newdata).save('aaa.txt')


model = Word2Vec(
    sentences=newdata,  # We will supply the pre-processed list of moive lists to this parameter
    iter=5,  # epoch
    min_count=1,  # a movie has to appear more than 10 times to be keeped
    size=200,  # size of the hidden layer
    workers=4,  # specify the number of threads to be used for training
    sg=1,  # Defines the training algorithm. We will use skip-gram so 1 is chosen.
    hs=0,  # Set to 0, as we are applying negative sampling.
    negative=5,  # If > 0, negative sampling will be used. We will use a value of 5.
    window=100,
)

print("-------------------------negativeword----------------------------------")
print((list(negativeword)))
print("-------------------------model.most_similar----------------------------------")
print(model.most_similar(positive=["22"], negative=list(negativeword)))

