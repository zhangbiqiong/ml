
# import time
import numpy as np
# import mxnet as mx
from gensim.models import Word2Vec
import random

mydict = { 
    1:'A1',
    2:'B1',
    3:'C1',
    4:'D1',
    5:'E1',
    6:'F1',
    7:'G1',
    8:'H1',
    9:'I1',
    10:'J1',
    11:'K1',
    12:'L1',

    13:'A2',
    14:'B2',
    15:'C2',
    16:'D2',
    17:'E2',
    18:'F2',
    19:'G2',
    20:'H2',
    21:'I2',
    22:'J2',
    23:'K2',
    24:'L2',


    113:'M',
    114:'N',
    115:'O',
    116:'P',
    117:'Q',
    118:'R',
    119:'S',
    120:'T',
    121:'U',
    122:'V',
    123:'W',
    124:'X',
    125:'Y',
    126:'Z',
}

lines=[]
row=1800
for n in range(1,row):#row
    line=[]
    for m in range(1,50):
        line.append(mydict[random.randint(1,24)])
    lines.append(line)

for line in lines[0:int(row/3)]:
    n=random.randint(10,40)
    line[n]='X'
    # line[n+1]='Z'
    line[n+random.randint(1,3)]='Y'

for line in lines[int(row/3):int(row/3)*2]:
    n=random.randint(10,40)
    line[n]='M'
    # line[n+1]='Z'
    line[n+random.randint(1,3)]='N'

for line in lines[int(row/3)*2:]:
    n=random.randint(10,40)
    line[n]='S'
    # line[n+1]='Z'
    line[n+random.randint(1,3)]='T'


np.savetxt('a.txt',lines,fmt='%s')
# print (lines)    

model = Word2Vec(
    sentences=lines,  # We will supply the pre-processed list of moive lists to this parameter
    iter=900,  # epoch
    min_count=10,  # a movie has to appear more than 10 times to be keeped
    size=1000,  # size of the hidden layer
    workers=10,  # specify the number of threads to be used for training
    sg=1,  # Defines the training algorithm. We will use skip-gram so 1 is chosen.
    hs=0,  # Set to 0, as we are applying negative sampling.
    negative=5,  # If > 0, negative sampling will be used. We will use a value of 5.
    window=25,
)


# X后面跟着Y M后面跟着N
print(model.wv.most_similar(positive=['X','N','S'], negative=['Y','T']))
# M后面跟着N S后面跟着T
# print(model.wv.most_similar(positive=['M','T'], negative=list('N')))

# print(model.wv.most_similar('M'))
