import numpy
import scipy.special
import matplotlib.pyplot
import random

class neuralNetwork:

    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes

        self.wih=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))

        self.lr=learningrate
        self.activation_function=lambda x:scipy.special.expit(x)

        pass

    def train(self,inputs_list,targets_list):
        inputs=numpy.array(inputs_list,ndmin=2).T
        targets=numpy.array(targets_list,ndmin=2).T

        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)

        final_inputs=numpy.dot(self.who,hidden_outputs)
        final_outpus=self.activation_function(final_inputs)

        output_errors=targets-final_outpus
        hidden_errors=numpy.dot(self.who.T,output_errors)


        self.who+=self.lr*numpy.dot((output_errors*final_outpus*(1.0-final_outpus)),numpy.transpose(hidden_outputs))
        self.wih+=self.lr*numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),numpy.transpose(inputs))

        return output_errors.sum()**2
        # pass

    def query(self,inputs_list):
        inputs=numpy.array(inputs_list,ndmin=2).T
        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)
        final_inputs=numpy.dot(self.who,hidden_outputs)
        final_outpus=self.activation_function(final_inputs)
        return final_outpus





X=[
    [1,1,1],
    [1,3,2],
    [1,3,1],
    [1,23,22],
    [1,33,42],
    [1,23,12],
    [1,11,22],
    [1,33,32],
    [1,33,12],
    [1,53,32],
    [1,1,29],
    [1,18,27],
    [1,1,12],
    [1,21,22],
    [1,31,42],
    [1,2,12],
    [1,11,2],
    [1,33,2],
    [1,3,2],
    [1,5,59],

    [1,73,72],
    [1,83,82],
    [1,73,99],
    [1,93,92],
    [1,93,72],
    [1,73,62],
    [1,73,62],
    [1,63,62],
    [1,93,82],
    [1,63,89],
    [1,83,72],
    [1,63,82],
    [1,63,92],
    [1,63,61],
    [1,63,77],
    [1,93,77],
    [1,93,72],
    [1,93,76],
    [1,93,76],
    [1,99,72],


    [1,113,102],
    [1,183,182],
    [1,173,199],
    [1,193,192],
    [1,193,172],
    [1,173,162],
    [1,173,162],
    [1,163,162],
    [1,193,112],
    [1,163,199],
    [1,183,172],
    [1,113,182],
    [1,103,192],
    [1,163,191],
    [1,163,127],
    [1,193,177],
    [1,123,172],
    [1,193,176],
    [1,103,176],
    [1,199,102],

    [1,213,212],
    [1,283,282],
    [1,273,299],
    [1,293,292],
    [1,293,272],
    [1,213,211],
    [1,273,202],
    [1,263,262],
    [1,293,212],
    [1,263,299],
    [1,203,272],
    [1,213,282],
    [1,263,292],
    [1,263,291],
    [1,263,227],
    [1,293,277],
    [1,203,202],
    [1,293,276],
    [1,293,276],
    [1,299,272],

]

Y=[
    [1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],
    [0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],
    [0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],
    [0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],
]


batchsize=100

fig=matplotlib.pyplot.figure()
ax=fig.add_subplot()
ax.set(
    xlim = [0, batchsize],
    ylim = [0, 5],
    title = "An Example Axes",
    ylabel = "Y-Axis",
    xlabel = "X-Axis",
)

ax_x = []
ax_y = []

# print (len(Y))

nn=neuralNetwork(3,9000,4,0.0001)

for batch in range(batchsize):
    error=0.0
    for n in range(len(Y)):
        m=random.randint(0,len(Y)-1)# 随机学习
        error=nn.train(X[m],Y[m])
    ax_x.append(batch)
    ax_y.append(error)

ax.plot(ax_x, ax_y)
# matplotlib.pyplot.show()

print(numpy.floor((nn.query([1,11.0,31.0])*100)).transpose())
print(numpy.floor((nn.query([1,65.0,71.0])*100)).transpose())
print(numpy.floor((nn.query([1,181.0,190.0])*100)).transpose())
print(numpy.floor((nn.query([1,241.0,280.0])*100)).transpose())

print(numpy.floor((nn.query([1,31.0,11.0])*100)).transpose())
print(numpy.floor((nn.query([1,71.0,91.0])*100)).transpose())
print(numpy.floor((nn.query([1,139.0,180.0])*100)).transpose())
print(numpy.floor((nn.query([1,251.0,230.0])*100)).transpose())



