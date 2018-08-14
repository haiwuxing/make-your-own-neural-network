# python notebook for Make Your Own Neural Network

import numpy

# neural network class definition
class neuralNetwork:
    
    # initialize the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each inpput, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # learning rate
        self.lr = learningrate

        # link weight matirces, wih and who
        # weights inside tha arrays are w_i_j, where link is from node i to node j int eh next layer
        # w11 w21
        # w12 w22 etc
        self.wih = numpy.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = numpy.random.rand(self.onodes, self.hnodes) - 0.5
    
    # train the neural network
    def train(self):
        pass
    
    # query the neural network
    def query(self):
        pass

    def __str__(self):
        return "wih:" +str(self.wih) + "\n" + \
        "who:" + str(self.who) + "\n"

# 使用我们创建的神经网络类
# number of input, hidden and output nodes
input_nodes = 2
hidden_nodes = 3
output_nodes = 1

#learning rate is 0.3
learning_rate = 0.3

#create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
print("n:"+str(n))