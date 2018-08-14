# python notebook for Make Your Own Neural Network

# neural network class definition
class neuralNetwork:
    
    # initialize the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each inpput, hidden, output layer
        self.inodes = inputnodes
        self.hondes = hiddennodes
        self.onodes = outputnodes
        
        # learning rate
        self.lr = learningrate
    
    # train the neural network
    def train():
        pass
    
    # query the neural network
    def query():
        pass

# 使用我们创建的神经网络类
# number of input, hidden and output nodes
input_nodes = 3
hidden_nodes = 3
output_nodes = 3

#learning rate is 0.3
learning_rate = 0.3

#create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)