# python notebook for Make Your Own Neural Network
# 测试2/3/1 神经网络

import numpy
# scipy.special for the sigmoid function expit()
import scipy.special

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
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # 使用逻辑函数作为激活函数
        self.activation_function = lambda x: scipy.special.expit(x)
    
    # train the neural network
    def train(self, input_list, target_list):
        # 将输入和输出列表转换为二维数组
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T

        # 计算到隐藏层的信号
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 计算从隐藏出来的信号
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算到最终输出层的信号
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 计算从最终输出层出来的信号，也就是最终结果
        final_outputs = self.activation_function(final_inputs)

        # 输出层误差：(target - actual)
        output_errors = targets - final_outputs
        # 隐藏层误差是输出层误差按权重分割，然后乘以输出节点
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # 更新隐藏层和输出层之间的权重. 这是最最重要的核心算法!!!!!!!!!
        self.who += self.lr * numpy.dot(output_errors * final_outputs * (1.0 - final_outputs), numpy.transpose(hidden_outputs))

        # 更新输入层和隐藏之间的权重。
        self.wih += self.lr * numpy.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs), numpy.transpose(inputs))
    
    # query the neural network
    def query(self, input_list):

        # 将输入列表转换为二维数组
        inputs = numpy.array(input_list, ndmin=2).T

        # 计算输入到隐藏层的信号
        hidden_inputs = numpy.dot(self.wih, inputs)

        # 计算从隐藏层输出的的信号
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算进到输出层的信号
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # 计算从最终输出层输出的信号
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def __str__(self):
        return "输入层到隐藏的权重：\n" + str(self.wih) + "\n" + \
        "隐藏层到输出层的权重：\n" + str(self.who) + "\n"
        "学习率：" + str(self.lr) + "\n"

# 使用我们创建的神经网络类
# number of input, hidden and output nodes
input_nodes = 2
hidden_nodes = 3
output_nodes = 1

#learning rate is 0.3
learning_rate = 0.3

#create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
print("n:\n"+str(n))
query_result = n.query([1.0, 0.5])
print("查询结果：\n" + str(query_result))

x = numpy.array(([2, 9], [5, 1], [3, 6]), dtype=float)
y = numpy.array(([92], [30], [89]), dtype=float)

# scale units
x = x / numpy.amax(x, axis=0)
y = y / 100

print("Input: \n" + str(x))
print("Actual Output: \n" + str(y))

for i in range(10):
    print("训练" + str(i) +"次")
    print("Predicted Output:\n" + str(n.query([2,9])))
    print("\n")
    n.train(x[0], y[0])