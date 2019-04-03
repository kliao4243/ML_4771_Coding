import numpy as np
import scipy.io
from matplotlib import pyplot as PLT
import random


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

class Layer(object):
    def __init__(self, input_size, neuron_number):
        weight = list()
        bias = list()
        for i in range(0, neuron_number):
            weight.append(np.random.random_sample(input_size)*0.001)
            bias.append(np.random.random_sample())
        self.weight = np.asarray(weight).reshape(neuron_number, input_size)
        self.bias = np.asarray(bias).reshape(neuron_number, 1)*0.001
        self.input = None
        self.output = None
        self.E_over_net = None
        self.del_w = None

    def forward(self):
        self.output = sigmoid(np.matmul(self.weight, self.input) + self.bias)
        return self.output

    # error is 3x1 np array
    def backpropagation(self, y_test, next_E_net, last):
        if last == 1:
            self.E_over_net = np.multiply(self.output - y_test, self.output)
            self.E_over_net = np.multiply(self.E_over_net, 1 - self.output)
            self.E_over_net = self.E_over_net.reshape(self.weight.shape[0], 1)
            self.del_w = np.matmul(self.E_over_net, self.input.T)
        else:
            self.E_over_net = np.multiply(next_E_net, self.output)
            self.E_over_net = np.multiply(self.E_over_net, 1 - self.output)
            self.E_over_net = self.E_over_net.reshape(self.weight.shape[0], 1)
            self.del_w = np.matmul(self.E_over_net, self.input.T)
        return self.del_w


class NeuralNetwork:
    def __init__(self, output_size=1):
        self.first_hidden = Layer(2, 128)
        self.second_hidden = Layer(128, 256)
        self.output_layer = Layer(256, output_size)

    # input data should be a 2x1 np array
    def forward(self, input_data):
        # first
        self.first_hidden.input = input_data
        self.first_hidden.forward()
        # second
        self.second_hidden.input = self.first_hidden.output
        self.second_hidden.forward()
        # third
        self.output_layer.input = self.second_hidden.output
        self.output_layer.forward()
        return self.output_layer.output

    def backpropagation(self, label):
        self.output_layer.backpropagation(label, 0, 1)
        self.second_hidden.backpropagation(0, np.matmul(self.output_layer.weight.T, self.output_layer.E_over_net), 0)
        self.first_hidden.backpropagation(0, np.matmul(self.second_hidden.weight.T, self.second_hidden.E_over_net), 0)
        return np.sum(0.5*np.square(label - self.output_layer.output))

    def train(self, data, labels, mini_batch_size, learning_rate=1E-3, iteration=2000):
        index = random.sample(range(0, data.shape[0]), mini_batch_size)
        training_data = [data[i] for i in index]
        training_labels = [labels[i] for i in index]
        try:
            for ite in range(0, iteration):
                i = 0
                error = 0
                for sample in training_data:
                    self.forward(sample.reshape(2, 1))
                    error += self.backpropagation(training_labels[i].reshape(3, 1))
                    self.first_hidden.weight -= learning_rate * self.first_hidden.del_w
                    print(self.first_hidden.del_w[0])
                    self.first_hidden.bias -= learning_rate * self.first_hidden.E_over_net
                    self.second_hidden.weight -= learning_rate * self.second_hidden.del_w
                    self.second_hidden.bias -= learning_rate * self.second_hidden.E_over_net
                    self.output_layer.weight -= learning_rate * self.output_layer.del_w
                    self.output_layer.bias -= learning_rate * self.output_layer.E_over_net
                    #print(self.output_layer.weight)
                    i += 1
                print("current error is " + str(error))
                #if (np.sum(self.first_hidden.E_over_net)+np.sum(self.second_hidden.E_over_net)+\
                #        np.sum(self.output_layer.E_over_net)) < 0.0000001:
                #    break
        except KeyboardInterrupt:
            exit()

    def adam_train(self, data, labels, mini_batch_size, learning_rate=1E-2, iteration=2000):
        index = random.sample(range(0, data.shape[0]), mini_batch_size)
        training_data = [data[i] for i in index]
        training_labels = [labels[i] for i in index]

        m1 = np.zeros((self.first_hidden.weight.shape[0], self.first_hidden.weight.shape[1]))
        v1 = np.zeros((self.first_hidden.weight.shape[0], self.first_hidden.weight.shape[1]))
        bm1 = np.zeros((self.first_hidden.bias.shape[0], self.first_hidden.bias.shape[1]))
        bv1 = np.zeros((self.first_hidden.bias.shape[0], self.first_hidden.bias.shape[1]))
        m2 = np.zeros((self.second_hidden.weight.shape[0], self.second_hidden.weight.shape[1]))
        v2 = np.zeros((self.second_hidden.weight.shape[0], self.second_hidden.weight.shape[1]))
        bm2 = np.zeros((self.second_hidden.bias.shape[0], self.second_hidden.bias.shape[1]))
        bv2 = np.zeros((self.second_hidden.bias.shape[0], self.second_hidden.bias.shape[1]))
        m3 = np.zeros((self.output_layer.weight.shape[0], self.output_layer.weight.shape[1]))
        v3 = np.zeros((self.output_layer.weight.shape[0], self.output_layer.weight.shape[1]))
        bm3 = np.zeros((self.output_layer.bias.shape[0], self.output_layer.bias.shape[1]))
        bv3 = np.zeros((self.output_layer.bias.shape[0], self.output_layer.bias.shape[1]))
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1E-8
        last_error = -1
        for ite in range(0, iteration):
            i = 0
            error = 0
            for sample in training_data:
                self.forward(sample.reshape(2, 1))
                error += self.backpropagation(training_labels[i].reshape(3, 1))
                i += 1
                # first_hidden update
                m1 = beta1 * m1 + (1-beta1) * self.first_hidden.del_w
                #print("m1              ",m1[0])
                #print("del_w    ", self.first_hidden.del_w[0])
                #print("E_over_net     ", self.first_hidden.E_over_net[0])
                v1 = beta2 * v1 + (1-beta2) * (self.first_hidden.del_w)**2
                #print("v1          ",v1[0])
                self.first_hidden.weight -= learning_rate * (m1/(np.sqrt(v1)+epsilon))
                #print(m1/(np.sqrt(v1)+epsilon))
                bm1 = beta1 * bm1 + (1-beta1) * self.first_hidden.E_over_net
                bv1 = beta2 * bv1 + (1-beta2) * self.first_hidden.E_over_net**2
                self.first_hidden.bias -= learning_rate * (bm1/(np.sqrt(bv1)+epsilon))
                # second_hidden update
                m2 = beta1 * m2 + (1-beta1) * self.second_hidden.del_w
                v2 = beta2 * v2 + (1-beta2) * self.second_hidden.del_w**2
                self.second_hidden.weight -= learning_rate * (m2/(np.sqrt(v2)+epsilon))
                bm2 = beta1 * bm2 + (1-beta1) * self.second_hidden.E_over_net
                bv2 = beta2 * bv2 + (1-beta2) * self.second_hidden.E_over_net**2
                self.second_hidden.bias -= learning_rate * (bm2/(np.sqrt(bv2)+epsilon))
                # output_layer update
                m3 = beta1 * m3 + (1-beta1) * self.output_layer.del_w
                v3 = beta2 * v3 + (1-beta2) * self.output_layer.del_w**2
                self.output_layer.weight -= learning_rate * (m3/(np.sqrt(v3)+epsilon))
                bm3 = beta1 * bm3 + (1-beta1) * self.output_layer.E_over_net
                bv3 = beta2 * bv3 + (1-beta2) * self.output_layer.E_over_net**2
                self.output_layer.bias -= learning_rate * (bm3/(np.sqrt(bv3)+epsilon))
            error = error/mini_batch_size
            if error - last_error < 0.00001:
                learning_rate = random.choice([1e-4, 5e-4, 1e-3, 5e-3])
            print("current average error is " + str(error))

def RGB_display(coordinate, RGB_value):
    x_max = 0
    y_max = 0
    for i in range(coordinate.shape[0]):
        if coordinate[i][0] > x_max:
            x_max = coordinate[i][0]
        if coordinate[i][1] > y_max:
            y_max = coordinate[i][1]
    temp_pic = np.zeros((y_max + 1, x_max + 1, 3), dtype=np.int)
    for i in range(coordinate.shape[0]):
        temp_pic[coordinate[i][1]][coordinate[i][0]] = RGB_value[i]
    temp_pic = temp_pic.reshape((x_max+1, y_max+1 , 3))
    return temp_pic

def main():
    '''
    data = []
    labels = []
    for i in range(0, 100):
        x = np.random.random_sample(2) * 5
        y = np.asarray((x[0] ** 2 + x[1] ** 2)/50)
        data.append(x)
        labels.append(y)
    data = np.asarray(data, dtype=np.float64).reshape(100, 2)
    labels = np.asarray(labels, dtype=np.float64).reshape(100, 1)
    test = NeuralNetwork(4)
    print(test.output_layer.weight)
    test.train(data, labels, 0, 50)
    '''

    mat = scipy.io.loadmat('hw2_data.mat')
    X1 = mat['X1']
    Y1 = mat['Y1']
    X2 = mat['X2']
    Y2 = mat['Y2']

    X1 = X1.astype(np.int)
    Y1 = Y1.astype(np.int)
    X2 = X2.astype(np.int)
    Y2 = Y2.astype(np.int)

    Y1 = Y1/255
    Y2 = Y2/255
    #test_pic = RGB_display(X1, Y1)
    #PLT.imshow(test_pic)
    #PLT.show()

    model = NeuralNetwork(3)
    #model.adam_train(X2, Y2, 1000, 1E-1, 200)
    model.adam_train(X2, Y2, 1000, 1E-4, 5000)
    #model.adam_train(X2, Y2, 1000, 1E-3, 4000)
    Y2_pre = list()
    for i in range(X2.shape[0]):
        Y2_pre.append((model.forward(X2[i].reshape(2, 1))).reshape(3, 1)*255)
    Y2_pre = np.asarray(Y2_pre, dtype=int)
    Y2_pre = Y2_pre.reshape(18620, 3)
    test_pic = RGB_display(X2, Y2_pre)
    PLT.imshow(test_pic)
    PLT.imsave('test.png',test_pic)
    PLT.show()
    return model, X2, Y2


if __name__ == "__main__":
    model, X2, Y2 = main()