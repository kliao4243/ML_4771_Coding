import numpy as np


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


class Neuron(object):
    def __init__(self, input_size):
        self.weight = np.random.random_sample(input_size)
        self.bias = np.random.random_sample()
        self.input = None
        self.output = None
        self.E_over_net = None
        self.del_w = None

    def forward(self):
        output = sigmoid(np.dot(self.weight, self.input) + self.bias)
        self.output = output
        return output

    def backpropagation(self, error, next_E_over_net, last):
        if last == 1:
            self.E_over_net = error * self.output * (1 - self.output)
            self.del_w = self.input * self.E_over_net
            self.del_w = self.del_w.reshape((self.del_w.shape[0],))
        else:
            self.E_over_net = next_E_over_net * self.output * (1 - self.output)
            self.del_w = self.input * self.E_over_net
            self.del_w = self.del_w.reshape((self.del_w.shape[0],))


class NeuralNetwork:
    def __init__(self, n_inter_layer, n_neuron, output_size=1):
        self.layers = list()
        first_layer = [Neuron(1), Neuron(1)]
        self.layers.append(first_layer)
        for i in range(1, n_inter_layer+1):
            temp_layer = list()
            for j in range(0, n_neuron):
                temp_layer.append(Neuron(len(self.layers[i-1])))
            self.layers.append(temp_layer)
        output_layer = list()
        for i in range(0, output_size):
            output_layer.append(Neuron(n_neuron))
        self.layers.append(output_layer)

    # input data should be a np array
    def forward(self, input_data):
        self.layers[0][0].input = np.asarray([input_data[0]], dtype=np.float64)
        self.layers[0][0].forward()
        self.layers[0][1].input = np.asarray([input_data[1]], dtype=np.float64)
        self.layers[0][1].forward()
        for i in range(1, len(self.layers)):
            temp_input = list()
            for neuron in self.layers[i-1]:
                temp_input.append(neuron.output)
            for neuron in self.layers[i]:
                neuron.input = np.asarray(temp_input, dtype=np.float64)
                neuron.forward()
        prediction = list()
        for neuron in self.layers[-1]:
            prediction.append(neuron.output)
        return np.asarray(prediction, dtype=np.float64)

    def backpropagation(self, error):
        for i in range(0, len(self.layers[-1])):
            self.layers[-1][i].backpropagation(error[i], 0, 1)
        for i in range(len(self.layers)-2, -1, -1):
            for j in range(0, len(self.layers[i])):
                temp_derivative = 0
                for neuron in self.layers[i+1]:
                    temp_derivative += neuron.E_over_net * neuron.weight[j]
                self.layers[i][j].backpropagation(0, temp_derivative, 0)
        return

    def train(self, data, labels, num_iterations, learning_rate=1E-2):
        for i in range(0, num_iterations):
            for j in range(0, data.shape[0]):
                prediction = self.forward(data[j])
                self.backpropagation(prediction-labels[j])
                for layer in self.layers:
                    for neuron in layer:
                        neuron.weight -= learning_rate * neuron.del_w
                        neuron.bias -= learning_rate * neuron.E_over_net


if __name__ == "__main__":
    data = []
    labels = []
    for i in range(0, 100):
        x = np.random.random_sample(2) * 10
        y = np.asarray([int((x[0]**2+x[1]**2)>50)])
        data.append(x)
        labels.append(y)
    data = np.asarray(data, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    test = NeuralNetwork(100, 128)
    test.train(data, labels, 2)



