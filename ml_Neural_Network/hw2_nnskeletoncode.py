## this is example skeleton code for a Tensorflow/PyTorch neural network 
## module. You are not required to, and indeed probably should not
## follow these specifications exactly. Just try to get a sense for the kind
## of program structure that might make this convenient to implement.

# overall module class which is subclassed by layer instances
# contains a forward method for computing the value for a given
# input and a backwards method for computing gradients using
# backpropogation.

class Module():
    def __init__(self):
        self.prev = None # previous network (linked list of layers)
        self.output = None # output of forward call for backprop.

    learning_rate = 1E-2 # class-level learning rate

    def __call__(self, input):
        if isinstance(input, Module):
            # todo. chain two networks together with module1(module2(x))
            # update prev and output
        else:
            # todo. evaluate on an input.
            # update output

        return self

    def forward(self, *input):
        raise NotImplementedError

    def backwards(self, *input):
        raise NotImplementedError


# sigmoid non-linearity
class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, input):
        # todo. compute sigmoid, update fields
        pass

    def backwards(self, gradient):
        # todo. compute gradients with backpropogation and data from forward pass
        pass


# linear (i.e. linear transformation) layer
class Linear(Module):
    def __init__(self, input_size, output_size, is_input=False):
        super(Linear, self).__init__()
        # todo. initialize weights and biases. 

    def forward(self, input):  # input has shape (batch_size, input_size)
        # todo compute forward pass through linear input
        pass

    def backwards(self, gradient):
        # todo compute and store gradients using backpropogation
        pass


# generic loss layer for loss functions
class Loss:
    def __init__(self):
        self.prev = None

    def __call__(self, input):
        self.prev = input
        return self

    def forward(self, input, labels):
        raise NotImplementedError

    def backwards(self):
        raise NotImplementedError


# MSE loss function
class MeanErrorLoss(Loss):
    def __init__(self):
        super(MeanErrorLoss, self).__init__()

    def forward(self, input, labels):  # input has shape (batch_size, input_size)
        # todo compute loss, update fields
        pass

    def backwards(self):
        # todo compute gradient using backpropogation


## overall neural network class
class Network(Module):
    def __init__(self):
        super(Network, self).__init__()
        # todo initializes layers, i.e. sigmoid, linear

    def forward(self, input):
        # todo compute forward pass through all initialized layers
        pass

    def backwards(self, grad):
        # todo iterate through layers and compute and store gradients
        pass

    def predict(self, data):
        # todo compute forward pass and output predictions

    def accuracy(self, test_data, test_labels):
        # todo evaluate accuracy of model on a test dataset


# function for training the network for a given number of iterations
def train(model, data, labels, num_iterations, minibatch_size, learning_rate):
    # todo repeatedly do forward and backwards calls, update weights, do 
    # stochastic gradient descent on mini-batches.
