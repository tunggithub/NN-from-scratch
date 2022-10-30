from turtle import forward
import numpy as np


class Dense():
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, lr):
        weights_error = np.dot(self.input.T, output_error)
        bias_error = output_error

        self.weights -= lr*weights_error
        self.bias -= lr*bias_error

        input_error = np.dot(output_error, self.weights.T)
        return input_error


class Sigmoid():
    def __init__(self, input_data):
        self.input_data = input_data

    def forward_propagation(self):
        output = 1 / (1 + np.exp(-self.input_data))
        return output
    
    def backward_propagation(self, output_error, lr):
        derivatives = self.forward_propagation(self.input_data) * (1-self.forward_propagation(self.input_data))
        error = output_error * derivatives
        return error