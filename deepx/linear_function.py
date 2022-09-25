import numpy as np

from function import Function
from tensor import Tensor

class Linear(Function):
    def __init__(self, inodes, onodes):
        self.weights = Tensor((inodes, onodes))
        self.bias    = Tensor((1, onodes))
        self.type    = "linear"

    def forward(self, x):
        output = np.dot(x, self.weights.data) + self.bias.data
        self.input = x
        return output

    def backward(self, dy):
        self.weights.grad += np.dot(self.input.T, dy)
        self.bias.grad    += np.sum(dy, axis=0, keepdims=True)
        grad_input         = np.dot(dy, self.weights.data.T)
        return grad_input

    def get_params(self):
        return [self.weights, self.bias]
