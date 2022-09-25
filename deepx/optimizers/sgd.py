import numpy as np

from optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, params, lr=0.001, weight_decay=0.0, momentum=0.9):
        super().__init__(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.velocity = [np.zeros_like(param.grad) for param in params]

    def step(self):
        for param, vel in zip(self.params, self.velocity):
            vel = self.momentum*vel + param.grad + self.weight_decay*param.data
            param.data -= self.lr * vel

