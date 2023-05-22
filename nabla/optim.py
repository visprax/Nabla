class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError("step method not implemented")

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0

class SGD(Optimizer):
    def __init__(self, params, lr=0.001, weight_decay=0.0, momentum=0.9):
        super().__init__(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum

