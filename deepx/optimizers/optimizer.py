class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError

    def reset_grad(self):
        for param in self.params:
            param.grad = 0.0
        
