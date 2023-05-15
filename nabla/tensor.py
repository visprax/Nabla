# TODO: remove numpy dependency!
import numpy as np

class Tensor:
    def __init__(self, data, dtype=None, requires_grad=None):
        self.data = data
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.grad = None

    def __repr__(self):
        return f"<Tensor data={self.data} grad={self.grad} dtype={self.dtype}>"
