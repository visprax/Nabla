import numpy as np
from enum import Enum, auto
from nabla import tensor

class Ops(Enum):
    ADD = auto()
    MUL = auto()
    EXP = auto()

class Function:
    def __init__(self, *tensors):
        self.parents = tensors

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"forward method not implemented")

    def backward(self, *args, **kwargs):
        raise NotImplementedError(f"backward method not implemented")

    @classmethod
    def apply(func, *tensors, **kwargs):
        op = func(*tensors)
        result = tensor.Tensor(op.forward(*[t.data for t in tensors]), _parents=op.parents, **kwargs)
        return result

class Add(Function):
    def forward(self, x, y):
        return x + y

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
