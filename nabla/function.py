import numpy as np
from nabla import tensor

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
