import numpy as np
from nabla.function import Add
from nabla.ops import Ops

class Tensor:
    def __init__(self, data, dtype=None, requires_grad=None):
        self.data = np.array(data, dtype=dtype if dtype else np.float32)
        self.requires_grad = requires_grad
        self.grad = None

    def __repr__(self):
        return f"<Tensor data={self.data} grad={self.grad} dtype={self.dtype}>"

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def __add__(self, other):
        return self._apply(Ops.ADD, other)

    def _apply(self, op, other):
        if op == Ops.ADD:
            out = Add.forward(self, other)
        return out

