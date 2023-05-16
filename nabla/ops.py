from nabla.function import Function
from nabla.tensor import Tensor


class Add(Function):
    def forward(self, x, y):
        return x + y
