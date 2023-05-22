import numpy as np
from nabla import ops

class Tensor:
    _default_type = np.float32

    def __init__(self, data, dtype=None, _parents=(), _label=None, _op=None, requires_grad=None):
        self.data = data
        self.dtype = dtype if dtype else Tensor._default_type
        self._parents = set(_parents)
        self._label = _label
        self._op = _op
        self.requires_grad = requires_grad
        self.grad = 0.0

    def __repr__(self):
        return f"<Tensor data={self.data} label={self._label} grad={self.grad} parents={self._parents} op={self._op} requires_grad={self.requires_grad}>"

    def __add__(self, other):
        return ops.Add.apply(self, other, _op=ops.Ops.ADD)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        result = self.data * other.data
        output = Tensor(result, dtype=result.dtype, _parents=(self, other), _op="MUL")
        return output

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        result = self.data @ other.data
        output = Tensor(result, dtype=result.dtype, _parents=(self, other), _op="MATMUL")
        return output

    def __rmatmul__(self, other):
        return self @ other

    def exp(self):
        return ops.Exp.apply(self)

