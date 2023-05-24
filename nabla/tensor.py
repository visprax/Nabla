import numpy as np
from nabla import ops

class Tensor:
    _default_type = np.float32

    # TODO: fix init from Tensor objects!
    def __init__(self, data, dtype=None, _parents=(), _label=None, _op=None, requires_grad=None):
        if isinstance(data, list):
            self.data = np.array(data, dtype=dtype if dtype is not None else Tensor._default_type)
        elif isinstance(data, (int, float)):
            self.data = np.array([data], dtype=dtype if dtype is not None else Tensor._default_type)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(dtype if dtype is not None else data.dtype)
        else:
            raise RuntimeError(f"can't init Tensor from {type(data)}")
        self._parents = set(_parents)
        self._label = _label
        self._op = _op
        self.requires_grad = requires_grad
        self.grad = 0.0

    def __repr__(self):
        return f"<Tensor data={self.data} label={self._label} grad={self.grad} parents={self._parents} op={self._op} requires_grad={self.requires_grad}>"

    # TODO: fix repetitive instance checks!
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return ops.Add.apply(self, other, _op=ops.OpCodes.ADD)

    def __radd__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return ops.Add.apply(other, self, _op=ops.OpCodes.ADD)

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return ops.Mul.apply(self, other, _op=ops.OpCodes.MUL)

    def __rmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return ops.Mul.apply(other, self, _op=ops.OpCodes.MUL)

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return ops.MatMul.apply(self, other, _op=ops.OpCodes.MATMUL)

    def __rmatmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return ops.MatMul.apply(other, self, _op=ops.OpCodes.MATMUL)

    def __pow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return ops.Pow.apply(self, other, _op=ops.OpCodes.POW)

    def exp(self):
        return ops.Exp.apply(self, _op=ops.OpCodes.EXP)

