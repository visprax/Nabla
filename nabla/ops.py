import numpy as np
from enum import Enum, auto
from nabla.function import Function

class OpCodes(Enum):
    ADD    = auto()
    MUL    = auto()
    MATMUL = auto()
    EXP    = auto()

class Add(Function):
    def forward(self, x, y):
        return x + y

class Mul(Function):
    def forward(self, x, y):
        return x * y

class MatMul(Function):
    def forward(self, x, y):
        return x @ y

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
