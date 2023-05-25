import numpy as np
from enum import Enum, auto
from nabla.function import Function

class OpCodes(Enum):
    ADD     = auto()
    SUB     = auto()
    NEG     = auto()
    MUL     = auto()
    TRUEDIV = auto()
    MATMUL  = auto()
    POW     = auto()
    EXP     = auto()
    TANH    = auto()
    EQUAL   = auto()

class Add(Function):
    def forward(self, x, y):
        return x + y

class Sub(Function):
    def forward(self, x, y):
        return x - y

class Neg(Function):
    def forward(self, x):
        return -x

class Mul(Function):
    def forward(self, x, y):
        return x * y

class Truediv(Function):
    def forward(self, x, y):
        return x / y

class MatMul(Function):
    def forward(self, x, y):
        return x @ y

class Pow(Function):
    def forward(self, x, y):
        return x ** y

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

class Tanh(Function):
    def forward(self, x):
        return np.tanh(x)

class Equal(Function):
    def forward(self, x, y):
        return x == y
