# The simple version of Nabla augrad engine
import numpy as np

class Scalar:
    def __init__(self, value, _parents=(), _operation="", _label=""):
        # the actual value itself that the scalar holds
        self.value = value
        # the parents of this scalar node
        self._prev = set(_parents)
        # the operation that resulted in this scalar
        self._operation = _operation
        # the label for this scalar
        self._label = _label
        # the initial gradient with respect to the final output node (e.g. loss function)
        self.grad = 0.0
        # the local backpropagation function for this node
        self._backward = lambda: None

    def __repr__(self):
        """Overloaded __repr__ method. How to print an Scalar object."""
        return f"Scalar(value={self.value}, grad={self.grad}, label={self._label})"

    def __add__(self, other):
        """Rule for adding two Scalar objects."""
        # we can only add int and float values with an scalar
        if not isinstance(other, (int, float, Scalar)):
            raise SystemExit("Only int, float or an Scalar value can be summed with an Scalar object.")
        # if the other number is not an instance of Scalar, convert it to an Scalar object
        if not isinstance(other, Scalar):
            other = Scalar(other)

        output = Scalar(self.value + other.value, (self, other), "PLUS", "")
        return output

    def __radd__(self, other):
        """Overloaded reverse add method."""
        # for expression like: 2 + Scalar(3.0)
        return self + other

    def __mul__(self, other):
        """Rule for multiplying two Scalar objects."""
        # we can only multiply int and float values with an scalar
        if not isinstance(other, (int, float, Scalar)):
            raise SystemExit("Only int, float or an Scalar value can be multiplied with an Scalar object.")
        # if the other number is not an instance of Scalar, convert it to an Scalar object
        if not isinstance(other, Scalar):
            other = Scalar(other)

        output = Scalar(self.value * other.value, (self, other), "MULP", "")
        return output

    def __rmul__(self, other):
        """Overloaded reverse multiplication method."""
        return self * other





