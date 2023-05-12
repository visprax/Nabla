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
        """Override __repr__ magic method for printing an Scalar object."""
        return f"Scalar(value={self.value}, grad={self.grad}, label={self._label})"

    def __add__(self, other):
        """Rule for adding two Scalar objects."""
        other = Scalar._check_instance(other)
        output = Scalar(self.value + other.value, (self, other), "PLUS", "")
        return output

    def __radd__(self, other):
        """Override reverse __add__ magic method."""
        # for expression like: 2 + Scalar(3.0)
        return self + other

    def __mul__(self, other):
        """Rule for multiplying two Scalar objects."""
        other = Scalar._check_instance(other)
        output = Scalar(self.value * other.value, (self, other), "MUL", "")
        return output

    def __rmul__(self, other):
        """Override reverse __mul__ magic method."""
        # for expressions like: 2 * Scalar(3.0)
        return self * other

    def __pow__(self, other):
        """Rule for an Scalar to the power of a number."""
        other = Scalar._check_instance(other)
        output = Scalar(self.value ** other.value, (self, other), "POW", "")
        return output

    # TODO: this needs testing, weird behaviour
    def __rpow__(self, other):
        """Override reverse __pow__ magic method."""
        other = Scalar._check_instance(other)
        # unlike add and multiplication operations, power is order dependent!
        output = Scalar(other ** self.value, (self, other), "POW", "")
        return output

    def __truediv__(self, other):
        """Rule for division. __div__ is for floor division."""
        other = Scalar._check_instance(other)
        return self * other ** -1

    def __rtruediv__(self, other):
        """Override reverse __truediv__ magic method."""
        other = Scalar._check_instance(other)
        return other * self ** -1

    def __sub__(self, other):
        """Rule for subtracting Scalar objects."""
        other = Scalar._check_instance(other)
        return self + (-other)

    def __neg__(self):
        """Rule for negating an Scalar."""
        return self * -1


    # TODO: check this method!
    def _check_instance(other):
        if not isinstance(other, (int, float, Scalar)):
            raise ValueError("Only operations on int, float or other Scalar objects are allowed.")
        # if the other object is an int or float instance, convert it to an Scalar object
        if not isinstance(other, Scalar):
            other = Scalar(other)

        return other





