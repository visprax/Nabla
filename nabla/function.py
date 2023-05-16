"""Provides Function abstract class as an interface for operators."""

class Function:
    """All operators are implemented by inheriting Function class.

    Each operator must provide an implementation of `forward` and
    `backward` methods, and optionally the `get_params` method to
    provide access to its parameters.
    """
    def __init__(self, *tensors):
        self.parents = tensors

    def forward(self, *args, **kwargs):
        """Receives the input and returns its transformation by the operator."""
        raise NotImplementedError(f"forward method not implemented for {type(self)}")

    def backward(self, *args, **kwargs):
        """Provides the capability to perform automatic differentiation.

        Receives partial derivatives of the loss function with respect to the
        operator input and its parameters, if there are any.
        """
        raise NotImplementedError(f"backward method not implemented for {type(self)}")

    def get_params(self):
        """Provides access to parameters of operators."""
        raise NotImplementedError(f"get_params method not implemented for {type(self)}")


class Add(Function):
    def forward(self, other):
        return self.data + other.data
