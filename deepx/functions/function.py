"""Provides Function abstract class as an interface for operators."""

class Function:
    """All operators are implemented by inheriting Function class.
    
    Each operator must provide an implementation of `forward` and 
    `backward` methods, and optionally the `get_params` method to 
    provide access to its parameters.
    """

    def forward(self):
        """Receives the input and returns its transformation by the operator."""
        raise NotImplementedError

    def backward(self):
        """Provides the capability to perform automatic differentiation.

        Receives partial derivatives of the loss function with respect to the
        operator input and its parameters, if there are any.
        """
        raise NotImplementedError

    def get_params(self):
        """Provides access to parameters of operators."""
        return []
