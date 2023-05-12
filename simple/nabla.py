# The simple version of Nabla augrad engine
import math
import random

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
        """Addition operator overload."""
        other = Scalar._check_instance(other)
        output = Scalar(self.value + other.value, (self, other), "PLUS", "")
        
        # calculate the local gradient wrt. loss in an addition node
        # output = self + other
        # dL/dself  = dL/doutput * doutput/dself  = output.grad * 1
        # dL/dother = dL/doutput * doutput/dother = output.grad * 1
        def _backward():
            # note the += and not a =, this is necessary to accumlate gradients
            # for the same scalars as an input, this will in turn will require 
            # that we zero out all the gradients after each backward pass, before 
            # we can do another backward pass
            self.grad  += output.grad * 1.0
            other.grad += output.grad * 1.0
        # set the backpropagation function for addition nodes
        output._backward = _backward
        return output

    def __radd__(self, other):
        """Override reverse __add__ magic method."""
        # for expression like: 2 + Scalar(3.0)
        return self + other

    def __mul__(self, other):
        """Multiplication operator overload."""
        other = Scalar._check_instance(other)
        output = Scalar(self.value * other.value, (self, other), "MUL", "")
        def _backward():
            # calculate the local gradient wrt. loss in a multiplication node
            # output = self * other
            # dL/dself  = dL/doutput * doutput/dself  = output.grad * other.value
            # dL/dother = dL/doutput * doutput/dother = output.grad * self.value
            self.grad  += output.grad * other.value
            other.grad += output.grad * self.value
        # set the backpropagation function for multiplication nodes
        output._backward = _backward
        return output

    def __rmul__(self, other):
        """Override reverse __mul__ magic method."""
        # for expressions like: 2 * Scalar(3.0)
        return self * other

    def __pow__(self, other):
        """Power operator overload."""
        other = Scalar._check_instance(other)
        output = Scalar(self.value ** other.value, (self, other), "POW", "")
        def _backward():
            # calculate the local gradient wrt. loss in a power node
            # output = self ** other
            # dL/self = dL/doutput * doutput/dself = output.grad * other * self ** (other-1)
            self.grad += output.grad * other.value * self.value ** (other.value -1)
        # set the backpropagation function for power nodes
        output._backward = _bakward
        return output

    # TODO: this needs testing, weird behaviour, also the backward function, maybe not the same?!
    # maybe we have to get rid of this altogether
    def __rpow__(self, other):
        """Override reverse __pow__ magic method."""
        other = Scalar._check_instance(other)
        # unlike add and multiplication operations, power is order dependent!
        output = Scalar(other ** self.value, (self, other), "POW", "")
        return output

    def __truediv__(self, other):
        """True division (as opposed to floor division __div__) operator overload."""
        other = Scalar._check_instance(other)
        return self * other ** -1

    def __rtruediv__(self, other):
        """Override reverse __truediv__ magic method."""
        other = Scalar._check_instance(other)
        return other * self ** -1

    def __sub__(self, other):
        """Subtraction operator overload."""
        other = Scalar._check_instance(other)
        return self + (-other)

    def __neg__(self):
        """Rule for negating an Scalar."""
        return self * -1
    
    def exp(self):
        """Exponantiation operator for Scalars."""
        output = Scalar(math.exp(self.value), (self,), "EXP", "")
        def _backward():
            # calculate the local gradient wrt. loss in an exponantiation node
            # output = exp(self)
            # dL/self = dL/doutput * doutput/dself = output.grad * exp(self)
            self.grad += output.grad * output.value
        # set the backpropagation function for exponantiation nodes
        output._backward = _bakward
        return output

    def tanh(self):
        """Hyperbolic tangent function for Scalars."""
        output = Scalar(math.tanh(self.value), (self,), "TANH", "")
        def _backward():
            # calculate the local gradient wrt. loss in an tanh node
            # output = tanh(self)
            # dL/self = dL/doutput * doutput/dself = output.grad * (1 - tanh(self)**2)
            self.grad += output.grad * (1 - math.tanh(self.value)**2)
        # set the backpropagation function for tanh nodes
        output._backward = _bakward

    def backward(self):
        """Backpropagation function.

        This method first sorts the computational graph so that all 
        the nodes flow from left to right, since the backpropagation 
        has to be done from final result, e.g. loss function to the 
        starting nodes.
        """
        nodes = []
        visited = set()
        def tree_walk():
            if node not in visited:
                visited.add(node)
                for child in node.__prev:
                    tree_walk(child)
                nodes.append(node)
        tree_walk(self)
        self.grad = 1.0
        for node in reversed(nodes):
            node._backward()

    # TODO: check this method!
    def _check_instance(other):
        if not isinstance(other, (int, float, Scalar)):
            raise ValueError("Only operations on int, float or other Scalar objects are allowed.")
        # if the other object is an int or float instance, convert it to an Scalar object
        if not isinstance(other, Scalar):
            other = Scalar(other)
        return other

class Neuron:
    def __init__(self, num_in):
        # num_in: the number of inputs to the neuron
        self.w = [Scalar(random.uniform(-1, 1)) for _ in range(num_in)]
        self.b = Scalar(random.uniform(-1, 1))

    def __call__(self, x):
        """Forward pass of the neuron. Perform Neuron(w*x+b)."""
        activation = sum(wi*xi for wi, xi in zip(self.w, x)) + self.b
        ouput = activation.tanh()
        return output

    def parameters(self):
        """Return weights and bias of the neuron."""
        return self.w + [self.b]


class Layer:
    def __init__(self, num_in, num_out):
        # num_in:  number of inputs to a single Neuron
        # num_out: how many Neurons are in a Layer
        self.neurons = [Neuron(num_in) for _ in range(num_out)]

    def __call__(self, x):
        outputs = [n(x) for n in self.neurons]
        # if there is only one neuron output don't return a list
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        """Return parameters of all the neurons in the layer."""
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, num_in, num_out):
        # num_in:  number of inputs to a single Neuron
        # num_out: the dimension of the layers in the multi-layer perceptron
        size = [num_in] + num_out
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(num_out))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

