# deepX
A simple autograd engine with neural nets build on top of it.

### Components

At the heart of the most deep learning frameworks is an automatic differentiation engine that 
provides support for automatically calculating gradients of the loss function with respect to 
various parameters in the model. A typical deep learning library will consists of the following 
components:

- **Operators**

Operators or *layers* are vector-based functions that transform data. Some commonly used 
operators are layers like, linear, convolution, and pooling, and activation functions like 
ReLU and Sigmoid.

TODO: more intuitive description with explanations for ReLU and Sigmoid functions (with mathemtical functions).

- **Optimizers**

Optimizers update the model parameters using their gradients with respect to the optimization 
objective. Some well-known optimizers are Stochastic Gradient Descent (SGD), RMSProp, and Adam.

TODO: more intuitive description with explanations for SGD, RMSProp, and Adam optimizers (with mathematical functions).

- **Loss Functions**

Differentiable functions that are used for the optimization objective of the problem at hand. 
For example in classification problems it is common to use cross-entropy and Hinge loss functions.

TODO: mathematical expressions for common loss functions (cross-entropy, minimum squared error, and Hinge).

- **Initializers**

They provide the initial values for the model parameters at the start of training. For example 
one way to initialize the network weights is to draw small random weights from the normal distribution.

TODO: PyTorch or Keras initializers examples.

- **Regualarizers**

They provide the necessary mechanism to avoid overfitting and promote generalization. One can regulate 
overfitting either through explicit or implicit measures. Explicit methods impose structural constraint 
on the weights, for example minimization of their L1-Norm and L2-Norm that make the weights sparser and 
uniform respectively. Implicit measures are specialized operators that do the transformation of intermediate 
representations, either through explicit normalization, for example BatchNorm, or by changing the network 
connectivity, for example DropOut and DropConnect.

TODO: A better and more intuitive explanation of regularization and L1-Norm, L2-Norm, BatchNorm, ...


##### Back Propagation

TODO: explanation using the computation graph (e.g. Wikipedia graph)

#### References:
- [Deep Learning Framework From Scratch Using Numpy](https://arxiv.org/pdf/2011.08461.pdf)
- [Implementing a Deep Learning Library from Scratch](https://www.kdnuggets.com/2020/09/implementing-deep-learning-library-scratch-python.html)
