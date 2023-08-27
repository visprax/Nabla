import numpy as np

def asarray(vals, *args, **kwargs):
    """Gradient supporting autograd asarray"""
    if isinstance(vals, np.ndarray):
        return np.asarray(vals, *args, **kwargs)
    return np.array(vals, *args, **kwargs)

class tensor(np.ndarray):
    def __new__(cls, input_array, *args, requires_grad=True, **kwargs):
        obj = asarray(input_array, *args, **kwargs)

        if isinstance(obj, np.ndarray):
            obj = obj.view(cls)
            obj.requires_grad = requires_grad

        return obj

    def __array_finalize__(self, obj):
        # pylint: disable=attribute-defined-outside-init
        if obj is None:  # pragma: no cover
            return

        self.requires_grad = getattr(obj, "requires_grad", None)

    def __repr__(self):
        string = super().__repr__()
        return string[:-1] + f", requires_grad={self.requires_grad})"

    def __array_wrap__(self, obj):
        out_arr = tensor(obj, requires_grad=self.requires_grad)
        return super().__array_wrap__(out_arr)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # pylint: disable=no-member,attribute-defined-outside-init

        # unwrap any outputs the ufunc might have
        outputs = [i.view(np.ndarray) for i in kwargs.get("out", ())]

        if outputs:
            # Insert the unwrapped outputs into the keyword
            # args dictionary, to be passed to ndarray.__array_ufunc__
            outputs = tuple(outputs)
            kwargs["out"] = outputs
        else:
            # If the ufunc has no ouputs, we simply
            # create a tuple containing None for all potential outputs.
            outputs = (None,) * ufunc.nout

        # unwrap the input arguments to the ufunc
        args = [i.unwrap() if hasattr(i, "unwrap") else i for i in inputs]

        # call the ndarray.__array_ufunc__ method to compute the result
        # of the vectorized ufunc
        res = super().__array_ufunc__(ufunc, method, *args, **kwargs)

        if ufunc.nout == 1:
            res = (res,)

        # construct a list of ufunc outputs to return
        ufunc_output = [
            (np.asarray(result) if output is None else output)
            for result, output in zip(res, outputs)
        ]

        # if any of the inputs were trainable, the output is also trainable
        requires_grad = any(
            isinstance(x, np.ndarray) and getattr(x, "requires_grad", True) for x in inputs
        )

        # Iterate through the ufunc outputs and convert each to a PennyLane tensor.
        # We also correctly set the requires_grad attribute.
        for i in range(len(ufunc_output)):  # pylint: disable=consider-using-enumerate
            ufunc_output[i] = tensor(ufunc_output[i], requires_grad=requires_grad)

        if len(ufunc_output) == 1:
            # the ufunc has a single output so return a single tensor
            return ufunc_output[0]

        # otherwise we must return a tuple of tensors
        return tuple(ufunc_output)

    def __getitem__(self, *args, **kwargs):
        item = super().__getitem__(*args, **kwargs)

        if not isinstance(item, tensor):
            item = tensor(item, requires_grad=self.requires_grad)

        return item

    def __hash__(self):
        if self.ndim == 0:
            # Allowing hashing if the tensor is a scalar.
            # We hash both the scalar value *and* the differentiability information,
            # to match the behaviour of PyTorch.
            return hash((self.item(), self.requires_grad))

        raise TypeError("unhashable type: 'numpy.tensor'")

    def __reduce__(self):
        # Called when pickling the object.
        # Numpy ndarray uses __reduce__ instead of __getstate__ to prepare an object for
        # pickling. self.requires_grad needs to be included in the tuple returned by
        # __reduce__ in order to be preserved in the unpickled object.
        reduced_obj = super().__reduce__()
        # The last (2nd) element of this tuple holds the data. Add requires_grad to this:
        full_reduced_data = reduced_obj[2] + (self.requires_grad,)
        return (reduced_obj[0], reduced_obj[1], full_reduced_data)

    def __setstate__(self, reduced_obj) -> None:
        # Called when unpickling the object.
        # Set self.requires_grad with the last element in the tuple returned by __reduce__:
        # pylint: disable=attribute-defined-outside-init,no-member
        self.requires_grad = reduced_obj[-1]
        # And call parent's __setstate__ without this element:
        super().__setstate__(reduced_obj[:-1])

    def unwrap(self):
        if self.ndim == 0:
            return self.view(np.ndarray).item()

        return self.view(np.ndarray)

    def numpy(self):
        return self.unwrap()

