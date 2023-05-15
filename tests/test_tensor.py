import unittest
import numpy as np

from nabla.tensor import Tensor

class TestNabla(unittest.TestCase):
    def test_tensor_init(self):
        a = Tensor([1.0, 2.0], dtype=np.float64, requires_grad=False)
        assert np.isclose(a.data[0], 1.0)
        assert a.requires_grad == False
        assert a.dtype == np.float64

if __name__ == "__main__":
    unittest.main()
