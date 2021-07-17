import unittest
from dezero import Variable
import dezero.functions as F
import numpy as np


class ReshapeTest(unittest.TestCase):
    def test_reshape(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.reshape(x, (6,))
        y.backward(retain_grad=True)

        self.assertEqual(y.shape, (6,))
        self.assertTrue(np.allclose(y.data, np.array([1, 2, 3, 4, 5, 6])))
        self.assertTrue(np.allclose(x.grad.data, np.array([[1, 1, 1], [1, 1, 1]])))


class TransposeTest(unittest.TestCase):
    def test_transpose(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = x.T

        self.assertEqual(y.shape, (3, 2))
        self.assertTrue(np.allclose(y.data, np.array([[1, 4], [2, 5], [3, 6]])))


class SumBroadcastTest(unittest.TestCase):
    def test_sum(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = x.sum()
        y.backward()

        self.assertEqual(y.data, 21)
        self.assertTrue(np.allclose(x.grad.data, np.array([[1, 1, 1], [1, 1, 1]])))
        self.assertTrue(np.allclose(x.sum(axis=0).data, np.array([5, 7, 9])))
        self.assertTrue(np.allclose(x.sum(axis=1).data, np.array([6, 15])))
        self.assertTrue(
            np.allclose(x.sum(axis=0, keepdims=True).data, np.array([[5, 7, 9]]))
        )
        self.assertTrue(
            np.allclose(x.sum(axis=1, keepdims=True).data, np.array([[6], [15]]))
        )


class MatMulTest(unittest.TestCase):
    def test_matmul(self):
        x = Variable(np.random.randn(2, 3))
        W = Variable(np.random.randn(3, 4))
        y = F.matmul(x, W)
        y.backward()

        self.assertEqual(y.shape, (2, 4))
        self.assertEqual(x.grad.shape, (2, 3))
        self.assertEqual(W.grad.shape, (3, 4))


unittest.main()
