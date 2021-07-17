import unittest
import numpy as np
from dezero import *


class VariableTest(unittest.TestCase):
    def test_reshape(self):
        x = Variable(np.array([1, 2, 3, 4]))
        y = x.reshape(2, 2)
        z = x.reshape((2, 2))

        self.assertEqual(y.shape, (2, 2))
        self.assertEqual(z.shape, (2, 2))
        self.assertTrue(np.allclose(y.data, z.data))


class OperatorTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad.data, expected)

    def test_gradient_check(self):
        x = Variable(np.array(1))
        y = square(x)
        y.backward()

        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad.data, num_grad)
        self.assertTrue(flg)

    def test_add(self):
        x = Variable(np.array(2))
        self.assertEqual((x + 2).data, 4.0)
        self.assertEqual((2 + x).data, 4.0)
        self.assertEqual((x + x).data, 4.0)

    def test_sub(self):
        x = Variable(np.array(2))
        self.assertEqual((x - 2).data, 0.0)
        self.assertEqual((2 - x).data, 0.0)
        self.assertEqual((x - x).data, 0.0)

    def test_mul(self):
        x = Variable(np.array(2))
        self.assertEqual((x * 2).data, 4.0)
        self.assertEqual((2 * x).data, 4.0)
        self.assertEqual((x * x).data, 4.0)

    def test_div(self):
        x = Variable(np.array(2))
        self.assertEqual((x / 2).data, 1.0)
        self.assertEqual((2 / x).data, 1.0)
        self.assertEqual((x / x).data, 1.0)

    def test_pow(self):
        x = Variable(np.array(2))
        self.assertEqual((x ** 2).data, 4.0)


class DifferentiationTest(unittest.TestCase):
    def test_sphere(self):
        def sphere(x, y):
            return x ** 2 + y ** 2

        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = sphere(x, y)
        z.backward()

        self.assertEqual(x.grad.data, 2.0)
        self.assertEqual(y.grad.data, 2.0)

    def test_matyas(self):
        def matyas(x, y):
            return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y

        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = matyas(x, y)
        z.backward()

        self.assertTrue(np.isclose(x.grad.data, 0.04))
        self.assertTrue(np.isclose(y.grad.data, 0.04))

    def test_goldstein(self):
        def goldstein(x, y):
            return (
                1
                + (x + y + 1) ** 2
                * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)
            ) * (
                30
                + (2 * x - 3 * y) ** 2
                * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2)
            )

        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = goldstein(x, y)
        z.backward()

        self.assertTrue(np.isclose(x.grad.data, -5376))
        self.assertTrue(np.isclose(y.grad.data, 8064))


unittest.main()
