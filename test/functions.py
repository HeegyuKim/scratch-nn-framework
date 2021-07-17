import unittest
from dezero import *


class SquareTest(unittest.TestCase):
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
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.array(1))
        y = square(x)
        y.backward()

        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
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


unittest.main()
