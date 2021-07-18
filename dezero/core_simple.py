import numpy as np
import weakref
import contextlib


def as_variable(obj):
    return obj if isinstance(obj, Variable) else Variable(obj)


@contextlib.contextmanager
def using_config(name, value):
    old_attr = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    except:
        setattr(Config, name, old_attr)


def as_array(x):
    return np.array(x) if np.isscalar(x) else x


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(as_array(x.data - eps))
    x1 = Variable(as_array(x.data + eps))
    y0, y1 = f(x0), f(x1)
    return (y1.data - y0.data) / (2 * eps)


class Config:
    enable_backprop = True


class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None and not isinstance(data, np.ndarray):
            raise TypeError(f"{type(data)}는 지원하지 않습니다.")

        self.data = data
        self.name = None
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                x.grad = gx if x.grad is None else x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def cleargrad(self):
        self.grad = None

    def reshape(self, *shape):
        from .functions import reshape

        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return reshape(self, shape)

    def transpose(self):
        from .functions import transpose

        return transpose(self)

    def sum(self, axis=None, keepdims=False):
        from .functions import sum

        return sum(self, axis, keepdims)

    @property
    def T(self):
        return self.transpose()

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return f"variable({p})"


class Parameter(Variable):
    pass


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, x):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        return (x + y,)

    def backward(self, gy):
        gx, gy = gy, gy

        if self.x_shape != self.y_shape:
            from . import functions as F

            gx = F.sum_to(gx, self.x_shape)
            gy = F.sum_to(gy, self.y_shape)

        return gx, gy


class Mul(Function):
    def forward(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        return x * y

    def backward(self, gy):
        x, y = self.inputs
        gx = gy * y
        gy = gy * x

        if self.x_shape != self.y_shape:
            from . import functions as F

            gx = F.sum_to(gx, self.x_shape)
            gy = F.sum_to(gy, self.y_shape)

        return gx, gy


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0]
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs[0]
        return np.exp(x) * gy


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 - x1

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy
        gx1 = -gy

        if self.x0_shape != self.x1_shape:
            from . import functions as F

            gx0 = F.sum_to(gx0, self.x0_shape)
            gx1 = F.sum_to(gx1, self.x1_shape)

        return gx0, gx1


class Div(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 / x1

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)

        if self.x0_shape != self.x1_shape:
            from . import functions as F

            gx0 = F.sum_to(gx0, self.x0_shape)
            gx1 = F.sum_to(gx1, self.x1_shape)

        return gx0, gx1


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x0):
        return x0 ** self.c

    def backward(self, gy):
        x = self.inputs[0]
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


def add(x, y):
    return Add()(x, as_array(y))


def sub(x0, x1):
    return Sub()(x0, as_array(x1))


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, as_array(x0))


def mul(x, y):
    return Mul()(x, as_array(y))


def div(x0, x1):
    return Div()(x0, as_array(x1))


def rdiv(x0, x1):
    return Div()(as_array(x1), x0)


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


def pow(x, c):
    return Pow(c)(x)


def neg(x):
    return Neg()(x)


def setup_variable():
    Variable.__add__ = add
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    Variable.__neg__ = neg
