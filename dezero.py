import numpy as np
import weakref
import contextlib


@contextlib.contextmanager
def using_config(name, value):
    old_attr = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    except:
        setattr(Config, name, old_attr)
        

def as_array(x):
    return np.array(x) if np.isscalar(x) \
        else x

class Config:
    enable_backprop = True
    
    
class Variable:
    
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
        
    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
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
                gxs = (gxs, )
            
            for x, gx in zip(f.inputs, gxs):
                x.grad = gx if x.grad is None else x.grad + gx
            
                if x.creator is not None:
                    add_func(x.creator)
                    
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None
    
    def cleargrad(self):
        self.grad = None
        
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
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return f"variable({p})"

    
class Function:
    
    def __call__(self, *inputs):
        xs = [x.data for x in inputs] 
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
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
        return (x + y, )
    
    def backward(self, gy):
        return gy, gy
    
class Mul(Function):
    def forward(self, x, y):
        return x * y
    def backward(self, gy):
        return self.inputs[0].data * gy, self.inputs[1].data * gy
    
class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.inputs[0].data
        return np.exp(x) * gy
    
def add(x, y):
    return Add()(x, y)

def mul(x, y):
    return Mul()(x, y)

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)
    
Variable.__add__ = add
Variable.__mul__ = mul
    
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(as_array(x.data - eps))
    x1 = Variable(as_array(x.data + eps))
    y0, y1 = f(x0), f(x1)
    return (y1.data - y0.data) / (2 * eps)

