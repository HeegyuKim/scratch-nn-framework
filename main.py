import numpy as np
from dezero import *


x = Variable(np.array(2))
y = add(x, x)
z = add(y, x)
z.backward()

print(x.grad, y.grad, z.grad)
