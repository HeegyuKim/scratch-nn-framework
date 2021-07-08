import numpy as np
from dezero import *


x = Variable(np.array(2))
z = add(x, x)
z = add(z, x)
z.backward()

print(z.data, x.grad)
