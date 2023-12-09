from dezero import Variable
import numpy as np


x0 = Variable(np.array([1,2,3]))
x1 = Variable(np.array([10]))
y = x0 + x1
print(y.data)

y.backward()
print(x1.grad.data)
