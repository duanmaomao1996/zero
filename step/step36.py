import numpy as np
from dezero import Variable


x = Variable(np.array(2.0))
y = x ** 2
y.backward()
gx = x.grad
x.clear_grad()
z = gx ** 3 + y
z.backward()
print(x.grad.data)