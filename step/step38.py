import numpy as np
from dezero import Variable
import dezero.functions as F


x = Variable(np.random.randn(1,2,3))
y = x.transpose()
print(y.data)
z = x.reshape((2,3))
print(z.data)

