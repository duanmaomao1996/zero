import numpy as np
import dezero.functions as F

from dezero import Variable

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = x.sum(axis=1)
print(y.data)
