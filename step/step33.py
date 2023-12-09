import numpy as np
from dezero import Variable


def f(x):
    y = x ** 3 - 2 * x ** 2
    return y


x = Variable(np.random.rand(2,3))
y = x.transpose()

