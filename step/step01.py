import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data


x = np.array(1.0)
a = Variable(x)
print(a.data)
