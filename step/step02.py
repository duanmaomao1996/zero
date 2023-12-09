import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError


class Square(Function):
    def forward(self, x):
        return x ** 2


def square(input):
    return Square()(input)


x = np.array(2.0)
y = Variable(x)
print(square(y).data)
