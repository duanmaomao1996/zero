import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None


class Function:
    def __call__(self, input):
        self.input = input
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, dy):
        return 2 * self.input.data * dy


def square(input):
    return Square()(input)


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, dy):
        return np.exp(self.input.data) * dy


x = np.array(3.0)
y = Variable(x)
A = Square()
B = Exp()
a = A(y)
b = B(a)
b.grad = np.array(1.0)
a.grad = B.backward(b.grad)
y.grad = A.backward(a.grad)
print(y.grad)
