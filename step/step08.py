import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        func_list = [self.creator]
        while func_list:
            func = func_list.pop()
            x, y = func.input, func.output
            x.grad = func.backward(y.grad)
            if x.creator is not None:
                func_list.append(x.creator)


class Function:
    def __call__(self, input):
        self.input = input
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.output = output
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


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, dy):
        return np.exp(self.input.data) * dy


A = Square()
B = Exp()
x = Variable(np.array(3))
a = A(x)
b = B(a)
b.grad = np.array(1)
b.backward()
print(x.grad)