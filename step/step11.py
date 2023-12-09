import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        func_list = [self.creator]
        while func_list:
            func = func_list.pop()
            x, y = func.input, func.output
            x.grad = func.backward(y.grad)
            if x.creator is not None:
                func_list.append(x.creator)


class Function:
    def __call__(self, inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.outputs = outputs
        self.inputs = inputs
        return outputs

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


class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y,)


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


xs = [Variable(np.array(2)), Variable(np.array(3))]
f = Add()
ys = f(xs)
y = ys[0]
print(y.data)
