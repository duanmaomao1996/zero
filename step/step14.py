import weakref

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
            gys = [output.grad for output in func.outputs]
            gxs = func.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(func.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    func_list.append(x.creator)


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.outputs = [weakref.ref(output) for output in outputs]
        self.inputs = inputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, dy):
        x = self.inputs[0].data
        return 2 * x * dy


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, dy):
        return np.exp(self.input.data) * dy


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


def add(x0, x1):
    return Add()(x0, x1)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


x = Variable(np.array(2.0))
a = square(x)
y = add(square(a),square(a))
y.backward()

print(y.data)
print(x.grad)