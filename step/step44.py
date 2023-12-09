import numpy as np
import dezero.layers as L
import dezero.functions as F
from dezero.core import Parameter, as_variable


def mean_squared_error(x0, x1):
    #x0,x1 = as_variable(x0), as_variable(x1)
    diff = x0 - x1
    print(diff)
    t1 = diff ** 2
    t2 = F.sum(t1)
    return t2 / len(diff)


np.random.seed(0)
x = np.random.rand(2, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(2, 1)
l1 = L.Linear(1)
l2 = L.Linear(1)


def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y


lr = 0.2
iters = 10000

for i in range(1):
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)
    l1.clear_grad()
    l2.clear_grad()
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data
    if i % 1000 == 0:
        pass
