import dezero.layers as L
from dezero import Variable
import dezero.functions as F
from dezero import Layer
import numpy as np


class TwoLayerNet(Layer):
    def __init__(self, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(output_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y

np.random.seed(0)
x = np.random.rand(100,1)
y = np.sin(2 * np.pi * x) + np.random.rand(100,1)
lr = 0.2
max_iter = 10000
hidden_size = 100

model = TwoLayerNet(100, 1)
for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)
    model.clear_grad()
    loss.backward()
    for p in model.params():
        p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(loss)
