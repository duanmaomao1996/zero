import numpy as np
import dezero.functions as F
from dezero import Variable


x = Variable(np.array(np.pi/2))
y = F.sin(x)
print(y.data)