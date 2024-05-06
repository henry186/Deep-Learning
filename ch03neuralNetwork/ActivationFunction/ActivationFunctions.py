import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return np.array(x > 0, dtype=np.int64)

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(-5.0, 5.0, 0.1)
y1 = step_function(x)
y2 = sigmoid_function(x)
plt.plot(x,y1, linestyle = "--", label = "Step function")
plt.plot(x,y2,label = "Sigmoid function")
plt.ylim(-0.1, 1.1)
plt.legend()
plt.show()