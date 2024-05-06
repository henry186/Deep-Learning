import numpy as np
import matplotlib.pyplot as plt

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)

    return(exp_a / sum_exp_a) 

x = np.arange(-0.5, 5.5, 0.1)
y = softmax(x)

plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()