import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# define function
def f(x0, x1):
    return x0**2 + x1**2

# make data
x0 = np.linspace(-3, 3, 50)
x1 = np.linspace(-3, 3, 50)

print("before meshgrid: " + str(x0.shape))
x0, x1 = np.meshgrid(x0, x1)
y = f(x0, x1)

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x0, x1, y, cmap='viridis')

# set the labels and names 
ax.set_xlabel('X0')
ax.set_ylabel('X1')
ax.set_zlabel('f(X0, X1)')
ax.set_title('Surface plot of f(X0, X1) = X0^2 + X1^2')

# show the plot
plt.show()