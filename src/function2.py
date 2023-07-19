import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function definitions
def func_collision(x):
    return np.exp(x)-1
    # return np.tanh(0.5*(x-5))+1

def func_arrive(x):
    return np.exp(-(x-5))
    # return 1-np.tanh(0.5*(x-0.3))

# Create a grid of points
x1 = np.linspace(0, 5, 500)  # distance to obstacle
x2 = np.linspace(0, 5, 500)  # distance to goal
x1_grid, x2_grid = np.meshgrid(x1, x2)

# Compute y values
y = func_collision(x1_grid) + func_arrive(x2_grid)

# Compute gradients
grad_collision = abs(np.gradient(func_collision(x1)))
grad_arrive = abs(np.gradient(func_arrive(x2)))

# Create a figure
fig = plt.figure(figsize=(18,6))

# Add 3d subplot for the function plot
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(x1_grid, x2_grid, y)
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_zlabel('Y')
ax1.set_title('Function Plot')

# Add subplot for the gradient plot
ax2 = fig.add_subplot(132)
ax2.plot(x1, grad_collision, label='Gradient of Collision Function')
ax2.plot(x2, grad_arrive, label='Gradient of Arrive Function')
ax2.set_xlabel('X')
ax2.set_ylabel('Gradient')
ax2.legend()
ax2.set_title('Gradient Plot')

# Add 3d subplot for the combined gradient plot
grad_combined = np.tile(grad_collision, (len(grad_arrive),1)) + np.tile(grad_arrive, (len(grad_collision),1)).T
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(x1_grid, x2_grid, grad_combined)
ax3.set_xlabel('X1')
ax3.set_ylabel('X2')
ax3.set_zlabel('Gradient of collison + arrive')
ax3.set_title('Combined Gradient Plot')

# Show the plot
plt.tight_layout()
plt.show()
