import numpy as np
import matplotlib.pyplot as plt

# Function definition
def func_collision(x):
    return np.exp(x)-1
    # return np.tanh(0.5*(x-5))+1

def func_arrive(x):
    return np.exp(-(x-5))

if __name__ == '__main__':
    # Generate x values
    x1 = np.linspace(0, 5, 500)  # 500 points between 0 and 1
    x2 = np.linspace(0, 5, 500)  # 500 points between 0 and 1
    # Compute y values
    y_collision = func_collision(x1)
    y_arrive = func_arrive(x2)

    # Plot
    fig, axs = plt.subplots(2, figsize=(8,10))
    # Plot func1
    axs[0].plot(x1, y_collision, label='collision')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('Plot of the Collision Function')
    axs[0].grid(True)

    # Plot func2
    axs[1].plot(x2, y_arrive, label='arrive')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    axs[1].set_title('Plot of the Arrive Function')
    axs[1].grid(True)

    # Show plot
    plt.tight_layout()
    plt.show()
