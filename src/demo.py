import numpy as np

# Initial arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Use np.tile to repeat them
a_tiled = np.tile(a, (len(b), 1))
b_tiled = np.tile(b, (len(a), 1)).T

# Print the results
print("a_tiled:")
print(a_tiled)

print("b_tiled:")
print(b_tiled)

print("a_tiled + b_tiled:")
print(a_tiled + b_tiled)
