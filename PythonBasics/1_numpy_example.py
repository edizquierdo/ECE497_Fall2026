import numpy as np

# Create a 2D array of shape (2, 3)
x = np.array([[3.0, 4.0, 0.0],
              [1.0, 2.0, 2.0]])

# # Compute the norm along the last dimension (columns)
# result = np.linalg.norm(x, axis=-1)

# print(result)

shape = (2,3)
rand_array = np.random.rand(2,3)
rand = np.random.randn(10)*0.1
print(rand)
# zeros_array = np.zeros(shape)

# print(f"Random Array: \n {rand_array} \n")
# print(f"Ones Array: \n {array_array} \n")
# print(f"Zeros Array: \n {zeros_array}")

np.save(rand_array, "rand_array.npy")
