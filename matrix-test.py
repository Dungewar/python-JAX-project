import numpy as np

# Create 3x3 matrix filled with zeros
zero_matrix: np.ndarray = np.zeros((3, 3))
for i in range(zero_matrix.shape[0]):
    for j in range(zero_matrix.shape[1]):
        zero_matrix[i, j] = i + 3 * j

one_matrix: np.ndarray = zero_matrix + 1

print(zero_matrix)
print(one_matrix)
print(zero_matrix @ one_matrix)
