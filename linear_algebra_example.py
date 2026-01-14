"""
Linear Algebra Example using NumPy and SciPy
Similar to MATLAB linear algebra operations
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# ============================================
# Matrix Operations (like MATLAB)
# ============================================

# Create matrices (non-singular for solving)
A = np.array([[1, 2, 3],
              [4, 5, 7],
              [7, 8, 10]], dtype=float)

B = np.array([[9, 8, 7],
              [6, 5, 4],
              [3, 2, 1]], dtype=float)

print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)

# ============================================
# Basic Matrix Operations
# ============================================

print("\n--- Basic Operations ---")
print("A + B (addition):")
print(A + B)

print("\nA * B (element-wise multiplication):")
print(A * B)

print("\nA @ B (matrix multiplication):")
print(A @ B)

print("\nA.T (transpose):")
print(A.T)

# ============================================
# Linear System Solving (Ax = b)
# ============================================

print("\n--- Solving Linear Systems ---")
b = np.array([1, 2, 3])
x = np.linalg.solve(A, b)
print("Solution to Ax = b:")
print(x)

# ============================================
# Eigenvalues and Eigenvectors
# ============================================

print("\n--- Eigenvalues and Eigenvectors ---")
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)

# ============================================
# Matrix Decompositions
# ============================================

print("\n--- LU Decomposition ---")
P, L, U = linalg.lu(A)
print("L:\n", L)
print("U:\n", U)

print("\n--- QR Decomposition ---")
Q, R = np.linalg.qr(A)
print("Q:\n", Q)
print("R:\n", R)

print("\n--- SVD (Singular Value Decomposition) ---")
U_svd, S, Vt = np.linalg.svd(A)
print("Singular values:", S)

# ============================================
# Rank, Determinant, Inverse
# ============================================

print("\n--- Matrix Properties ---")
print("Rank of A:", np.linalg.matrix_rank(A))
print("Determinant of A:", np.linalg.det(A))

# Create an invertible matrix for demonstration
A_inv = np.array([[1, 2], [3, 4]], dtype=float)
print("Inverse of A_inv:")
print(np.linalg.inv(A_inv))

# ============================================
# Visualization Example
# ============================================

# Create a simple plot
x_vals = np.linspace(-10, 10, 100)
y_vals = 2 * x_vals + 3

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='y = 2x + 3')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Algebra Example - Linear Function')
plt.legend()
plt.savefig('linear_plot.png')
print("\nPlot saved as 'linear_plot.png'")
plt.close()

print("\n✓ Setup complete! You can now perform MATLAB-like linear algebra operations.")
