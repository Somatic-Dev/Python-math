# MATLAB-like Python environment imports
import numpy as np
from numpy.linalg import matrix_rank, solve, lstsq
from scipy import linalg
from scipy.linalg import eig, svd, qr, lu
import matplotlib.pyplot as plt
from sympy import Matrix, symbols, solve as sym_solve, simplify, factor, expand
from sympy.matrices import eye, zeros, ones, diag

# Example matrix
A = np.array([[-3, -5, 36, 10],
              [-1, 0, 7, -8],
              [1, 1, -10, -4]], dtype=float)

print("Matrix A:\n", A)

# Convert to RREF using sympy
A_rref = Matrix(A).rref()
print("\nRREF of A:\n", A_rref[0])
print("Pivot columns:", A_rref[1])

# Lab 1