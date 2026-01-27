# MATLAB-like Python environment imports
import numpy as np
from numpy.linalg import matrix_rank, solve, lstsq
from scipy import linalg
from scipy.linalg import eig, svd, qr, lu
import matplotlib.pyplot as plt
from sympy import Matrix, symbols, solve as sym_solve, simplify, factor, expand, pprint
from sympy.matrices import eye, zeros, ones, diag

# ============================================
# Helper Functions
# ============================================

def format_val(x):
    """Convert to int if no decimal part, otherwise keep as float"""
    return int(x) if x == int(x) else x

def print_equations(rref_matrix, var_names=None):
    """Print equations from RREF matrix."""
    num_vars = rref_matrix.cols - 1  # Last column is augmented
    
    if var_names is None:
        var_names = [chr(120 + i) for i in range(num_vars)]  # x, y, z, ...
    
    for i in range(rref_matrix.rows):
        row = rref_matrix[i, :]
        terms = []
        
        # Build left side, only including non-zero terms
        for j in range(num_vars):
            coeff = format_val(row[j])
            if coeff != 0:
                if coeff == 1:
                    terms.append(f"{var_names[j]}")
                elif coeff == -1:
                    terms.append(f"-{var_names[j]}")
                else:
                    terms.append(f"{coeff}*{var_names[j]}")
        
        # Join terms with + or -, handle signs properly
        if terms:
            left_side = " + ".join(terms).replace(" + -", " - ")
        else:
            left_side = "0"
        
        right_side = format_val(row[-1])
        print(f"{left_side} = {right_side}")

def print_rref_and_equations(matrix_array, matrix_name="Matrix", var_names=None):
    """
    Convert array to RREF, print augmented matrix, RREF, and equations.
    """
    # Convert to SymPy Matrix
    matrix = Matrix(matrix_array)
    
    print(f"\nAugmented Matrix {matrix_name}:\n")
    pprint(matrix)
    
    # Get RREF
    rref_matrix, pivot_cols = matrix.rref()
    
    print(f"\nRREF of {matrix_name}:\n")
    pprint(rref_matrix)
    # Increment pivot columns by 1 for 1-based indexing
    pivot_cols_1based = tuple(col + 1 for col in pivot_cols)
    print(f"Pivot columns: {pivot_cols_1based}")
    
    # Print equations
    print(f"\nEquations:")
    print_equations(rref_matrix, var_names)

# Example matrix
A = np.array([[-3, -5, 36, 10],
              [-1, 0, 7, -8],
              [1, 1, -10, -4]], dtype=float)

print_rref_and_equations(A, "A", var_names=['x', 'y', 'z'])

# Lab 1


# Lab 2
print("\n\n========== PROBLEM 1 ==========")
A2 = np.array([[1, -1, 3],
               [2, 1, 3]])
print_rref_and_equations(A2, "A2", var_names=['x', 'y'])


print("\n\nProblem 2:")
B2 = np.array([[10, 15, 20],
               [2, 3, 4]])
print_rref_and_equations(B2, "B2", var_names=['x', 'y'])

print("\n\nProblem 3:")
C2 = np.array([[-4, 6, 3],
               [2, -3, -3]])
print_rref_and_equations(C2, "C2", var_names=['x', 'y'])

print("\n\nProblem 4:")
D2 = np.array([[1,-1,-1,4],
               [2,4,6,-6],
               [1,3,-2,2]])
print_rref_and_equations(D2, "D2", var_names=['x', 'y', 'z'])

print("\n\nProblem 5:")
E2 = np.array([[1,-1,1,4],
               [2,0,4,5],
               [2,0,-4,-5]])
print_rref_and_equations(E2, "E2", var_names=['x', 'y', 'z'])

print("\n\nProblem 6:")
F2 = np.array([[1,-1,2,5],
               [2,1,0,1],
               [1,8,-1,3],
               [-1,-5,-12,41]])
print_rref_and_equations(F2, "F2", var_names=['x1', 'x2', 'x3'])

print("\n\nProblem 7:")
G2 = np.array([[3,-1,1,2,-2],
               [1,2,-1,1,1],
               [-1,-3,2,-4,-6]])
print_rref_and_equations(G2, "G2", var_names=['x1', 'x2', 'x3', 'x4'])