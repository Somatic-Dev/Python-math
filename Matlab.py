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
#  A=[■(1&1&1@1&2&3@4&5&6)] , B=[■(3@2@4)],   C=[■(0@1@1)]  ,  D=[■(2&1&3)] , E=[■(7&8&9)] 

A = np.array([[1, 1, 1],
              [1, 2, 3],
              [4, 5, 6]], dtype=float); 
B = np.array([[3],
              [2],
              [4]], dtype=float);
C = np.array([[0],
              [1],
              [1]], dtype=float);
D = np.array([[2, 1, 3]], dtype=float);
E = np.array([[7, 8, 9]], dtype=float);
M = np.array([[1,3,6,2],
              [0,1,4,5],
              [0,0,2,6]], dtype=float);
N = np.array([[1, 0, 0, 2, 1, 3],
              [0, 1, 0, 0, 4, 1],
              [0, 0, 1, 2, 3, 9]], dtype=float);


print("\n1. i")

hStackedAB = np.hstack((A, B))  # Augmented matrix [A|B]
print("Augmented matrix [A,B]:\n", hStackedAB)

print("\n1. ii")
hStackedBC = np.hstack((B, C))  # Augmented matrix [B|C]
print("Augmented matrix [B,C]:\n", hStackedBC)
print("\n1. iii")
HstackedDE = np.hstack((D, E))  # Augmented matrix [D|E]
print("Augmented matrix [D,E]:\n", HstackedDE)

print("\n1. iv")
VStackedBC   = np.vstack((B, C))  # Augmented matrix [B;C]
print("Augmented matrix [B;C]:\n", VStackedBC)

print("\n1. v")
VStackedAD = np.vstack((A, D))  # Augmented matrix [A;D]
print("Augmented matrix [A;D]:\n", VStackedAD)

print("\n1. vi")
VStackedDE = np.vstack((D, E))  # Augmented matrix [D;E]
print("Augmented matrix [D;E]:\n", VStackedDE)

print("\n2. i")
A_transpose = A.T
print("Transpose of A:\n", A_transpose)

print("\n2. ii")
B_transpose = B.T
print("Transpose of B:\n", B_transpose)

print("\n2. iii")
HStackedCE_transpose = np.hstack((C, E.T))
print("Transpose of E then HStack [C,E]:\n", HStackedCE_transpose)

print("\n2. iv")
HStackedCD_Transpose = np.hstack((C.T, D)) 
print("Transpose of C then HStack [C,D]:\n", HStackedCD_Transpose)

print("\n4. i")
AChangerow3column2 = A.copy()
AChangerow3column2[1, 2] = 9
print("Modified A (row 2, column 3 set to 9):\n", AChangerow3column2)

print("\n4. ii")
AChangeRow3 = A.copy()
AChangeRow3[2,:] = [1, 2, 4]
print("Modified A (row 3 set to [1,2,4]):\n", AChangeRow3)

print("\n4. iii")
AChangeRow1 = A.copy()
AChangeRow1[:, 0] = [1, 1, 1]
print("Modified A (Row 1 set to [1,1,1]):\n", AChangeRow1)

print("\n5. i")
submatrix_M_3x3 = M[0:3, 0:3]
print("Submatrix of M M1 (3x3):\n", submatrix_M_3x3)
print("\n5. ii")
submatrix_M_1x1 = M[0:3, 3:4]
print("Submatrix of M M2:\n", submatrix_M_1x1)
print("\n5. iii")
submatrix_N_3x3 = N[0:3, 0:3]
print("Submatrix of N N1:\n", submatrix_N_3x3)
print("\n5. iv")
submatrix_N_3x3x2 = N[0:3, 3:6]
print("Submatrix of N N2:\n", submatrix_N_3x3x2)



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