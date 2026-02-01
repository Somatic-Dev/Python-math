# MATLAB-like Python environment imports
import matplotlib.pyplot as plt
from sympy import Matrix, symbols, solve as sym_solve, simplify, factor, expand, pprint, pretty
from sympy.matrices import eye, zeros, ones, diag

# ============================================
# Helper Functions
# ============================================

def print_matrix_equation(A, x, b):
    """Print matrices side by side: A * x = b using pretty print format"""
    # Get pretty string representations
    A_str = pretty(A).split('\n')
    x_str = pretty(x).split('\n')
    b_str = pretty(b).split('\n')
    
    # Find max heights and pad shorter ones
    max_height = max(len(A_str), len(x_str), len(b_str))
    
    def pad_to_height(lines, height):
        width = max(len(line) for line in lines) if lines else 0
        padding_top = (height - len(lines)) // 2
        padding_bottom = height - len(lines) - padding_top
        empty_line = ' ' * width
        return [empty_line] * padding_top + lines + [empty_line] * padding_bottom
    
    A_str = pad_to_height(A_str, max_height)
    x_str = pad_to_height(x_str, max_height)
    b_str = pad_to_height(b_str, max_height)
    
    # Find widths for alignment
    A_width = max(len(line) for line in A_str)
    x_width = max(len(line) for line in x_str)
    
    # Find middle row for operators
    mid = max_height // 2
    
    # Print each row
    for i in range(max_height):
        A_part = A_str[i].ljust(A_width)
        x_part = x_str[i].ljust(x_width)
        b_part = b_str[i]
        mult = " * " if i == mid else "   "
        equals = " = " if i == mid else "   "
        print(f"{A_part}{mult}{x_part}{equals}{b_part}")
    print()

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
A = Matrix([[-3, -5, 36, 10],
            [-1, 0, 7, -8],
            [1, 1, -10, -4]])

print_rref_and_equations(A, "A", var_names=['x', 'y', 'z'])

# Lab 1


# Lab 2
print("\n\n========== PROBLEM 1 ==========")
A2 = Matrix([[1, -1, 3],
             [2, 1, 3]])
print_rref_and_equations(A2, "A2", var_names=['x', 'y'])


print("\n\nProblem 2:")
B2 = Matrix([[10, 15, 20],
             [2, 3, 4]])
print_rref_and_equations(B2, "B2", var_names=['x', 'y'])

print("\n\nProblem 3:")
C2 = Matrix([[-4, 6, 3],
             [2, -3, -3]])
print_rref_and_equations(C2, "C2", var_names=['x', 'y'])

print("\n\nProblem 4:")
D2 = Matrix([[1,-1,-1,4],
             [2,4,6,-6],
             [1,3,-2,2]])
print_rref_and_equations(D2, "D2", var_names=['x', 'y', 'z'])

print("\n\nProblem 5:")
E2 = Matrix([[1,-1,1,4],
             [2,0,4,5],
             [2,0,-4,-5]])
print_rref_and_equations(E2, "E2", var_names=['x', 'y', 'z'])

print("\n\nProblem 6:")
F2 = Matrix([[1,-1,2,5],
             [2,1,0,1],
             [1,8,-1,3],
             [-1,-5,-12,41]])
print_rref_and_equations(F2, "F2", var_names=['x1', 'x2', 'x3'])

print("\n\nProblem 7:")
G2 = Matrix([[3,-1,1,2,-2],
             [1,2,-1,1,1],
             [-1,-3,2,-4,-6]])
print_rref_and_equations(G2, "G2", var_names=['x1', 'x2', 'x3', 'x4'])

# Lab 3


def MatrixInverseRRefSolve(A, b):
    """Solve Ax = B using the inverse matrix then RREF with inverse matrix"""
   
    print("\nUsing Inverse Matrix Method with identity matrix:\n")
    print("MatrixForm: Ax = b\n")

    # Create symbolic x vector
    n = A.shape[1]
    x = Matrix(symbols(f'x0:{n}'))
    
    # Print the system in nice matrix form (side by side)
    print("System of equations in matrix form:\n")
    print_matrix_equation(A, x, b)
    
    I = eye(A.shape[0]) # Identity matrix of size of A
    aug = A.row_join(I)
    print("\nAugmented Matrix [A | I]:\n")
    pprint(aug)
    rref_aug, _ = aug.rref()
    print("\nRREF of [A | I]:\n")
    pprint(rref_aug)
    A_inv = rref_aug[:, A.shape[1]:]
    print("\nInverse of A:\n")
    pprint(A_inv)
    x_inv = A_inv * b
    print("\nSolution x using Inverse Matrix Method:\n")
    pprint(x_inv)



print("\nProblem 1:\n")
A3_1 = Matrix([[1, -1],
                [2, 1]])
b3_1 = Matrix([[3],
                [3]])
MatrixInverseRRefSolve(A3_1, b3_1)

print("\nProblem 2:\n")
A3_2 = Matrix([[-4, -6],
               [2, -3]])
b3_2 = Matrix([[3],
               [-3]])
MatrixInverseRRefSolve(A3_2, b3_2)

print("\nProblem 3:\n")
A3_3 = Matrix([[1, -1],
               [2, 4]])
b3_3 = Matrix([[4],
               [14]])
MatrixInverseRRefSolve(A3_3, b3_3)

print("\nProblem 4:\n")
A3_4 = Matrix([[10, 15],
               [2, 3]])
b3_4 = Matrix([[20],
               [4]])
MatrixInverseRRefSolve(A3_4, b3_4)
print_rref_and_equations(A3_4.row_join(b3_4), "Augmented A3_4", var_names=['x', 'y'])


print("\nProblem 5:\n")
A3_5 = Matrix([[1, -1, -1],
               [2, 4, 6],
               [1, 3, -2]])
b3_5 = Matrix([[4],
               [-6],
               [2]])
MatrixInverseRRefSolve(A3_5, b3_5)


print("\nProblem 6:\n")
A3_6 = Matrix([[1, -1, 1],
               [2, 0, 4],
               [-2, 0, -4]])
b3_6 = Matrix([[4],
               [5],
               [-5]])
MatrixInverseRRefSolve(A3_6, b3_6)
print_rref_and_equations(A3_6.row_join(b3_6), "Augmented A3_6", var_names=['x', 'y', 'z'])