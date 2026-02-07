# MATLAB-like Python environment imports
import numpy as np
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



#Lab 4 
print("\nProblem 1:\n")
A4_1 = Matrix([[-1, -3],
               [4, 5]])
b4_1 = Matrix([[-2],
               [8]])

print("\nDeterminant of A:")
pprint(A4_1.det())

MatrixInverseRRefSolve(A4_1, b4_1)

print("\nProblem 2:\n")
A4_2 = Matrix([[1, 2, 3],
               [2, 5, 3],
               [1, 0, 8]])
b4_2 = Matrix([[5],
               [3],
               [17]])

print("\nDeterminant of A:")
pprint(A4_2.det())
MatrixInverseRRefSolve(A4_2, b4_2)


print("\nProblem 3:\n")
A4_3 = Matrix([[0, 3, 2],
               [1, 2, 1],
               [2, 1, 1]])

print ("\nDeterminant of A:")
pprint(A4_3.det())

#swap row1 and row 2 of A and then calculate a new det
A4_3_swapped = A4_3.copy()
A4_3_swapped.row_swap(0, 1)
print("\nDeterminant of A after swapping row 1 and row 2:")
pprint(A4_3_swapped.det())

#Multiply row 2 by 3 and compute new det
A4_3_multiplied = A4_3.copy()
A4_3_multiplied.row_op(1, lambda v, j: 3 * v)
print("\nDeterminant of A after multiplying row 2 by 3:")
pprint(A4_3_multiplied.det())


#replace row 3 with -2 times row 2 plus row 3 and then calculate a new det
A4_3_replaced = A4_3.copy()
A4_3_replaced.row_op(2, lambda v, j: -2 * A4_3_replaced[1, j] + v)
print("\nDeterminant of A after replacing row 3 with -2 times row 2 plus row 3:")
pprint(A4_3_replaced.det())


print("\nProblem 4:\n")
A4_4 = Matrix([[2, 4, 6],
               [2, 6, 9],
               [3, 12, 5]])

print("\nDeterminant of A:")
pprint(A4_4.det())

#perform row operations in succession
A4_4_row_equivalent = A4_4.copy()

#multiply row 1 by .5 compute new det
A4_4_row_equivalent.row_op(0, lambda v, j: 0.5 * v)
print("\nDeterminant of A after multiplying row 1 by 0.5:")
pprint(A4_4_row_equivalent.det())

#replace row 2 with -2 times row 1 plus row 2 and compute new det
A4_4_row_equivalent.row_op(1, lambda v, j: -2 * A4_4_row_equivalent[0, j] + v)
print("\nDeterminant of A after replacing row 2 with -2 times row 1 plus row 2:")
pprint(A4_4_row_equivalent.det())

#replace row 3 with -3 times row 1 plus row 3 and compute new det
A4_4_row_equivalent.row_op(2, lambda v, j: -3 * A4_4_row_equivalent[0, j] + v)
print("\nDeterminant of A after replacing row 3 with -3 times row 1 plus row 3:")
pprint(A4_4_row_equivalent.det())

# replace row 3 with -3 times row 2 plus row 3 and compute new det
A4_4_row_equivalent.row_op(2, lambda v, j: -3 * A4_4_row_equivalent[1, j] + v)
print("\nDeterminant of A after replacing row 3 with -3 times row 2 plus row 3:")
pprint(A4_4_row_equivalent.det())

print("\nRow equivalent matrix to A after performing all row operations:\n")
pprint(A4_4_row_equivalent)

#write equation that relates the 2 determinants together 
print("\nThe determinant of the row equivalent matrix is related to the determinant of A by the following equation:")
print("det(A4_4_row_equivalent) = (0.5) * det(A4_4)")

print("\nProblem 5:\n")
A4_5 = Matrix([[2, 6, 8],
               [1, 2, 3],
               [3, 0, 2]])

#compute determinant of A
print("\nDeterminant of A:")
pprint(A4_5.det())

#turn A into an upper triangular REF matrix using row operations and compute new determinant
A4_5_ref = A4_5.copy()
# at each step identify the relationship of the determinant of the current matrix with the original determinant of A
A4_5_ref.row_op(1, lambda v, j: v - 0.5 * A4_5_ref[0, j])  # R2 - 0.5*R1
print("\nDeterminant after R2 - 0.5*R1:")
pprint(A4_5_ref.det())
A4_5_ref.row_op(2, lambda v, j: v - 1.5 * A4_5_ref[0, j])  # R3 - 1.5*R1
print("\nDeterminant after R3 - 1.5*R1:")
pprint(A4_5_ref.det())
A4_5_ref.row_op(2, lambda v, j: v - 0.5 * A4_5_ref[1, j])  # R3 - 0.5*R2
print("\nDeterminant after R3 - 0.5*R2:")
pprint(A4_5_ref.det())
A4_5_ref.row_op(0, lambda v, j: 0.5 * v)  # 0.5*R1
print("\nDeterminant after 0.5*R1:")
pprint(A4_5_ref.det())
A4_5_ref.row_op(1, lambda v, j: -1 * v)  # -1*R2
print("\nDeterminant after -1*R2:")
pprint(A4_5_ref.det())
A4_5_ref.row_op(2, lambda v, j: v + 8.5 * A4_5_ref[1, j]) 
print("\nDeterminant after R3 - 8.5*R2:")
pprint(A4_5_ref.det())

A4_5_ref.row_op(2, lambda v, j: -1 * v)  # -1*R3
print("\nDeterminant after -1*R3:")
pprint(A4_5_ref.det())

pprint("\nUpper triangular REF of A:\n")
pprint(A4_5_ref)

print("\n problem 6:\n")
A4_6 = Matrix([[1, -1, 2, 4],
               [2, 1, 0, 3],
               [1, 8, -1, -2],
               [-1, -5, -12, 1]])
b4_6 = Matrix([[-1],
               [8],
               [11],
               [30]])

print("\nDeterminant of A:")
pprint(A4_6.det())

#create the cramer's rule matrix for x1, x2, x3, x4 and compute the determinant of each printing them inbetween
A4_6_x1 = A4_6.copy()
A4_6_x1[:, 0] = b4_6
pprint("\nMatrix for Cramer's rule to solve for x1 (column 1 replaced by b):\n")
pprint(A4_6_x1)
print("\nDeterminant of A with column 1 replaced by b:")
pprint(A4_6_x1.det())

A4_6_x2 = A4_6.copy()
A4_6_x2[:, 1] = b4_6
pprint("\nMatrix for Cramer's rule to solve for x2 (column 2 replaced by b):\n")
pprint(A4_6_x2)
print("\nDeterminant of A with column 2 replaced by b:")
pprint(A4_6_x2.det())

A4_6_x3 = A4_6.copy()
A4_6_x3[:, 2] = b4_6
pprint("\nMatrix for Cramer's rule to solve for x3 (column 3 replacedby b):\n")
pprint(A4_6_x3)
print("\nDeterminant of A with column 3 replaced by b:")
pprint(A4_6_x3.det())

A4_6_x4 = A4_6.copy()
A4_6_x4[:, 3] = b4_6
pprint("\nMatrix for Cramer's rule to solve for x4 (column 4 replaced by b):\n")
pprint(A4_6_x4)
print("\nDeterminant of A with column 4 replaced by b:")
pprint(A4_6_x4.det())


#solve for x1, x2, x3, x4 using Cramer's rule and print the solution
x1 = A4_6_x1.det() / A4_6.det()
x2 = A4_6_x2.det() / A4_6.det()
x3 = A4_6_x3.det() / A4_6.det()
x4 = A4_6_x4.det() / A4_6.det()
print("\nSolution using Cramer's rule:")
pprint(Matrix([[x1], [x2], [x3], [x4]])) 