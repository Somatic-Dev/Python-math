# Python Linear Algebra for MATLAB-like Computing

This workspace is configured for linear algebra projects using Python with MATLAB-comparable functionality.

## Setup Complete ✓

### Installed Packages
- **NumPy** - Fundamental package for numerical computing
- **SciPy** - Scientific and technical computing functions
- **Matplotlib** - Plotting and visualization
- **Pandas** - Data analysis and manipulation
- **Jupyter** - Interactive notebooks for exploratory work
- **IPython** - Enhanced interactive Python shell

### Installed VS Code Extensions
- **Python** - Official Python extension with linting, debugging, and intellisense
- **Pylance** - Advanced Python language intelligence

## Quick Start

### Running Python Scripts
```powershell
cd "d:\Assignments\2270\Python math"
.\.venv\Scripts\python.exe your_script.py
```

### Common Linear Algebra Operations

#### Creating Matrices
```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.ones((3, 3))
C = np.zeros((2, 4))
D = np.eye(3)  # Identity matrix
```

#### Matrix Operations
```python
# Element-wise operations
A + B
A - B
A * B  # Element-wise multiplication

# Matrix multiplication
A @ B
np.dot(A, B)

# Transpose
A.T

# Trace
np.trace(A)
```

#### Solving Linear Systems (Ax = b)
```python
A = np.array([[1, 2], [3, 4]], dtype=float)
b = np.array([5, 6])
x = np.linalg.solve(A, b)
```

#### Eigenvalues and Eigenvectors
```python
eigenvalues, eigenvectors = np.linalg.eig(A)
```

#### Matrix Decompositions
```python
# QR Decomposition
Q, R = np.linalg.qr(A)

# Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(A)

# LU Decomposition
from scipy import linalg
P, L, U = linalg.lu(A)
```

#### Matrix Properties
```python
# Rank
rank = np.linalg.matrix_rank(A)

# Determinant
det = np.linalg.det(A)

# Inverse
A_inv = np.linalg.inv(A)

# Norm
norm = np.linalg.norm(A)
```

#### Visualization
```python
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
y = 2*x + 3
plt.plot(x, y)
plt.show()
```

## Example Project

Run the example script to verify the setup:
```powershell
.\.venv\Scripts\python.exe linear_algebra_example.py
```

This demonstrates:
- Matrix operations (addition, multiplication, transpose)
- Linear system solving
- Eigenvalue decomposition
- Matrix decompositions (LU, QR, SVD)
- Matrix properties (rank, determinant, inverse)
- Visualization with Matplotlib

## Virtual Environment

The project uses a Python virtual environment (`.venv`) to manage dependencies. All packages are installed in this isolated environment.

Python Version: 3.11.9

## Additional Resources

### NumPy Documentation
- Official: https://numpy.org/doc/

### SciPy Linear Algebra
- Official: https://docs.scipy.org/doc/scipy/reference/linalg.html

### Matplotlib Visualization
- Official: https://matplotlib.org/

## Project Structure
```
d:\Assignments\2270\Python math\
├── .venv/                          # Virtual environment
├── linear_algebra_example.py        # Example script
├── linear_plot.png                 # Example output
└── README.md                        # This file
```

---

You're all set! Start creating your linear algebra projects with Python.
