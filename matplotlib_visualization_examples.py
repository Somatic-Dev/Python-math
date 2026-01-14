"""
Matplotlib Visualization Examples for Linear Algebra
Demonstrates various plot types similar to MATLAB
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')

# ============================================
# 1. Basic Line Plots
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Matplotlib Visualization Examples', fontsize=16, fontweight='bold')

# Linear function
x = np.linspace(-10, 10, 100)
y1 = 2*x + 3
axes[0, 0].plot(x, y1, 'b-', linewidth=2, label='y = 2x + 3')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_title('Linear Function')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('y')
axes[0, 0].legend()

# Polynomial function
x = np.linspace(-3, 3, 100)
y2 = x**3 - 2*x**2 + x - 1
axes[0, 1].plot(x, y2, 'r-', linewidth=2, label='y = x³ - 2x² + x - 1')
axes[0, 1].fill_between(x, y2, alpha=0.3, color='red')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_title('Polynomial Function (with fill)')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('y')
axes[0, 1].legend()

# Trigonometric functions
x = np.linspace(0, 2*np.pi, 200)
y_sin = np.sin(x)
y_cos = np.cos(x)
axes[1, 0].plot(x, y_sin, 'g-', linewidth=2, label='sin(x)')
axes[1, 0].plot(x, y_cos, 'm-', linewidth=2, label='cos(x)')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_title('Trigonometric Functions')
axes[1, 0].set_xlabel('x (radians)')
axes[1, 0].set_ylabel('y')
axes[1, 0].legend()

# Multiple scatter plots with regression line
x_scatter = np.random.randn(50) * 2
y_scatter = 2 * x_scatter + np.random.randn(50)
z = np.polyfit(x_scatter, y_scatter, 1)
p = np.poly1d(z)
x_line = np.linspace(x_scatter.min(), x_scatter.max(), 100)
axes[1, 1].scatter(x_scatter, y_scatter, color='blue', s=50, alpha=0.6, label='Data')
axes[1, 1].plot(x_line, p(x_line), 'r-', linewidth=2, label='Fit line')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_title('Scatter Plot with Regression Line')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('y')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('matplotlib_example_1.png', dpi=150, bbox_inches='tight')
print("✓ Saved: matplotlib_example_1.png")

# ============================================
# 2. Heatmaps and Matrix Visualization
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Matrix Visualization', fontsize=14, fontweight='bold')

# Create a correlation matrix
data = np.random.randn(100, 4)
correlation_matrix = np.corrcoef(data.T)
im1 = axes[0].imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
axes[0].set_title('Correlation Heatmap')
axes[0].set_xticks(range(4))
axes[0].set_yticks(range(4))
axes[0].set_xticklabels(['Var1', 'Var2', 'Var3', 'Var4'])
axes[0].set_yticklabels(['Var1', 'Var2', 'Var3', 'Var4'])
plt.colorbar(im1, ax=axes[0])

# Create a random matrix
random_matrix = np.random.randn(10, 10)
im2 = axes[1].imshow(random_matrix, cmap='viridis', aspect='auto')
axes[1].set_title('Random Matrix Visualization')
axes[1].set_xlabel('Column')
axes[1].set_ylabel('Row')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig('matplotlib_example_2.png', dpi=150, bbox_inches='tight')
print("✓ Saved: matplotlib_example_2.png")

# ============================================
# 3. 3D Plots
# ============================================

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(14, 5))

# 3D Surface plot
ax1 = fig.add_subplot(121, projection='3d')
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9)
ax1.set_title('3D Surface Plot: sin(√(x² + y²))')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
fig.colorbar(surf, ax=ax1, shrink=0.5)

# 3D Scatter plot
ax2 = fig.add_subplot(122, projection='3d')
n_points = 100
x_3d = np.random.randn(n_points)
y_3d = np.random.randn(n_points)
z_3d = np.random.randn(n_points)
colors = np.sqrt(x_3d**2 + y_3d**2 + z_3d**2)
scatter = ax2.scatter(x_3d, y_3d, z_3d, c=colors, cmap='plasma', s=50, alpha=0.6)
ax2.set_title('3D Scatter Plot')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
fig.colorbar(scatter, ax=ax2, shrink=0.5)

plt.tight_layout()
plt.savefig('matplotlib_example_3.png', dpi=150, bbox_inches='tight')
print("✓ Saved: matplotlib_example_3.png")

# ============================================
# 4. Histograms and Statistical Plots
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Statistical Plots', fontsize=14, fontweight='bold')

# Histogram
data_hist = np.random.normal(100, 15, 1000)
axes[0, 0].hist(data_hist, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(np.mean(data_hist), color='red', linestyle='--', linewidth=2, label='Mean')
axes[0, 0].set_title('Histogram with Mean Line')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()

# Box plot
box_data = [np.random.normal(0, 1, 100), 
            np.random.normal(1, 1.5, 100),
            np.random.normal(-1, 0.8, 100)]
axes[0, 1].boxplot(box_data, labels=['Group A', 'Group B', 'Group C'])
axes[0, 1].set_title('Box Plot Comparison')
axes[0, 1].set_ylabel('Values')
axes[0, 1].grid(True, alpha=0.3)

# Probability distribution
x_dist = np.linspace(-4, 4, 100)
y_normal = stats.norm.pdf(x_dist, 0, 1)
y_student = stats.t.pdf(x_dist, df=5)
axes[1, 0].plot(x_dist, y_normal, 'b-', linewidth=2, label='Normal Distribution')
axes[1, 0].plot(x_dist, y_student, 'r-', linewidth=2, label='Student-t Distribution (df=5)')
axes[1, 0].fill_between(x_dist, y_normal, alpha=0.2, color='blue')
axes[1, 0].set_title('Probability Distributions')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('Probability Density')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Bar plot
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]
colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
bars = axes[1, 1].bar(categories, values, color=colors_bar, edgecolor='black', linewidth=1.5)
axes[1, 1].set_title('Bar Plot')
axes[1, 1].set_ylabel('Values')
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('matplotlib_example_4.png', dpi=150, bbox_inches='tight')
print("✓ Saved: matplotlib_example_4.png")

# ============================================
# 5. Vector Field / Quiver Plot
# ============================================

fig, ax = plt.subplots(figsize=(10, 8))

x = np.linspace(-3, 3, 15)
y = np.linspace(-3, 3, 15)
X, Y = np.meshgrid(x, y)

# Gradient field of f(x,y) = x^2 + y^2
U = 2*X
V = 2*Y

# Normalize for better visualization
magnitude = np.sqrt(U**2 + V**2)
U_normalized = U / magnitude
V_normalized = V / magnitude

quiver = ax.quiver(X, Y, U_normalized, V_normalized, magnitude, cmap='hot', scale=25)
ax.set_title('Vector Field: ∇f where f(x,y) = x² + y²', fontsize=12)
ax.set_xlabel('x')
ax.set_ylabel('y')
cbar = plt.colorbar(quiver, ax=ax)
cbar.set_label('Magnitude')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('matplotlib_example_5.png', dpi=150, bbox_inches='tight')
print("✓ Saved: matplotlib_example_5.png")

# ============================================
# 6. Contour Plot
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2)) * np.exp(-(X**2 + Y**2)/10)

# Filled contour plot
contourf = axes[0].contourf(X, Y, Z, levels=20, cmap='RdYlBu_r')
axes[0].set_title('Filled Contour Plot')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
cbar1 = plt.colorbar(contourf, ax=axes[0])
cbar1.set_label('Z value')

# Contour lines with labels
contour = axes[1].contour(X, Y, Z, levels=15, cmap='RdYlBu_r', linewidths=1.5)
axes[1].clabel(contour, inline=True, fontsize=8)
axes[1].set_title('Contour Lines with Labels')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')

plt.tight_layout()
plt.savefig('matplotlib_example_6.png', dpi=150, bbox_inches='tight')
print("✓ Saved: matplotlib_example_6.png")

print("\n" + "="*50)
print("All visualization examples created successfully!")
print("="*50)
print("\nGenerated PNG files:")
print("  - matplotlib_example_1.png (Line plots)")
print("  - matplotlib_example_2.png (Heatmaps)")
print("  - matplotlib_example_3.png (3D plots)")
print("  - matplotlib_example_4.png (Statistical plots)")
print("  - matplotlib_example_5.png (Vector fields)")
print("  - matplotlib_example_6.png (Contour plots)")
print("\nYou can view these images in VS Code or any image viewer.")
print("Matplotlib can also display plots interactively with plt.show()")
