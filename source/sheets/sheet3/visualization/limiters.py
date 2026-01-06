import numpy as np
import matplotlib.pyplot as plt

def _minmod(a, b):
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))

def _minmod3(a, b, c):
    """Minmod function for three arguments in NumPy."""
    same_sign = (np.sign(a) == np.sign(b)) & (np.sign(b) == np.sign(c))
    return np.where(same_sign, np.sign(a) * np.minimum(np.abs(a), np.minimum(np.abs(b), np.abs(c))), 0.0)

def _double_minmod(a, b):
    """
    Double minmod limiter in NumPy.
    """
    return np.where(
        a * b > 0, 
        _minmod3((a + b) / 2, 2 * a, 2 * b),
        0.0
    )

# Set up a few cells with a discontinuity
cell_centers = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
cell_values = np.array([0.9, 1.0, 2.0, 2.1, 2.2])  # discontinuity between cells 1 and 2

dx = 1.0
n_cells = len(cell_centers)

# Compute left and right differences
left_diffs = cell_values[1:-1] - cell_values[:-2]
right_diffs = cell_values[2:] - cell_values[1:-1]

# Central average slope, minmod slope, and double minmod slope
avg_slopes = 0.5 * (left_diffs + right_diffs)
minmod_slopes = _minmod(left_diffs, right_diffs)
double_minmod_slopes = _double_minmod(left_diffs, right_diffs)

# Extend slopes to all cells (set boundary slopes to 0)
avg_slopes_full = np.zeros_like(cell_values)
avg_slopes_full[1:-1] = avg_slopes

minmod_slopes_full = np.zeros_like(cell_values)
minmod_slopes_full[1:-1] = minmod_slopes

double_minmod_slopes_full = np.zeros_like(cell_values)
double_minmod_slopes_full[1:-1] = double_minmod_slopes

# Reconstruction for each method
def reconstruct(cell_centers, cell_values, slopes, dx):
    segments_x = []
    segments_y = []
    for xc, val, slope in zip(cell_centers, cell_values, slopes):
        x_left = xc - 0.5 * dx
        x_right = xc + 0.5 * dx
        y_left = val - 0.5 * dx * slope
        y_right = val + 0.5 * dx * slope
        segments_x.append([x_left, x_right])
        segments_y.append([y_left, y_right])
    return segments_x, segments_y

avg_x, avg_y = reconstruct(cell_centers, cell_values, avg_slopes_full, dx)
minmod_x, minmod_y = reconstruct(cell_centers, cell_values, minmod_slopes_full, dx)
double_minmod_x, double_minmod_y = reconstruct(cell_centers, cell_values, double_minmod_slopes_full, dx)

# Plotting
fig, ax = plt.subplots(figsize=(8, 4))

# Cell boundaries
for x in np.arange(0, 6):
    ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.5)

# Cell values
ax.plot(cell_centers, cell_values, 'ko', label='Cell average')

# Reconstructions
for xseg, yseg in zip(avg_x, avg_y):
    ax.plot(xseg, yseg, 'b-', label='average slope reconstruction' if xseg == avg_x[0] else "")

for xseg, yseg in zip(minmod_x, minmod_y):
    ax.plot(xseg, yseg, 'r--', label='minmod slope reconstruction' if xseg == minmod_x[0] else "")

for xseg, yseg in zip(double_minmod_x, double_minmod_y):
    ax.plot(xseg, yseg, 'g-.', label='double minmod reconstruction' if xseg == double_minmod_x[0] else "")

ax.set_xlabel("x")
ax.set_ylabel("value")
ax.set_title("Linear Reconstruction Comparison")
ax.legend()
plt.tight_layout()
plt.savefig("linear_reconstruction_comparison.svg")
