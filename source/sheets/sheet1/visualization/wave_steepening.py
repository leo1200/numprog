import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# Constants
x0 = 1.0  # meters
v0 = 1.0  # m/s
cs = 1.0  # m/s

# Time points to evaluate
times = [0.0, 0.5, 0.99]  # seconds
x_vals = np.linspace(-2, 2, 500)

# Implicit function for v(x, t)
def implicit_v(v, x, t):
    return v + v0 * np.sin(0.5 * np.pi / x0 * (x + (cs - v) * t))

# Solve v(x, t) numerically
def solve_v(x, t):
    sol = root_scalar(implicit_v, args=(x, t), method='brentq', bracket=[-2*v0, 2*v0])
    if sol.converged:
        return sol.root
    else:
        return np.nan

# Solve and plot
plt.figure(figsize=(8, 4))
for t in times:
    v_xt = [solve_v(x, t) for x in x_vals]
    plt.plot(x_vals, v_xt, label=f"t = {t:.2f} s", linewidth=4)

# Mark expected location of discontinuity
t_break = x0 / v0
x_break = -cs * t_break
plt.axvline(x_break, color='k', linestyle='--', label=f"expected discontinuity \nat x = {x_break:.2f} m")
plt.xlabel("x in m")
plt.ylabel("v in m/s")
plt.title("Solution v(x, t) to the Isothermal Euler Equations at Various Times \n" + r"$x_0 = 1 m, v_0 = 1 m/s, c_s = 1 m/s$")
plt.legend(loc ='upper right')
plt.ylim(-1.2, 1.2)
plt.tight_layout()
plt.savefig("wave_steepening.pdf", bbox_inches='tight')