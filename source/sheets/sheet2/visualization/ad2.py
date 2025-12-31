import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, BacksolveAdjoint, RecursiveCheckpointAdjoint
import equinox as eqx
import time

NEVAL = 50

# Use equinox.filter_jit for better JAX integration with Equinox modules
from equinox import filter_jit

# 1. Implement the vector field for the Brusselator system
class Brusselator(eqx.Module):
    A: float
    B: float

    def __call__(self, t, y, args):
        X, Y = y
        dX = self.A - (self.B + 1) * X + (X**2) * Y
        dY = self.B * X - (X**2) * Y
        return jnp.array([dX, dY])

# 2. Updated solver function to accept max_steps
def solve_brusselator(B, A, y0, t0, t1, adjoint, max_steps):
    """Solves the Brusselator ODEs, now with a configurable max_steps."""
    ts = jnp.linspace(t0, t1, NEVAL)
    saveat = SaveAt(ts=ts)
    vector_field = Brusselator(A=A, B=B)
    term = ODETerm(vector_field)
    solver = Dopri5()
    dt0 = (ts[1] - ts[0]) / NEVAL

    # Pass max_steps directly to the solver
    solution = diffeqsolve(term, solver, t0, t1, dt0, y0, saveat=saveat, adjoint=adjoint, max_steps=max_steps)
    return solution.ts, solution.ys

def main():
    # --- Simulation Parameters ---
    A = 1.0
    B = 3.0
    y0 = jnp.array([1.0, 2.0]) 
    t0 = 0.0
    t1 = 50.0
    
    # Increase max_steps significantly. The default is 2**16. We'll use 2**20.
    # We only really need this for the unstable backsolve, but we'll pass it to both for consistency.
    max_solver_steps = 2**20

    # --- 3. Set up Jacobian functions using equinox.filter_jit ---
    # BacksolveAdjoint Jacobian function
    jac_fn_backsolve = filter_jit(jax.jacrev(
        lambda b, a, y0, t0, t1: solve_brusselator(b, a, y0, t0, t1, BacksolveAdjoint(), max_solver_steps)[1], 
        argnums=0
    ))
    
    # RecursiveCheckpointAdjoint Jacobian function
    jac_fn_recursive = filter_jit(jax.jacrev(
        lambda b, a, y0, t0, t1: solve_brusselator(b, a, y0, t0, t1, RecursiveCheckpointAdjoint(), max_solver_steps)[1], 
        argnums=0
    ))

    # --- 4. Execute and Time the Jacobian Calculations ---
    print("--- Adjoint Method Comparison for the Brusselator ---")
    
    # Time BacksolveAdjoint (this will now be very slow but should complete)
    print("Warming up and JIT-compiling BacksolveAdjoint (this may take a while)...")
    _ = jac_fn_backsolve(B, A, y0, t0, t1).block_until_ready()
    print("Timing BacksolveAdjoint...")
    start_time = time.time()
    jacobian_backsolve = jac_fn_backsolve(B, A, y0, t0, t1).block_until_ready()
    backsolve_time = time.time() - start_time
    print(f"BacksolveAdjoint took: {backsolve_time:.4f} seconds")

    # Time RecursiveCheckpointAdjoint
    print("\nWarming up and JIT-compiling RecursiveCheckpointAdjoint...")
    _ = jac_fn_recursive(B, A, y0, t0, t1).block_until_ready()
    print("Timing RecursiveCheckpointAdjoint...")
    start_time = time.time()
    jacobian_recursive = jac_fn_recursive(B, A, y0, t0, t1).block_until_ready()
    recursive_time = time.time() - start_time
    print(f"RecursiveCheckpointAdjoint took: {recursive_time:.4f} seconds")
    
    difference = jnp.linalg.norm(jacobian_backsolve - jacobian_recursive)
    print(f"\nNorm of the difference between Jacobians: {difference:.6f}")

    # --- 5. Plot the Results ---
    ts, numerical_ys = solve_brusselator(B, A, y0, t0, t1, RecursiveCheckpointAdjoint(), max_solver_steps)

    fig = plt.figure(figsize=(14, 12))
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 1, 2)
    fig.suptitle(f'Brusselator Analysis: Differentiating w.r.t. rate $B={B}$', fontsize=16)

    ax1.plot(ts, numerical_ys[:, 0], label='Concentration X(t)')
    ax1.plot(ts, numerical_ys[:, 1], label='Concentration Y(t)')
    ax1.set_title('Solution Trajectory')
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Concentration')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(numerical_ys[:, 0], numerical_ys[:, 1], label='Limit Cycle')
    ax2.set_title('Phase Portrait')
    ax2.set_xlabel('Concentration X')
    ax2.set_ylabel('Concentration Y')
    ax2.grid(True)

    ax3.plot(ts, jacobian_backsolve[:, 0], label=r'$\frac{dX}{dB}$ (Backsolve)', linestyle='--')
    ax3.plot(ts, jacobian_recursive[:, 0], label=r'$\frac{dX}{dB}$ (Recursive)', linestyle=':')
    ax3.plot(ts, jacobian_backsolve[:, 1], label=r'$\frac{dY}{dB}$ (Backsolve)', linestyle='--')
    ax3.plot(ts, jacobian_recursive[:, 1], label=r'$\frac{dY}{dB}$ (Recursive)', linestyle=':')
    ax3.set_title(f'Comparison of Derivatives (Backsolve: {backsolve_time:.2f}s, Recursive: {recursive_time:.2f}s)')
    ax3.set_xlabel('Time (t)')
    ax3.set_ylabel('Derivative value')
    ax3.legend(ncol=2)
    ax3.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    main()