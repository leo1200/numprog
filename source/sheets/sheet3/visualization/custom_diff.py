import jax
import jax.numpy as jnp
from jax import custom_vjp, custom_jvp, jacobian, jvp, vjp
from jax.lax import fori_loop
from functools import partial
import matplotlib.pyplot as plt

# use double precision for better accuracy
jax.config.update("jax_enable_x64", True)


# === Slower-converging fixed-point map (α near 1) ===
def f_slow(x, theta, alpha=0.9):
    nonlinear = jnp.array([
        jnp.exp(theta[0]**2 + 2 * theta[1]) / x[0],
        jnp.exp(2 * theta[0] + theta[1]**2) / x[1]
    ])
    return alpha * x + (1 - alpha) * nonlinear


# === Analytic solution ===
def x_star_analytic(theta):
    return jnp.array([
        jnp.exp(0.5 * theta[0]**2 + theta[1]),
        jnp.exp(theta[0] + 0.5 * theta[1]**2)
    ])


# === Fixed-point iteration (no custom AD) ===
def fixed_point_direct(f, x_guess, theta, num_iters=100):
    def body(_, x): return f(x, theta)
    return fori_loop(0, num_iters, body, x_guess)


# === Fixed-point with custom VJP ===
@partial(custom_vjp, nondiff_argnums=(0, 3))
def fixed_point_vjp(f, x_guess, theta, num_iters=100):
    def body(_, x): return f(x, theta)
    return fori_loop(0, num_iters, body, x_guess)

def fixed_point_vjp_fwd(f, x_guess, theta, num_iters):
    x_star = fixed_point_vjp(f, x_guess, theta, num_iters)
    # primal output, stored variables for backward pass
    return x_star, (x_star, theta)

def fixed_point_vjp_bwd(f, num_iters, res, v):
    x_star, theta = res
    A = jacobian(f, argnums=0)(x_star, theta)
    B = jacobian(f, argnums=1)(x_star, theta)
    u = jnp.linalg.solve((jnp.eye(x_star.shape[0]) - A).T, v)
    return jnp.zeros_like(x_star), u @ B

fixed_point_vjp.defvjp(fixed_point_vjp_fwd, fixed_point_vjp_bwd)


# === Fixed-point with custom JVP ===
@partial(custom_jvp, nondiff_argnums=(0, 3))
def fixed_point_jvp(f, x_guess, theta, num_iters=100):
    def body(_, x): return f(x, theta)
    return fori_loop(0, num_iters, body, x_guess)

@fixed_point_jvp.defjvp
def fixed_point_jvp_rule(f, num_iters, primals, tangents):
    x_guess, theta = primals
    dx_guess, dtheta = tangents
    x_star = fixed_point_jvp(f, x_guess, theta, num_iters)
    A = jacobian(f, argnums=0)(x_star, theta)
    B = jacobian(f, argnums=1)(x_star, theta)
    dx = jnp.linalg.solve(jnp.eye(x_star.shape[0]) - A, B @ dtheta)
    # primal output, tangent output
    return x_star, dx


# === Test and Benchmark ===
def run_error_test(theta_val=jnp.array([1.0, 1.0]), v=jnp.array([1.0, 1.0]), iters_range=range(1, 31), alpha=0.9):
    x_guess = jnp.ones_like(theta_val)
    jac_true = jacobian(x_star_analytic)(theta_val)
    jvp_true = jac_true @ v
    vjp_true = v @ jac_true
    x_true = x_star_analytic(theta_val)

    errors = {
        "jvp_direct": [],
        "jvp_custom": [],
        "vjp_direct": [],
        "vjp_custom": [],
        "fixed_point_error": []
    }

    f = lambda x, th: f_slow(x, th, alpha)

    for k in iters_range:
        # Fixed-point estimate
        x_k = fixed_point_direct(f, x_guess, theta_val, k)
        errors["fixed_point_error"].append(jnp.max(jnp.abs(x_k - x_true)))

        # --- Direct autodiff ---
        _, jvp_est_direct = jvp(lambda th: fixed_point_direct(f, x_guess, th, k), (theta_val,), (v,))
        _, vjp_pull_direct = vjp(lambda th: fixed_point_direct(f, x_guess, th, k), theta_val)
        vjp_est_direct = vjp_pull_direct(v)[0]

        # --- Custom JVP ---
        _, jvp_est_custom = jvp(lambda th: fixed_point_jvp(f, x_guess, th, k), (theta_val,), (v,))

        # --- Custom VJP ---
        _, vjp_pull_custom = vjp(lambda th: fixed_point_vjp(f, x_guess, th, k), theta_val)
        vjp_est_custom = vjp_pull_custom(v)[0]

        # --- Errors ---
        errors["jvp_direct"].append(jnp.max(jnp.abs(jvp_est_direct - jvp_true)))
        errors["jvp_custom"].append(jnp.max(jnp.abs(jvp_est_custom - jvp_true)))
        errors["vjp_direct"].append(jnp.max(jnp.abs(vjp_est_direct - vjp_true)))
        errors["vjp_custom"].append(jnp.max(jnp.abs(vjp_est_custom - vjp_true)))

    return errors


# === Run + Plot ===
if __name__ == "__main__":
    theta_val = jnp.array([1.0, 1.0])
    v = jnp.array([1.0, 1.0])
    iters = list(range(1, 31))
    alpha = 0.8

    errors = run_error_test(theta_val, v, iters, alpha=alpha)

    plt.figure(figsize=(8, 4))
    plt.plot(iters, errors["fixed_point_error"], label="Fixed-point error", linestyle="-", marker='s')
    plt.plot(iters, errors["jvp_direct"], label="JVP (direct)", linestyle="--", marker='o')
    plt.plot(iters, errors["jvp_custom"], label="JVP (implicit)", linestyle="-", marker='o')
    plt.plot(iters, errors["vjp_direct"], label="VJP (direct)", linestyle="--", marker='x')
    plt.plot(iters, errors["vjp_custom"], label="VJP (implicit)", linestyle="-", marker='x')

    plt.yscale("log")
    plt.xlabel("Fixed-point iterations")
    plt.ylabel("Max error vs. ground truth")
    plt.title(f"Convergence of Fixed-point and Gradient Accuracy (α={alpha})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("custom_diff_errors.pdf", bbox_inches='tight')
