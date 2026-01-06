import jax.numpy as jnp
import jax
from matplotlib import pyplot as plt

from functools import partial

import jax.numpy as jnp

import diffrax
from diffrax import diffeqsolve, ODETerm, Dopri5


def implicit_euler(initial_state, dt, num_steps, alpha):
    return jax.lax.fori_loop(
        0, num_steps,
        lambda _, y: y / (1 - alpha * dt),
        initial_state
    )

def implicit_euler_adjoint_decayA(initial_state, dt, num_steps, alpha):

    # we consider y' = alpha * y
    # 1. solve the forward problems
    # y_{n+1} = y_n / (1 - alpha * dt)

    final_state = jax.lax.fori_loop(
        0, num_steps, 
        lambda _, y: y / (1 - alpha * dt), 
        initial_state
    )

    # 2. solve the adjoint problem
    # a_y(t_fin) = 1, a_y' = -alpha * a_y
    # a_alpha(t_fin) = 0, a_alpha' = -a_y * y
    # dydalpha = a_alpha(0)

    a_y = 1.0
    a_alpha = 0.0
    y = final_state

    def adjoint_step(i, carry):
        a_y, a_alpha, y = carry
        # integrate backwards in time
        a_y = a_y / (1 - alpha * dt)
        a_alpha = a_alpha + a_y * y * dt
        y = y * (1 - alpha * dt)
        return a_y, a_alpha, y
    
    a_y, a_alpha, _ = jax.lax.fori_loop(
        0, num_steps, 
        adjoint_step, 
        (a_y, a_alpha, y)
    )

    return final_state, a_alpha


def implicit_euler_adjoint_decayB(initial_state, dt, num_steps, alpha):

    # we consider y' = alpha * y
    # 1. solve the forward problems
    # y_{n+1} = y_n / (1 - alpha * dt)

    final_state = jax.lax.fori_loop(
        0, num_steps, 
        lambda _, y: y / (1 - alpha * dt), 
        initial_state
    )

    # 2. solve the adjoint problem
    # a_y(t_fin) = 1, a_y' = -alpha * a_y
    # a_alpha(t_fin) = 0, a_alpha' = -a_y * y
    # dydalpha = a_alpha(0)

    a_y = 1.0
    a_alpha = 0.0
    y = final_state

    def adjoint_step(i, carry):
        a_y, a_alpha, y = carry
        # integrate backwards in time
        y = y / (1 + alpha * dt)
        a_y = a_y / (1 - alpha * dt)
        a_alpha = a_alpha + a_y * y * dt
        return a_y, a_alpha, y
    
    a_y, a_alpha, _ = jax.lax.fori_loop(
        0, num_steps, 
        adjoint_step, 
        (a_y, a_alpha, y)
    )

    return final_state, a_alpha

# example

# analytical solution: y(t) = y_0 * exp(alpha * t),
# derivative w.r.t. alpha: dy/dalpha = y_0 * t * exp(alpha * t)
if __name__ == "__main__":
    initial_state = 1.0
    dt = 0.3
    num_steps = jnp.arange(60)
    t_fin = dt * num_steps
    alpha = -0.5

    # vmap over num_steps
    final_statesA, a_alphasA = jax.vmap(
        implicit_euler_adjoint_decayA,
        in_axes=(None, None, 0, None)
    )(initial_state, dt, num_steps, alpha)

    final_statesB, a_alphasB = jax.vmap(
        implicit_euler_adjoint_decayB,
        in_axes=(None, None, 0, None)
    )(initial_state, dt, num_steps, alpha)


    true_final_state = initial_state * jnp.exp(alpha * dt * num_steps)
    true_a_alphas = initial_state * dt * num_steps * jnp.exp(alpha * dt * num_steps)

    # also calculate the automatic differentiation version
    a_alphas_ad = jax.vmap(
        lambda n: jax.jacfwd(implicit_euler, argnums=3)(
            initial_state, dt, n, alpha
        )
    )(num_steps)

    # also do it in diffrax for reference
    def f(t, y, args):
        return args * y

    term = ODETerm(f)
    solver = diffrax.Dopri5()
    y0 = jnp.array([initial_state])
    sol_func_AD = lambda alpha, t_fin: diffeqsolve(
        term, solver, t0 = 0, t1 = t_fin, dt0 = dt, y0 = y0, args = alpha, 
        adjoint = diffrax.RecursiveCheckpointAdjoint(), 
        stepsize_controller = diffrax.ConstantStepSize()
    ).ys[0, 0]

    sol_func_AJ = lambda alpha, t_fin: diffeqsolve(
        term, solver, t0=0, t1=t_fin, dt0=dt, y0=y0, args=alpha,
        adjoint=diffrax.BacksolveAdjoint(),
        stepsize_controller=diffrax.ConstantStepSize()
    ).ys[0, 0]

    # use jax.value_and_grad to compute the final states and derivatives
    final_states_dopri5_AD, derivatives_dopri5_AD = jax.vmap(
        lambda n: jax.value_and_grad(sol_func_AD, argnums=0)(alpha, dt * n)
    )(num_steps)

    final_states_dopri5_AJ, derivatives_dopri5_AJ = jax.vmap(
        lambda n: jax.value_and_grad(sol_func_AJ, argnums=0)(alpha, dt * n)
    )(num_steps)

    # plot the results
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # State plot
    ax[0].plot(num_steps * dt, true_final_state, label='ground truth', linewidth=5, color='black', linestyle='-')
    ax[0].plot(num_steps * dt, final_statesA, label='implicit Euler', color='blue', linewidth=2, linestyle='-')
    ax[0].plot(num_steps * dt, final_states_dopri5_AD, label='dopri5', color='orange', linewidth=2, linestyle='-')
    
    ax[0].set_xlabel('time')
    ax[0].set_ylabel('state y')
    ax[0].legend()

    # Derivative plot
    ax[1].plot(num_steps * dt, true_a_alphas, label='ground truth', linewidth=5, color='black', linestyle='-')
    ax[1].plot(num_steps * dt, a_alphas_ad, label='direct autodiff with implicit Euler', linewidth=5, color='darkblue', linestyle='-')
    ax[1].plot(num_steps * dt, a_alphasA, label='discrete adjoint of implicit Euler', color='lavender', linewidth=2, linestyle='-')
    ax[1].plot(num_steps * dt, a_alphasB, label='continous adjoint with implicit Euler\nfor forward- and backsolve', color='green', linewidth=2, linestyle='-')
    ax[1].plot(num_steps * dt, derivatives_dopri5_AJ, label='continous adjoint with dopri5\nfor forward- and backsolve', color='orange', linewidth=2, linestyle='-')
    ax[1].plot(num_steps * dt, derivatives_dopri5_AD, label='direct autodiff with dopri5', color='darkred', linewidth=1, linestyle='--')

    ax[1].set_xlabel('time')
    ax[1].set_ylabel(r"derivative with respect to $\alpha$")
    ax[1].legend()

    plt.suptitle(r'differentiation of the solution of $\frac{dy}{dt} = \alpha y$ with respect to $\alpha$, $\Delta t = 0.3$, $\alpha = -0.5$', fontsize=16)
    plt.subplots_adjust(top=0.85)

    plt.tight_layout()
    plt.savefig("differentiation_decay.pdf", bbox_inches='tight')