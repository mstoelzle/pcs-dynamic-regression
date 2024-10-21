import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

import dill
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, Euler, ODETerm, SaveAt, Tsit5
from jax import numpy as jnp
from jax import Array, lax, vmap, jit
import numpy as onp
from pathlib import Path

num_segments = 1
model_dir = Path("./Source/Soft Robot/ns-1_high_shear_stiffness/model")
n_q_gt = 2 * num_segments  
n_q_hat = 2 * num_segments   # shear is deactivated

# load the ground truth model
with open(str(model_dir / 'true_model.dill'), 'rb') as f:
    true_model = dill.load(f)
# Load the trained model
with open(str(model_dir / 'learned_model.dill'), 'rb') as f:
    learned_model = dill.load(f)

# define the time step
dt = 1e-6
t0 = 0.0
t1 = 2e-1

# Define the initial condition
q0 = jnp.zeros((n_q_gt, ))
# q0 = jnp.array([0.0, 0.0, 1.0])
q_d0 = jnp.zeros_like(q0)
x0 = jnp.concatenate([q0, q_d0])


def apply_eps_to_bend_strains_jnp(q_bend: Array, eps: float = 1e-3):

    q_bend_sign = jnp.sign(q_bend)
    q_bend_sign = jnp.where(q_bend_sign == 0, 1, q_bend_sign)

    q_epsed = lax.select(
        jnp.abs(q_bend) < eps,
        q_bend_sign*eps,
        q_bend
    )
    # old implementation
    # q_epsed = q_bend + (q_bend_sign * eps)
    return q_epsed


def ode_gt_fn(t: Array, x: Array, tau: Array) -> Array:
    q, q_d = jnp.split(x, 2)

    bend_strains_selector = onp.arange(0, n_q_gt, n_q_gt//num_segments)
    q_bend_epsed = apply_eps_to_bend_strains_jnp(q[bend_strains_selector])
    q_epsed = q.at[bend_strains_selector].set(q_bend_epsed)
    print("q_epsed: ", q_epsed)

    # configuration-space forcing
    f_q = tau - true_model["eta_expr_lambda"](*q, *q_d, *q_epsed).T @ q_d - true_model["delta_expr_lambda"](*q, *q_d, *q_epsed)[:,0] - true_model["D"] @ q_d
    print("f_q: ", f_q)
    
    # compute the configuration-space acceleration
    q_dd = jnp.linalg.inv(true_model["zeta_expr_lambda"](*q, *q_epsed).T) @ f_q
    print("q_dd: ", q_dd)

    # construct the state derivative
    x_d = jnp.concatenate([q_d, q_dd])
    print("x_d: ", x_d)

    return x_d

def ode_hat_fn(t: Array, x: Array, tau: Array) -> Array:
    q, q_d = jnp.split(x, 2)

    bend_strains_selector = onp.arange(0, n_q_gt, n_q_hat//num_segments)
    q_bend_epsed = apply_eps_to_bend_strains_jnp(q[bend_strains_selector], eps=5e0)
    q_epsed = q.at[bend_strains_selector].set(q_bend_epsed)
    print("q_epsed: ", q_epsed)

    # configuration-space forcing
    f_q = tau - learned_model["eta_expr_lambda"](*q, *q_d, *q_epsed).T @ q_d - learned_model["delta_expr_lambda"](*q, *q_d, *q_epsed)[:,0] - learned_model["D"] @ q_d
    print("f_q: ", f_q)
    
    # compute the configuration-space acceleration
    q_dd = jnp.linalg.inv(learned_model["zeta_expr_lambda"](*q, *q_epsed).T) @ f_q
    # q_dd = f_q
    print("q_dd: ", q_dd)

    # construct the state derivative
    x_d = jnp.concatenate([q_d, q_dd])
    print("x_d: ", x_d)

    return x_d


if __name__ == '__main__':
    # test the ode_fn
    print(ode_hat_fn(t0, x0, jnp.zeros_like(q0)))

    # define the ODE term
    ode_term = ODETerm(jit(ode_hat_fn))
    
    # use diffrax to solve the ODE
    ts = jnp.arange(t0, t1, dt)
    sol = diffeqsolve(
        ode_term, 
        Tsit5(), 
        t0, 
        t1,
        dt,
        x0, 
        args=jnp.zeros_like(q0),
        saveat=SaveAt(ts=ts),
        max_steps=None
    )
    x_ts = sol.ys

    # plot the results
    fig, ax = plt.subplots(1, 1)
    for i in range(n_q_gt):
        ax.plot(ts, x_ts[:, i], label=r"$q_" + str(i+1) + "$")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("q")
    ax.grid(True)
    ax.legend()
    plt.show()