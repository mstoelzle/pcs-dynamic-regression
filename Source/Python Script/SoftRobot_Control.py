import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

from diffrax import diffeqsolve, Euler, ODETerm, SaveAt, Tsit5
import dill
from functools import partial
import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import Array, lax, vmap, jit
import numpy as onp
from pathlib import Path
from typing import Dict, Tuple

num_segments = 1
model_dir = Path("./Source/Soft Robot/ns-1_high_shear_stiffness/model")
n_q_gt = 3 * num_segments  
n_q_hat = 2 * num_segments   # shear is deactivated

# load the ground truth model
with open(str(model_dir / 'true_model.dill'), 'rb') as f:
    true_model = dill.load(f)
# Load the trained model
with open(str(model_dir / 'learned_model.dill'), 'rb') as f:
    learned_model = dill.load(f)

# define the time step
dt = 1e-5
t0 = 0.0
t1 = 1e1

# Define the initial condition
q0 = jnp.zeros((n_q_gt, ))
q_d0 = jnp.zeros_like(q0)
x0 = jnp.concatenate([q0, q_d0])

# define the control target
q_des = jnp.array([-10.0, 0.5])
q_d_des = jnp.zeros_like(q_des)

# define the control gains
Kp = 1e-2 * jnp.diag(jnp.array([1.0, 1e-2]))
Ki = 1e-2 * jnp.diag(jnp.array([1.0, 1e-2]))
Kd = 1e-3 * jnp.diag(jnp.array([1.0, 1e-2]))


def apply_eps_to_bend_strains(q_bend: Array, eps: float = 1e-3):

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


def apply_eps_to_configuration(q: Array, eps: float = 1e-3):
    bend_strains_selector = onp.arange(0, n_q_hat, n_q_hat//num_segments)
    q_bend_epsed = apply_eps_to_bend_strains(q[bend_strains_selector], eps=eps)
    q_epsed = q.at[bend_strains_selector].set(q_bend_epsed)

    return q_epsed


def ode_gt_fn(t: Array, x: Array, tau: Array) -> Array:
    q, q_d = jnp.split(x, 2)
    q_epsed = apply_eps_to_configuration(q, eps=1e-1)

    # configuration-space forcing
    f_q = tau - true_model["eta_expr_lambda"](*q, *q_d, *q_epsed).T @ q_d + true_model["delta_expr_lambda"](*q, *q_d, *q_epsed)[:,0] - true_model["D"] @ q_d
    
    # compute the configuration-space acceleration
    q_dd = jnp.linalg.inv(true_model["zeta_expr_lambda"](*q, *q_epsed).T) @ f_q

    # construct the state derivative
    x_d = jnp.concatenate([q_d, q_dd])

    return x_d

def ode_hat_fn(t: Array, x: Array, tau: Array) -> Array:
    q, q_d = jnp.split(x, 2)
    q_epsed = apply_eps_to_configuration(q)

    # configuration-space forcing
    f_q = tau - learned_model["eta_expr_lambda"](*q, *q_d, *q_epsed).T @ q_d + learned_model["delta_expr_lambda"](*q, *q_d, *q_epsed)[:,0] - learned_model["D"] @ q_d
    
    # compute the configuration-space acceleration
    q_dd = jnp.linalg.inv(learned_model["zeta_expr_lambda"](*q, *q_epsed).T) @ f_q

    # construct the state derivative
    x_d = jnp.concatenate([q_d, q_dd])

    return x_d

def control_fn(
    t: Array, 
    y: Array, 
    q_des: Array,
    q_d_des: Array,
    Kp: Array,
    Ki: Array,
    Kd: Array,
    gamma: float = 1e0
) -> Tuple[Array, Array]:
    # extract the system state and integral error
    x = y[:-n_q_hat]
    q, q_d = jnp.split(x, 2)
    e_int = y[-n_q_hat:]

    # the observed configuration
    q_hat, q_d_hat = q[::2], q_d[::2]

    # compute the error
    e_q = q_des - q_hat

    # compute the feedforward term
    q_des_epsed = apply_eps_to_configuration(q_des, eps=5e0)
    K = -learned_model["delta_expr_lambda"](*q_des, *q_d_des, *q_des_epsed)[:,0]
    tau_ff = K

    # compute the feedback term
    tau_fb = Kp @ e_q + Ki @ e_int + Kd @ (q_d_des - q_d_hat)

    # compute the control input
    tau = tau_ff + tau_fb
    tau = jnp.concat([tau[0:1], jnp.zeros((1, )), tau[1:]])

    # infitesimal change in the integral error
    delta_e_int = jnp.tanh(gamma * e_q)

    return tau, delta_e_int

def closed_loop_control_ode_fn(
    t: Array, 
    y: Array, 
    args, 
    **control_kwargs: Dict[str, Array]
) -> Array:
    # compute the control input
    tau, delta_e_int = control_fn(t, y, **control_kwargs)

    # compute the state derivative of the ground truth model
    x_d = ode_gt_fn(t, y[:-n_q_hat], tau)

    y_d = jnp.concatenate([x_d, delta_e_int])

    return y_d


if __name__ == '__main__':
    # use diffrax to solve the ODE
    ts = jnp.arange(t0, t1, dt)

    # define the ODE term
    ode_term = ODETerm(jit(partial(
        closed_loop_control_ode_fn, 
        q_des=q_des,
        q_d_des=q_d_des,
        Kp=Kp,
        Ki=Ki,
        Kd=Kd
    )))

    # initialize integral error
    e_int0 = jnp.zeros((n_q_hat, ))

    # initialize the state
    y0 = jnp.concatenate([x0, e_int0])

    sol = diffeqsolve(
        ode_term, 
        Tsit5(), 
        t0, 
        t1,
        dt,
        y0, 
        # args=jnp.zeros_like(q0),
        saveat=SaveAt(ts=ts),
        max_steps=None
    )
    x_ts = sol.ys
    q_ts = x_ts[:, :n_q_gt]
    q_d_ts = x_ts[:, n_q_gt:]

    # plot the results
    fig, ax = plt.subplots(1, 1)
    for i in range(n_q_gt):
        ax.plot(ts, q_ts[:, i], label=r"$q_" + str(i+1) + "$")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("q")
    ax.grid(True)
    ax.legend()
    plt.show()