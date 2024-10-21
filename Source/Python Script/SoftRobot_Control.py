import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

from diffrax import diffeqsolve, Euler, ODETerm, SaveAt, Tsit5
import dill
from functools import partial
import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import Array, jit, lax, random, vmap
import numpy as onp
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple


# plotting settings
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Romand"],
    }
)

figsize = (5.0, 3.0)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


num_segments = 2
model_dir = Path(f"./Source/Soft Robot/ns-{num_segments}_dof-3/model")
n_q_gt = 3 * num_segments  
n_q_hat = 3 * num_segments   # shear is deactivated

# load the ground truth model
with open(str(model_dir / 'true_model.dill'), 'rb') as f:
    true_model = dill.load(f)
# Load the trained model
with open(str(model_dir / 'learned_model.dill'), 'rb') as f:
    learned_model = dill.load(f)

# Define the initial condition
q0 = jnp.zeros((n_q_gt, ))
q_d0 = jnp.zeros_like(q0)
x0 = jnp.concatenate([q0, q_d0])

# define the control target
# q_des = jnp.array([-10.0, 0.0])
# q_des = jnp.zeros((n_q_hat, ))
# q_d_des = jnp.zeros_like(q_des)
num_setpoints = 7
sim_duration_per_setpoint = 5.0  # s

# define the time step
dt = 1e-2
sim_dt = 1e-4
t0 = 0.0
t1 = num_setpoints * sim_duration_per_setpoint

# define the control gains
K_diag = jnp.diag(jnp.array([1e-1, 5e-1, 1e1, 1e-1, 5e-1, 1e1]))
Kp = 1e-1 * K_diag
Ki = 2e0 * K_diag
Kd = 1e-2 * K_diag
gamma = 1e0 * jnp.diag(jnp.array([40.0, 0.1, 0.2, 40.0, 0.1, 0.2]))


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
) -> Tuple[Array, Dict[str, Array]]:
    # extract the system state and integral error
    x = y[:-n_q_hat]
    q, q_d = jnp.split(x, 2)
    e_int = y[-n_q_hat:]

    # the observed configuration
    q_hat, q_d_hat = q, q_d

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

    # infitesimal change in the integral error
    delta_e_int = jnp.tanh(gamma @ e_q)

    aux = dict(
        tau_ff=tau_ff,
        tau_fb=tau_fb,
        delta_e_int=delta_e_int
    )

    return tau, aux

def closed_loop_control_ode_fn(
    t: Array, 
    y: Array, 
    control_args, 
    **control_kwargs: Dict[str, Array]
) -> Array:
    # compute the control input
    tau, control_aux = control_fn(t, y, *control_args, **control_kwargs)

    # compute the state derivative of the ground truth model
    x_d = ode_gt_fn(t, y[:-n_q_hat], tau)

    y_d = jnp.concatenate([x_d, control_aux["delta_e_int"]])

    return y_d


if __name__ == '__main__':
    # use diffrax to solve the ODE
    ts = jnp.arange(t0, t1 + dt, dt)

    # define the setpointn sequence
    rng_setpoint = random.PRNGKey(seed=1)
    q_des_ps = random.uniform(
        rng_setpoint, shape=(num_setpoints, n_q_hat), minval=-1.0, maxval=1.0
    )
    # rescale the setpoints
    q_des_ps = q_des_ps * jnp.array([
        40.0, 0.1, 0.2, 10.0, 0.1, 0.2
    ])
    q_d_des_ps = jnp.zeros_like(q_des_ps)

    # define the control kwargs
    control_kwargs = dict(
        q_d_des=jnp.zeros((n_q_hat, )),
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        gamma=gamma
    )

    # define the ODE term
    ode_term = ODETerm(jit(partial(
        closed_loop_control_ode_fn, 
        **control_kwargs
    )))

    # define the solver
    ode_solver = Tsit5()


    save_ts = jnp.arange(dt, sim_duration_per_setpoint + dt, dt)

    @jit
    def rollout_closed_loop_system(t0: Array, q_des: Array) -> Array:
        # solve the ODE
        sol = diffeqsolve(
            ode_term, 
            ode_solver, 
            t0, 
            t0 + sim_duration_per_setpoint + dt,
            sim_dt,
            y0_i,
            args=(q_des, ),
            # args=jnp.zeros_like(q0),
            saveat=SaveAt(ts=t0 + save_ts),
            max_steps=None
        )

        return sol.ys

    # initialize integral error
    e_int0 = jnp.zeros((n_q_hat, ))

    # initialize the state
    y0 = jnp.concatenate([x0, e_int0])

    y_ts = [y0[None, :]]
    q_des_ts = [q_des_ps[0:1]]
    q_d_des_ts = [q_d_des_ps[0:1]]

    t0_i = t0
    t1_i = t0_i + sim_duration_per_setpoint
    y0_i = y0
    for setpoint_idx in tqdm(range(num_setpoints)):
        q_des = q_des_ps[setpoint_idx]
        q_d_des = q_d_des_ps[setpoint_idx]

        # define the time sequence
        ts_i = jnp.arange(t0_i, t1_i, dt)
        print("t0_i: ", t0_i, "t1_i: ", t1_i, "ts_i: ", t0_i + save_ts)

        y_ts_i = rollout_closed_loop_system(t0_i, q_des)
        print("before trim: y_ts_i: ", y_ts_i.shape)
        y_ts.append(y_ts_i)
        q_des_ts.append(jnp.tile(q_des[None, :], (y_ts_i.shape[0], 1)))
        print("appended", jnp.tile(q_des[None, :], (y_ts_i.shape[0], 1)).shape)
        q_d_des_ts.append(jnp.tile(q_d_des[None, :], (y_ts_i.shape[0], 1)))

        # update the time and the initial condition
        t0_i = t1_i
        t1_i = t1_i + sim_duration_per_setpoint
        y0_i = y_ts_i[-1]

    y_ts = jnp.concatenate(y_ts, axis=0)
    q_des_ts = jnp.concatenate(q_des_ts, axis=0)
    q_d_des_ts = jnp.concatenate(q_d_des_ts, axis=0)
    print("y_ts: ", y_ts.shape, "q_des_ts: ", q_des_ts.shape, "q_d_des_ts: ", q_d_des_ts.shape)

    q_ts = y_ts[:, :n_q_gt]
    q_d_ts = y_ts[:, n_q_gt:2*n_q_gt]
    e_int_ts = y_ts[:, -n_q_hat:]

    # define the control function to reconstruct the control input
    tau_ts, control_aux_ts = vmap(partial(control_fn, **control_kwargs))(ts, y_ts, q_des_ts)

    # save the results
    sim_ts = dict(
        ts=ts,
        q_ts=q_ts,
        q_d_ts=q_d_ts,
        q_des_ts=q_des_ts,
        q_d_des_ts=q_d_des_ts,
        e_int_ts=e_int_ts,
        tau_ts=tau_ts,
        tau_ff_ts=control_aux_ts["tau_ff"],
        tau_fb_ts=control_aux_ts["tau_fb"],
    )
    control_dir = model_dir.parent / "control"
    control_dir.mkdir(parents=True, exist_ok=True)
    jnp.savez(str(control_dir / "closed_loop_control_data.npz"), **sim_ts)

    # plot the results
    fig, ax = plt.subplots(1, 1)
    for setpoint_idx in range(n_q_gt):
        ax.plot(ts, q_ts[:, setpoint_idx], label=r"$q_" + str(setpoint_idx+1) + "$")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("q")
    ax.grid(True)
    ax.legend()
    plt.show()