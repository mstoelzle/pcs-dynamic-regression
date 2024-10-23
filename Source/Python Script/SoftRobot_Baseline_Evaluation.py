import jax
import numpy as onp
import os

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
os.environ["KERAS_BACKEND"] = "jax"

import diffrax as dfx
from jax import Array, debug, jit, lax, vmap
import jax.numpy as jnp
# Note that Keras should only be imported after the backend
# has been configured. The backend cannot be changed once the
# package is imported.
import keras
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter
from typing import Callable, Dict, Optional


# Set the seed using keras.utils.set_random_seed. This will set:
# 1) `numpy` seed
# 2) backend random seed
# 3) `python` random seed
keras.utils.set_random_seed(0)

# define directories
case_dir = Path(f"./Source/Soft Robot/ns-2_dof-3")
dataset_dir = case_dir / "training" / "cv"
model_dir = case_dir / "model"
print("Model dir", model_dir.resolve())

# define the simulation parameters
sim_dt = 5e-6

# plotting settings
plot_position_skip = 6
plot_marker_skip = 5

# load the trained model
learned_model = keras.models.load_model(str(model_dir / 'learned_node_model.keras'), safe_mode=False)
trainable_variables = learned_model.trainable_variables
non_trainable_variables = learned_model.non_trainable_variables
model_forward_fn = lambda x: learned_model.stateless_call(trainable_variables, non_trainable_variables, x)[0]


@jit
def ode_fn(t: Array, y: Array, tau: Array) -> Array:
    chi, chi_d = jnp.split(y, 2)

    # call the model
    model_input = jnp.concatenate([chi, chi_d, tau], axis=-1)[None, :]
    model_output = model_forward_fn(model_input)

    chi_dd = model_output.squeeze(axis=0)
    # debug.print("chi_dd min={min}, max={max}", min=chi_dd.min(), max=chi_dd.max())

    y_d = jnp.concatenate([chi_d, chi_dd], axis=-1)
    return y_d


def simulate_dynamics(
    x0: Array,
    x_d0: Array,
    tau_ts: Array,
    sim_dt: float,
    control_dt: float,
    control_ts: Array,
    ode_solver_class=dfx.Tsit5,
) -> Dict[str, Array]:
    n_x = x0.shape[0]

    ode_solver = ode_solver_class()
    
    assert control_ts.shape[0] == tau_ts.shape[0], "The control time steps must match the control inputs."
    y0 = jnp.concatenate((x0, x_d0), axis=-1)

    @jit
    def scan_fn(carry, input):
        t, tau = input["t"], input["tau"]
        y = carry["y_next"]
        x, x_d = y[:n_x], y[n_x:]

        ode_term = dfx.ODETerm(ode_fn)

        sol = dfx.diffeqsolve(
            ode_term,
            solver=ode_solver,
            t0=t,
            t1=t + control_dt,
            dt0=sim_dt,
            y0=y,
            args=tau,
            max_steps=None,
        )

        y_next = sol.ys[-1]

        carry["y_next"] = y_next

        output = {
            "ts": t,
            "x_ts": x,
            "x_d_ts": x_d,
            "y_ts": y,
        }

        return carry, output

    init_carry = dict(y_next=y0)
    input_ts = dict(t=control_ts, tau=tau_ts)
    last_carry, sim_ts = lax.scan(scan_fn, init_carry, input_ts)

    return sim_ts



if __name__ == "__main__":
    # load the dataset
    Y = jnp.load(dataset_dir / 'Y.npy')
    Y_d = jnp.load(dataset_dir / 'Ydot.npy')
    Tau = jnp.load(dataset_dir / "Tau.npy")
    tau_ts = Tau.reshape(Y.shape[0], Tau.shape[-1])
    # tau_ts = jnp.zeros_like(tau_ts)

    # only simulate for the first video
    dataset_selector = jnp.arange(Y.shape[0] // 8)
    Y = Y[dataset_selector]
    Y_d = Y_d[dataset_selector]
    tau_ts = tau_ts[dataset_selector]

    # set the time steps
    dt = 1e-3
    ts = dt * dataset_selector

    # split the dataset
    n_chi = Y.shape[-1] // 2
    chi_ts, chi_d_ts = Y[:, :n_chi], Y[:, n_chi:]
    chi_dd_ts = Y_d[:, n_chi:]

    # # marker sub-sampling
    # num_samples = chi_ts.shape[0]
    # num_markers = chi_ts.shape[1] // 3
    # print("Number of markers:", num_markers)
    # marker_indices = onp.array([num_markers // 2, num_markers - 1])
    # # marker_indices = np.array([num_markers - 1])
    # print("Marker indices:", marker_indices)
    # # reshape tensors
    # chi_ts = chi_ts.reshape(num_samples, num_markers, 3)
    # chi_d_ts = chi_d_ts.reshape(num_samples, num_markers, 3)
    # chi_dd_ts = chi_dd_ts.reshape(num_samples, num_markers, 3)
    # # sub-sample the data
    # chi_ts = chi_ts[:, marker_indices, :].reshape(num_samples, -1)
    # chi_d_ts = chi_d_ts[:, marker_indices, :].reshape(num_samples, -1)
    # chi_dd_ts = chi_dd_ts[:, marker_indices, :].reshape(num_samples, -1)
    # # update the number of markers
    # num_markers = marker_indices.shape[0]
    # n_chi = chi_ts.shape[-1]

    # evaluate accelerations on the training data
    model_input = jnp.concatenate([chi_ts, chi_d_ts, tau_ts], axis=-1)
    Chi_dd_hat = model_forward_fn(model_input)
    # plot the ground-truth and predicted accelerations
    fig, ax = plt.subplots(1, 1, dpi=200, num="acceleration")
    for i in range(0, chi_dd_ts.shape[-1], plot_position_skip):
        ax.plot(ts, chi_dd_ts[:, i], linestyle=":", linewidth=2.5, label=r"$\ddot{\chi}_{" + str(i + 1) + "}$")
    ax.set_prop_cycle(None)
    for i in range(0, Chi_dd_hat.shape[-1], plot_position_skip):
        ax.plot(ts, Chi_dd_hat[:, i], linewidth=2.0, label=r"$\hat{\ddot{\chi}}_{" + str(i + 1) + "}$")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Acceleration [m/s$^2$]")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # define the initial condition
    y0 = jnp.concat([chi_ts[0], chi_d_ts[0]], axis=-1)
    tau0 = jnp.array(tau_ts[0])

    # test the ode function
    y_d0 = ode_fn(0.0, y0, tau0)
    print("y_d0:\n", y_d0)

    # simulate the learned dynamics on the training data
    sim_ts = simulate_dynamics(
        x0=chi_ts[0],
        x_d0=chi_d_ts[0],
        tau_ts=tau_ts,
        sim_dt=sim_dt,
        control_dt=dt,
        control_ts=ts,
    )

    # extract the data
    px_gt_ts = chi_ts[:, 0::3]
    py_gt_ts = chi_ts[:, 1::3]
    theta_gt_ts = chi_ts[:, 2::3]
    px_hat_ts = sim_ts["x_ts"][:, 0::3]
    py_hat_ts = sim_ts["x_ts"][:, 1::3]
    theta_hat_ts = sim_ts["x_ts"][:, 2::3]
    print("px_hat_ts\n", px_hat_ts)

    # plot the x position
    fig, ax = plt.subplots(1, 1, dpi=200, num="x-position")
    # plot the reference trajectory
    for i in range(0, px_gt_ts.shape[-1], plot_marker_skip):
        ax.plot(ts, px_gt_ts[:, i], linestyle=":", linewidth=2.5, label=r"$p_{\mathrm{x}," + str(i + 1) + "}$")
    # reset the color cycle
    ax.set_prop_cycle(None)
    # plot the predicted trajectory
    for i in range(0, px_hat_ts.shape[-1], plot_marker_skip):
        ax.plot(ts, px_hat_ts[:, i], linewidth=2.0, label=r"$\hat{p}_{\mathrm{x}," + str(i + 1) + "}$")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [m]")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # plot the y position
    fig, ax = plt.subplots(1, 1, dpi=200, num="y-position")
    # plot the reference trajectory
    for i in range(0, py_gt_ts.shape[-1], plot_marker_skip):
        ax.plot(ts, py_gt_ts[:, i], linestyle=":", linewidth=2.5, label=r"$p_{\mathrm{y}," + str(i + 1) + "}$")
    # reset the color cycle
    ax.set_prop_cycle(None)
    # plot the predicted trajectory
    for i in range(0, py_hat_ts.shape[-1], plot_marker_skip):
        ax.plot(ts, py_hat_ts[:, i], linewidth=2.0, label=r"$\hat{p}_{\mathrm{y}," + str(i + 1) + "}$")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [m]")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # plot the orientation
    fig, ax = plt.subplots(1, 1, dpi=200, num="orientation")
    # plot the reference trajectory
    for i in range(0, theta_gt_ts.shape[-1], plot_marker_skip):
        ax.plot(ts, theta_gt_ts[:, i], linestyle=":", linewidth=2.5, label=r"$\theta_{" + str(i + 1) + "}$")
    # reset the color cycle
    ax.set_prop_cycle(None)
    # plot the predicted trajectory
    for i in range(0, theta_hat_ts.shape[-1], plot_marker_skip):
        ax.plot(ts, theta_hat_ts[:, i], linewidth=2.0, label=r"$\hat{\theta}_{" + str(i + 1) + "}$")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Orientation [rad]")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()
