import jax
import numpy as onp
import os

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

# parameters
num_segments = 2
case_dir = Path("Source") / "Soft Robot" / f"ns-2_dof-3"
evaluation_dir = case_dir / "evaluation"
dataset_type = "train"
assert dataset_type in ["train", "test"]
marker_indices = jnp.array([10, 15, 20])
model_types = ["node"]
assert all(model_type in ["pcs_regression", "node", "lstm", "con"] for model_type in model_types)

# time step
dt = 1e-3

# plotting settings
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Romand"],
    }
)

figsize = (5.0, 3.0)
dpi = 200
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plot_marker_skip = 1

if __name__ == "__main__":
    # load the ground-truth data
    match dataset_type:
        case "train":
            dataset_dir = case_dir / "training" / "cv"

            # load the dataset
            Y = jnp.load(dataset_dir / 'Y.npy')
            Y_d = jnp.load(dataset_dir / 'Ydot.npy')
            Tau = jnp.load(dataset_dir / "Tau.npy")
            tau_ts = Tau.reshape(Y.shape[0], Tau.shape[-1])

            # only consider the first video
            dataset_selector = jnp.arange(Y.shape[0] // 8)
            Y = Y[dataset_selector]
            Y_d = Y_d[dataset_selector]
            tau_ts = tau_ts[dataset_selector]

            # split the dataset
            n_chi = Y.shape[-1] // 2
            chi_gt_ts, chi_d_gt_ts = Y[:, :n_chi], Y[:, n_chi:]
            chi_dd_gt_ts = Y_d[:, n_chi:]

            # set the time steps
            ts = dt * dataset_selector
        case "val":
            dataset_dir = case_dir / "validation" / "sinusoidal_actuation"

            # load the dataset
            Chi = jnp.load(dataset_dir / 'Chi_val.npy')
            chi_gt_ts = Chi.reshape(Chi.shape[0], -1)
            Tau = jnp.load(dataset_dir / "Tau_val.npy")
            n_chi = chi_gt_ts.shape[-1]
            tau_ts = Tau.reshape(-1, Tau.shape[-1])

            # set the time steps
            ts = dt * jnp.arange(chi_gt_ts.shape[0])

            # differentiate the target poses to obtain the velocities and accelerations for visualization
            savgol_window_length = 51
            chi_d_gt_ts = savgol_filter(chi_gt_ts, window_length=savgol_window_length, polyorder=3, deriv=1, delta=dt, axis=0)
            chi_dd_gt_ts = savgol_filter(chi_gt_ts, window_length=savgol_window_length, polyorder=3, deriv=2, delta=dt, axis=0)
        case _:
            raise ValueError("Invalid dataset type.")
        
    # reshape the ground-truth data
    num_samples = chi_gt_ts.shape[0]
    num_markers_gt = chi_gt_ts.shape[1] // 3
    chi_gt_ts = chi_gt_ts.reshape(num_samples, -1, 3)
    chi_d_gt_ts = chi_d_gt_ts.reshape(num_samples, -1, 3)
    chi_dd_gt_ts = chi_dd_gt_ts.reshape(num_samples, -1, 3)
        
    # subsample the ground-truth data
    # marker sub-sampling
    # marker_indices = jnp.array([num_markers - 1])
    # marker_indices = jnp.array([num_markers // 2, num_markers - 1])
    # sub-sample the data
    chi_gt_ts = chi_gt_ts[:, marker_indices, :]
    chi_d_gt_ts = chi_d_gt_ts[:, marker_indices, :]
    chi_dd_gt_ts = chi_dd_gt_ts[:, marker_indices, :]
    # update the number of markers
    num_markers = marker_indices.shape[0]
    n_chi = chi_gt_ts.shape[-1]
        
    # load the predicted data
    rollout_ts_mdls = {}
    for model_type in model_types:
        rollout_ts = jnp.load(evaluation_dir / f"rollout_{model_type}_{dataset_type}.npz")

        # reshape the predicted data
        rollout_ts.update(dict(
            chi_ts=
        ))

        if model_type == "pcs_regression":
            # sub-sample the data

        rollout_ts_mdls[model_type] = rollout_ts

    # plot the end-effector x-ccordinate
