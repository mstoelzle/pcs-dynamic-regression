import jax
import numpy as onp
import os

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter

# parameters
num_segments = 2
case_dir = Path("Source") / "Soft Robot" / f"ns-2_dof-3"
evaluation_dir = case_dir / "evaluation"
dataset_type = "val"
assert dataset_type in ["train", "val"]
marker_indices = jnp.array([10, 15, 20])
model_types = ["node", "con", "pcs_regression"]
assert all(model_type in ["pcs_regression", "node", "lstm", "con"] for model_type in model_types)

# time step
dt = 1e-3
time_step_skip = 5

# plotting settings
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Romand"],
    }
)

# if dataset_type == "train":
#     figsize = (5.0, 3.0)
# else:
#     figsize = (5.0, 3.0)
figsize = (5.0, 4.5)
dpi = 200
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt_markers = ["o", "s", "^", "v", "D", "P", "X", "H"]
plt_marker_skip = 1
plt_lw = 2.2

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
            chi_gt_ts = Chi.transpose((0, 2, 1)).reshape(Chi.shape[0], -1)
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

    if dataset_type == "val":
        # subsample the ground-truth data time steps
        ts = ts[::time_step_skip]
        chi_gt_ts = chi_gt_ts[::time_step_skip]
        chi_d_gt_ts = chi_d_gt_ts[::time_step_skip]
        chi_dd_gt_ts = chi_dd_gt_ts[::time_step_skip]
        tau_ts = tau_ts[::time_step_skip]
        
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
        print(f"Loading model {model_type}...")
        if model_type == "pcs_regression":
            chi_ts = jnp.load(evaluation_dir / f"rollout_{model_type}_{dataset_type}.npy")
            # sub-sample the data
            chi_ts = chi_ts[:, marker_indices-1, :]  # -1 to account that predictions start the at the second marker
            rollout_ts = dict(chi_ts=chi_ts)
        else:
            rollout_ts = dict(jnp.load(evaluation_dir / f"rollout_{model_type}_{dataset_type}.npz"))

            # reshape the predicted data
            rollout_ts.update(dict(
                chi_ts=rollout_ts["x_ts"].reshape(num_samples, -1, 3),
                chi_d_ts=rollout_ts["x_d_ts"].reshape(num_samples, -1, 3),
            ))

            if dataset_type == "val":
                # sub-sample the time steps
                rollout_ts = {key: rollout_ts[key][::time_step_skip] for key in rollout_ts}

        # compute the errors
        # body shape error
        position_error = jnp.mean(jnp.linalg.norm(chi_gt_ts[..., :2] - rollout_ts["chi_ts"][..., :2], axis=-1))
        orientation_error = jnp.mean(jnp.abs(chi_gt_ts[..., 2] - rollout_ts["chi_ts"][..., 2]))
        print(f"Model {model_type} - Position error: {position_error * 1e3:.4f} mm, Orientation error: {orientation_error:.4f} rad")
        # end-effector error
        end_effector_position_error = jnp.mean(jnp.linalg.norm(chi_gt_ts[..., -1, :2] - rollout_ts["chi_ts"][..., -1, :2], axis=-1))
        end_effector_orientation_error = jnp.mean(jnp.abs(chi_gt_ts[..., -1, 2] - rollout_ts["chi_ts"][..., -1, 2]))
        print(f"Model {model_type} - End-effector Position error: {end_effector_position_error * 1e3:.4f} mm, End-effector Orientation error: {end_effector_orientation_error:.4f} rad")

        rollout_ts_mdls[model_type] = rollout_ts

    for i in range(num_markers):
        marker_idx = marker_indices[i]

        fig, axes = plt.subplots(3, 1, figsize=figsize, dpi=dpi)
        axes[0].plot(ts, chi_gt_ts[:, i, 0], label='Ground-truth', linewidth=4, linestyle='dotted', color='k')
        axes[1].plot(ts, chi_gt_ts[:, i, 1], label='Ground-truth', linewidth=4, linestyle='dotted', color='k')
        axes[2].plot(ts, chi_gt_ts[:, i, 2], label='Ground-truth', linewidth=4, linestyle='dotted', color='k')
        for model_type in model_types:
            line_label = "PCS Regression (Ours)" if model_type == "pcs_regression" else model_type.upper()
            axes[0].plot(ts, rollout_ts_mdls[model_type]["chi_ts"][:, i, 0], linewidth=plt_lw, label=line_label)
            axes[1].plot(ts, rollout_ts_mdls[model_type]["chi_ts"][:, i, 1], linewidth=plt_lw, label=line_label)
            axes[2].plot(ts, rollout_ts_mdls[model_type]["chi_ts"][:, i, 2], linewidth=plt_lw, label=line_label)
        axes[0].set_ylabel(r'Position $p_\mathrm{x}$ [m]')
        axes[1].set_ylabel(r'Position $p_\mathrm{y}$ [m]')
        axes[2].set_ylabel(r'Orientation $\theta$ [rad]')
        axes[-1].set_xlabel(r'Time $t$ [s]')
        axes[0].legend(fontsize=8)
        for j, ax in enumerate(axes):
            ax.set_xlim([0, ts[-1]])
            if dataset_type == "val":
                y_min = jnp.min(chi_gt_ts[:, i, j])
                y_max = jnp.max(chi_gt_ts[:, i, j])
                y_delta = (y_max - y_min)
                ax.set_ylim([y_min - 0.8 * y_delta, y_max + 0.8 * y_delta])
            # ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(evaluation_dir / f"rollout_{dataset_type}_marker_{marker_idx}.pdf", bbox_inches='tight')
        plt.show()

    # plot the torque
    fig, axes = plt.subplots(3, 1, figsize=figsize, dpi=dpi)
    axes[0].plot(ts, tau_ts[:, 0], linewidth=plt_lw, color=colors[0], label=r'$\tau_1$')
    axes[1].plot(ts, tau_ts[:, 1], linewidth=plt_lw, color=colors[1], label=r'$\tau_2$')
    axes[2].plot(ts, tau_ts[:, 2], linewidth=plt_lw, color=colors[2], label=r'$\tau_3$')
    axes[0].plot(ts, tau_ts[:, 3], linewidth=plt_lw, color=colors[3], label=r'$\tau_4$')
    axes[1].plot(ts, tau_ts[:, 4], linewidth=plt_lw, color=colors[4], label=r'$\tau_5$')
    axes[2].plot(ts, tau_ts[:, 5], linewidth=plt_lw, color=colors[5], label=r'$\tau_6$')

    axes[-1].set_xlabel(r'Time $t$ [s]')
    axes[0].set_ylabel(r'Bend. torques [Nm]')
    axes[1].set_ylabel(r'Shear forces [N]')
    axes[2].set_ylabel(r'Elongation forces [N]')
    for j, ax in enumerate(axes):
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(evaluation_dir / f"torque_{dataset_type}.pdf", bbox_inches='tight')
    plt.show()