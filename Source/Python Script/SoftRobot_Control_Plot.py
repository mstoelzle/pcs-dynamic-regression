import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import Array, jit, lax, random, vmap
import numpy as onp
from pathlib import Path
from typing import Dict, Tuple

num_segments = 2
control_dir = Path(f"./Source/Soft Robot/ns-{num_segments}_dof-3/control")

# plotting settings
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Romand"],
    }
)

figsize = (4.0, 3.8)
dpi = 200
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


if __name__ == "__main__":
    # load the simulation data
    sim_ts = jnp.load(str(control_dir / 'closed_loop_control_data.npz'))

    # plot the bending strains
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    # plot the reference trajectory
    ax.plot(sim_ts['ts'], sim_ts['q_des_ts'][:, 0], linewidth=3.5, linestyle=':', label=r"$\kappa_{\mathrm{be},1}^\mathrm{d}$")
    ax.plot(sim_ts['ts'], sim_ts['q_des_ts'][:, 3], linewidth=3.5, linestyle=':', label=r"$\kappa_{\mathrm{be},2}^\mathrm{d}$")
    # reset the color cycle
    ax.set_prop_cycle(None)
    # plot the actual trajectory
    ax.plot(sim_ts['ts'], sim_ts['q_ts'][:, 0], linewidth=2.0, label=r"$\kappa_{\mathrm{be},1}$")
    ax.plot(sim_ts['ts'], sim_ts['q_ts'][:, 3], linewidth=2.0, label=r"$\kappa_{\mathrm{be},2}$")
    ax.set_xlabel(r"Time $t$ [s]")
    ax.set_ylabel(r"Bending strain $\kappa_\mathrm{be}$ [rad/m]")
    ax.legend(fontsize=8)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(str(control_dir / 'setpoint_control_bending.pdf'))
    plt.savefig(str(control_dir / 'setpoint_control_bending.svg'))
    plt.show()

    # plot the axial and shear strains
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    # plot the reference trajectory for the shear strains
    ax.plot(sim_ts['ts'], sim_ts['q_des_ts'][:, 1], linewidth=3.5, linestyle=':', label=r"$\sigma_{\mathrm{sh},1}^\mathrm{d}$")
    ax.plot(sim_ts['ts'], sim_ts['q_des_ts'][:, 4], linewidth=3.5, linestyle=':', label=r"$\sigma_{\mathrm{sh},2}^\mathrm{d}$")
    # plot the reference trajectory for the axial strains
    ax.plot(sim_ts['ts'], sim_ts['q_des_ts'][:, 2], linewidth=3.5, linestyle=':', label=r"$\sigma_{\mathrm{ax},1}^\mathrm{d}$")
    ax.plot(sim_ts['ts'], sim_ts['q_des_ts'][:, 5], linewidth=3.5, linestyle=':', label=r"$\sigma_{\mathrm{ax},2}^\mathrm{d}$")
    # reset the color cycle
    ax.set_prop_cycle(None)
    # plot the actual trajectory for the shear strains
    ax.plot(sim_ts['ts'], sim_ts['q_ts'][:, 1], linewidth=2.0, label=r"$\sigma_{\mathrm{sh},1}$")
    ax.plot(sim_ts['ts'], sim_ts['q_ts'][:, 4], linewidth=2.0, label=r"$\sigma_{\mathrm{sh},2}$")
    # plot the actual trajectory for the axial strains
    ax.plot(sim_ts['ts'], sim_ts['q_ts'][:, 2], linewidth=2.0, label=r"$\sigma_{\mathrm{ax},1}$")
    ax.plot(sim_ts['ts'], sim_ts['q_ts'][:, 5], linewidth=2.0, label=r"$\sigma_{\mathrm{ax},2}$")
    ax.set_xlabel(r"Time $t$ [s]")
    ax.set_ylabel(r"Linear strains $\sigma$ [-]")
    ax.legend(ncols=2, fontsize=8)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(str(control_dir / 'setpoint_control_linear.pdf'))
    plt.savefig(str(control_dir / 'setpoint_control_linear.svg'))
    plt.show()

    # plot the control signal
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    for i in range(sim_ts["tau_ts"].shape[-1]):
        ax.plot(sim_ts["ts"], sim_ts["tau_ts"][:, i], linewidth=2.0, label=r"$\tau_{" + str(i+1) + "}$")
    ax.set_xlabel(r"Time $t$ [s]")
    ax.set_ylabel(r"Control signal $\tau$")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(str(control_dir / 'setpoint_control_signal.pdf'))
    plt.savefig(str(control_dir / 'setpoint_control_signal.svg'))
    plt.show()

    # plot the feedforward and feedback control signals
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    # plot the feedforward control signal
    for i in range(sim_ts["tau_ff_ts"].shape[-1]):
        ax.plot(sim_ts["ts"], sim_ts["tau_ff_ts"][:, i], marker="o", ms=5, markevery=100, label=r"$\tau_\mathrm{ff}^{" + str(i+1) + "}$")
    # reset the color cycle
    ax.set_prop_cycle(None)
    # plot the feedback control signal
    for i in range(sim_ts["tau_fb_ts"].shape[-1]):
        ax.plot(sim_ts["ts"], sim_ts["tau_fb_ts"][:, i], marker="v", ms=5, markevery=100, label=r"$\tau_\mathrm{fb}^{" + str(i+1) + "}$")
    ax.set_xlabel(r"Time $t$ [s]")
    ax.set_ylabel(r"Control signal $\tau$")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(str(control_dir / 'setpoint_control_signal_ff_fb.pdf'))
    plt.savefig(str(control_dir / 'setpoint_control_signal_ff_fb.svg'))
    plt.show()

    # plot the the bending control signal
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    # plot the feedforward control signal
    ax.plot(sim_ts["ts"], sim_ts["tau_ff_ts"][:, i], marker="o", ms=5, markevery=100, label=r"$\tau_\mathrm{ff,1}$")
    ax.plot(sim_ts["ts"], sim_ts["tau_ff_ts"][:, 3], marker="o", ms=5, markevery=100, label=r"$\tau_\mathrm{ff,2}$")
    # reset the color cycle
    ax.set_prop_cycle(None)
    # plot the feedback control signal
    ax.plot(sim_ts["ts"], sim_ts["tau_fb_ts"][:, 0], marker="v", ms=5, markevery=100, label=r"$\tau_{\mathrm{fb},1}$")
    ax.plot(sim_ts["ts"], sim_ts["tau_fb_ts"][:, 3], marker="v", ms=5, markevery=100, label=r"$\tau_{\mathrm{fb},2}$")
    # reset the color cycle
    ax.set_prop_cycle(None)
    # plot the total control signal
    ax.plot(sim_ts["ts"], sim_ts["tau_ts"][:, 0], linewidth=2.0, label=r"$\tau_1$")
    ax.plot(sim_ts["ts"], sim_ts["tau_ts"][:, 3], linewidth=2.0, label=r"$\tau_2$")
    ax.set_xlabel(r"Time $t$ [s]")
    ax.set_ylabel(r"Bending torque [Nm]")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(str(control_dir / 'setpoint_control_signal_bending.pdf'))
    plt.savefig(str(control_dir / 'setpoint_control_signal_bending.svg'))
    plt.show()

    # plot the shear control signal
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    # plot the feedforward control signal
    ax.plot(sim_ts["ts"], sim_ts["tau_ff_ts"][:, 1], marker="o", ms=5, markevery=100, label=r"$\tau_\mathrm{ff,1}$")
    ax.plot(sim_ts["ts"], sim_ts["tau_ff_ts"][:, 4], marker="o", ms=5, markevery=100, label=r"$\tau_\mathrm{ff,2}$")
    # reset the color cycle
    ax.set_prop_cycle(None)
    # plot the feedback control signal
    ax.plot(sim_ts["ts"], sim_ts["tau_fb_ts"][:, 1], marker="v", ms=5, markevery=100, label=r"$\tau_{\mathrm{fb},1}$")
    ax.plot(sim_ts["ts"], sim_ts["tau_fb_ts"][:, 4], marker="v", ms=5, markevery=100, label=r"$\tau_{\mathrm{fb},2}$")
    # reset the color cycle
    ax.set_prop_cycle(None)
    # plot the total control signal
    ax.plot(sim_ts["ts"], sim_ts["tau_ts"][:, 1], linewidth=2.0, label=r"$\tau_1$")
    ax.plot(sim_ts["ts"], sim_ts["tau_ts"][:, 4], linewidth=2.0, label=r"$\tau_2$")
    ax.set_xlabel(r"Time $t$ [s]")
    ax.set_ylabel(r"Shear force [N]")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(str(control_dir / 'setpoint_control_signal_shear.pdf'))
    plt.savefig(str(control_dir / 'setpoint_control_signal_shear.svg'))
    plt.show()

    # plot the axial control signal
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    # plot the feedforward control signal
    ax.plot(sim_ts["ts"], sim_ts["tau_ff_ts"][:, 2], marker="o", ms=5, markevery=100, label=r"$\tau_\mathrm{ff,1}$")
    ax.plot(sim_ts["ts"], sim_ts["tau_ff_ts"][:, 5], marker="o", ms=5, markevery=100, label=r"$\tau_\mathrm{ff,2}$")
    # reset the color cycle
    ax.set_prop_cycle(None)
    # plot the feedback control signal
    ax.plot(sim_ts["ts"], sim_ts["tau_fb_ts"][:, 2], marker="v", ms=5, markevery=100, label=r"$\tau_{\mathrm{fb},1}$")
    ax.plot(sim_ts["ts"], sim_ts["tau_fb_ts"][:, 5], marker="v", ms=5, markevery=100, label=r"$\tau_{\mathrm{fb},2}$")
    # reset the color cycle
    ax.set_prop_cycle(None)
    # plot the total control signal
    ax.plot(sim_ts["ts"], sim_ts["tau_ts"][:, 2], linewidth=2.0, label=r"$\tau_1$")
    ax.plot(sim_ts["ts"], sim_ts["tau_ts"][:, 5], linewidth=2.0, label=r"$\tau_2$")
    ax.set_xlabel(r"Time $t$ [s]")
    ax.set_ylabel(r"Axial force [N]")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(str(control_dir / 'setpoint_control_signal_axial.pdf'))
    plt.savefig(str(control_dir / 'setpoint_control_signal_axial.svg'))
    plt.show()
