#%%
import numpy as np
import sys 
from sympy import symbols, simplify, derive_by_array, ordered, poly
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from xLSINDy import EulerLagrangeExpressionTensor
import sympy
import torch
torch.set_printoptions(precision=10)
import math
from math import pi
import HLsearch as HL
from itertools import chain
import dill
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, Dopri5, Euler, ODETerm, SaveAt, Tsit5
from jax import numpy as jnp
from jax import config, lax, vmap, jit
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
# Plotting settings
# plt.rc('font', family='serif', serif='Times')
# plt.rc('text', usetex=True)
# plt.rc('xtick', labelsize=10)
# plt.rc('ytick', labelsize=10)
# plt.rc('axes', labelsize=10)

from utils.math_utils import blk_diag
from utils.utils import compute_planar_stiffness_matrix, compute_strain_basis

####################################################################
#### Soft manipulator parameters - change based on the use case ####
num_segments = 2
strain_selector = np.array([True, True, True, True, True, True]) # bending, shear and axial
string_strains = ['Bending','Shear','Axial','Bending','Shear','Axial']

epsilon_bend = 5e-2
E_max = 1e8 
G_max = 1e5
G_max = 1e9
####################################################################
## Compute additional parameters
bending_map = [] # from the list of active states says which ones are bending state and which ones are not
for i in range(len(strain_selector)):
    if strain_selector[i]==True:
        if i%3==0:
            bending_map.append(True)
        else:
            bending_map.append(False)
bending_map = np.asarray(bending_map)
n_dof = np.sum(strain_selector)
strain_segments = [seg for seg in range(num_segments) for i in range(3)]

######################## Load dataset #############################
# rootdir = "./Source/Soft Robot/ns-1_dof-2/"
# rootdir = "./Source/Soft Robot/ns-1_bending_axial/"
# rootdir = "./Source/Soft Robot/ns-1_dof-3_stiff_shear_and_torques/"
# rootdir = "./Source/Soft Robot/ns-1_dof-3_G_1e7_true_acc/"
rootdir = "./Source/Soft Robot/ns-1_dof-3_G_1e6_torque_2_high_res/"
# rootdir = "./Source/Soft Robot/ns-1_dof-3/"
# rootdir = "./Source/Soft Robot/ns-1_dof-3_G_1e7_large_torque_larger_D/"
# rootdir = "./Source/Soft Robot/ns-1_dof-3_high_shear_stiffness_true_acc/"
# rootdir_true = "./Source/Soft Robot/ns-1_bending_axial_true/"
# rootdir_true = "./Source/Soft Robot/ns-1_dof-3_stiff_shear_and_torques_true_acc/"
# rootdir_true = "./Source/Soft Robot/ns-1_dof-3_G_1e6_true_acc/"
rootdir_true = "./Source/Soft Robot/ns-1_dof-3_G_1e6_torque_2_high_res_true_acc/"
# rootdir_true = "./Source/Soft Robot/ns-1_dof-3_true/"
# rootdir_true = "./Source/Soft Robot/ns-1_dof-3_G_1e7_large_torque_larger_D_true_acc/"
# rootdir_2 = "./Source/Soft Robot/ns-1_dof-3_G_1e7_large_torque_true_acc/"

rootdir = "./Source/Soft Robot/ns-2_dof-3_true/"
rootdir_true = "./Source/Soft Robot/ns-2_dof-3_true/"

X_all = np.load(rootdir + "X.npy")
Xdot_all = np.load(rootdir + "Xdot.npy")
Tau_all = np.load(rootdir + "Tau.npy")
X_all_true = np.load(rootdir_true + "X.npy")
Xdot_all_true = np.load(rootdir_true + "Xdot.npy")
Tau_all_true = np.load(rootdir_true + "Tau.npy")

# X_all_2 = np.load(rootdir_2 + "X.npy")
# Xdot_all_2 = np.load(rootdir_2 + "Xdot.npy")


# Stack variables (from all initial conditions)
X = np.vstack(X_all[:-1])
Xdot = np.vstack(Xdot_all[:-1])
Tau = np.vstack(Tau_all[:-1])
X_true = np.vstack(X_all_true[:-1])
Xdot_true = np.vstack(Xdot_all_true[:-1])
Tau_true = np.vstack(Tau_all_true[:-1])

# X_2 = np.vstack(X_all_2[:-1])
# Xdot_2 = np.vstack(Xdot_all_2[:-1])

# X[:,3:] = X_true[:,3:]
# Xdot[:,3:] = Xdot_true[:,3:]

X_val = np.array(X_all[-1])
Xdot_val = np.array(Xdot_all[-1])
Tau_val = np.array(Tau_all[-1])
X_val_true = np.array(X_all_true[-1])
Xdot_val_true = np.array(Xdot_all_true[-1])
Tau_val_true = np.array(Tau_all_true[-1])

# Tau = np.insert(Tau, [1], np.zeros((Tau.shape[0], 1)), axis=1)
# Tau_val = np.insert(Tau_val, [1], np.zeros((Tau_val.shape[0], 1)), axis=1)

# # Delete some strains
# X = np.delete(X, [1,4], 1)
# Xdot = np.delete(Xdot, [1,4], 1)
# Tau = np.delete(Tau, 1, 1)
# X_val = np.delete(X_val, [1,4], 1)
# Xdot_val = np.delete(Xdot_val, [1,4], 1)
# Tau_val = np.delete(Tau_val, 1, 1)

## Add true strains
# X_true = np.insert(X_true, [1,3], np.zeros((X_true.shape[0], 1)), axis=1)
# Xdot_true = np.insert(Xdot_true, [1,3], np.zeros((Xdot_true.shape[0], 1)), axis=1)
# X = np.insert(X, [1,3], np.zeros((X.shape[0], 1)), axis=1)
# Xdot = np.insert(Xdot, [1,3], np.zeros((Xdot.shape[0], 1)), axis=1)

for i in range(n_dof):
    fig, ax = plt.subplots(3,1)
    ax[2].plot(X[:,i], label='Estimate')
    ax[2].plot(X_true[:,i], label='GT')
    # ax[2].plot(X_2[:,i], label='Estimate (large torques & D=5e0)')
    ax[2].set_ylabel('$q$')
    # ax[2].legend(loc="upper right")
    ax[2].grid(True)
    ax[1].plot(Xdot[:,i], label='Estimate')
    ax[1].plot(X_true[:,n_dof+i], label='GT')
    # ax[1].plot(X_2[:,n_dof+i], label='Estimate (large torques & D=5e0)')
    ax[1].set_ylabel('$\dot{q}$')
    # ax[1].legend(loc="upper right")
    ax[1].grid(True)
    ax[0].plot(Xdot[:,n_dof+i], label='Estimate')
    ax[0].plot(Xdot_true[:,n_dof+i], label='GT')
    # ax[0].plot(Xdot_2[:,n_dof+i], label='Estimate (large torques & D=5e0)')
    ax[0].set_ylabel('$\ddot{q}$')
    # ax[0].legend(loc="upper right")
    ax[0].grid(True)
    fig.suptitle('Dataset - ' + string_strains[i])
    plt.show()

# Remove samples where bending is smaller than a certain threshold => allows better learning
bending_indices = [i for i in range(len(bending_map)) if bending_map[i]==True]
if bending_indices != []:
    mask = True
    for idx in bending_indices:
        mask = mask & (np.abs(X[:,idx]) >= 5.0)
        mask_true = mask & (np.abs(X_true[:,idx]) >= 5.0)
    X = X[mask]
    X_true = X_true[mask_true]
    Xdot = Xdot[mask]
    Xdot_true = Xdot_true[mask_true]
    Tau = Tau[mask]
    Tau_true = Tau_true[mask_true]

def apply_eps_to_bend_strains(q_bend, eps):

    q_bend_sign = np.sign(q_bend)
    q_bend_sign = np.where(q_bend_sign == 0, 1, q_bend_sign)

    q_epsed = np.select(
        [np.abs(q_bend)<eps, np.abs(q_bend)>=eps],            
        [q_bend_sign*eps, q_bend]
    )
    # old implementation
    # q_epsed = q_bend + (q_bend_sign * eps)
    return q_epsed

def apply_eps_to_bend_strains_jnp(q_bend, eps):

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

# Compute epsed bending states
X_epsed = np.zeros((X.shape[0], n_dof))
for i in range(n_dof):
    if bending_map[i] == True:
        q_epsed = apply_eps_to_bend_strains(X[:,i], epsilon_bend)
    else:
        q_epsed = X[:,i]
    
    X_epsed[:,i] = q_epsed

# for i in range(n_dof):
#     fig, ax = plt.subplots(3,1)
#     ax[2].plot(X_true[:,i], label='GT w/ Num Diff acc')
#     ax[2].plot(X[:,i], label='CV')
#     ax[2].set_ylabel('$q$')
#     ax[2].legend(loc="upper right")
#     ax[2].grid(True)
#     ax[1].plot(X_true[:,n_dof+i], label='GT w/ Num Diff acc')
#     ax[1].plot(X[:,n_dof+i], label='CV')
#     ax[1].set_ylabel('$\dot{q}$')
#     ax[1].legend(loc="upper right")
#     ax[1].grid(True)
#     ax[0].plot(Xdot_true[:,n_dof+i], label='GT w/ Num Diff acc')
#     ax[0].plot(Xdot[:,n_dof+i], label='CV')
#     ax[0].set_ylabel('$\ddot{q}$')
#     ax[0].legend(loc="upper right")
#     ax[0].grid(True)
#     fig.suptitle('Dataset after removing samples - ' + string_strains[i])
#     plt.show()

# Create the states nomenclature
states_dim = 2*n_dof  #q and q_dot
states = ()
states_dot = ()
for i in range(states_dim):
    if(i<states_dim//2):
        states = states + (symbols('x{}'.format(i)),)
        states_dot = states_dot + (symbols('x{}_t'.format(i)),)
    else:
        states = states + (symbols('x{}_t'.format(i-states_dim//2)),)
        states_dot = states_dot + (symbols('x{}_tt'.format(i-states_dim//2)),)
print('states are:',states)
print('states derivatives are: ', states_dot)

states_epsed =()
for i in range(len(states)//2):
    states_epsed = states_epsed + (symbols('x{}_epsed'.format(i)),)

# Turn from sympy to str
states_sym = list(states)
states_dot_sym = list(states_dot)
states_epsed_sym = list(states_epsed)
states = list(str(descr) for descr in states)
states_dot = list(str(descr) for descr in states_dot)
states_epsed = list(str(descr) for descr in states_epsed)

## Load basis functions and respective derivatives and double derivatives
basis_fcns_filepath = f"./Source/Soft Robot/symbolic_basis_functions/planar_pcs_ns-{num_segments}.dill"
expr_basis_fcns = dill.load(open(basis_fcns_filepath, 'rb'))
Lagr_expr = expr_basis_fcns['Lagr_expr']
true_coeffs_before_norm = expr_basis_fcns['true_coeffs_before_norm']
phi_q = expr_basis_fcns['phi_q']
phi_qdot2 = expr_basis_fcns['phi_qdot2']
phi_qdotq = expr_basis_fcns['phi_qdotq']
params = expr_basis_fcns['params']
B_xi = expr_basis_fcns['B_xi']

convergence = False
count = 0
while convergence == False:
    n_dof = len(states_sym)//2

    ########### From Lagrangian expr to EoM expr #############
    if count > 0:
        # if some strain was removed, need to recompute the derivatives and double derivatives
        phi_q, phi_qdot2, phi_qdotq = EulerLagrangeExpressionTensor(Lagr_expr, states, states_epsed_sym)
    
    phi_qdot2_expr = (sympy.Matrix(phi_qdot2.reshape(n_dof,-1)).T @ sympy.Matrix(states_dot_sym[n_dof:])).reshape(n_dof, phi_q.shape[1])
    phi_qdotq_expr = (sympy.Matrix(phi_qdotq.reshape(n_dof,-1)).T @ sympy.Matrix(states_dot_sym[:n_dof])).reshape(n_dof, phi_q.shape[1])
    phi_q_expr = sympy.Matrix(phi_q)
    EoMrhs_expr = phi_qdot2_expr + phi_qdotq_expr - phi_q_expr
    # Add entries for damping basis functions (velocity)
    EoMrhs_expr_array = np.asarray(EoMrhs_expr, dtype=object)
    EoMrhs_expr_array = np.append(EoMrhs_expr_array, np.diag(np.asarray(states_sym[n_dof:], dtype=object)), axis=1)
    EoMrhs_expr = sympy.Matrix(EoMrhs_expr_array)

    # Evaluate EoM basis functions on the training dataset for normalization
    # EoMrhs_lambda = sympy.lambdify([*states_sym[:n_dof], *states_sym[n_dof:], *states_dot_sym[n_dof:], *states_epsed_sym], EoMrhs_expr, 'jax')
    # def compute_EoMrhs(n_dof, EoMrhs_lambda, X, Xdot, X_epsed):
    #     EoMrhs = EoMrhs_lambda(*X[:n_dof], *X[n_dof:], *Xdot[n_dof:], *X_epsed[:])
    #     return EoMrhs
    # compute_batch_EoMrhs = vmap(compute_EoMrhs, in_axes=(None, None, 0, 0, 0), out_axes=2)
    # EoMrhs = compute_batch_EoMrhs(n_dof, EoMrhs_lambda, X, Xdot, X_epsed)
    # EoMrhs = torch.from_numpy(np.asarray(EoMrhs).copy())
    # EoMrhs = torch.flatten(EoMrhs.permute(0,2,1), end_dim=1)

    # Normalize EoM basis functions
    # norm_factor = (1/(EoMrhs.shape[0]))*torch.sum(torch.abs(EoMrhs), 0)
    norm_factor = torch.ones(len(Lagr_expr) + n_dof)
    for i in range(norm_factor.shape[0]):
        if norm_factor[i] == 0:
            norm_factor[i] = 1
    EoMrhs_bar_expr = sympy.Matrix(np.asarray(EoMrhs_expr, dtype=object) / np.asarray(norm_factor))

    # Least-square regression
    EoMrhs_bar_lambda = sympy.lambdify([*states_sym[:n_dof], *states_sym[n_dof:], *states_dot_sym[n_dof:], *states_epsed_sym], EoMrhs_bar_expr, 'jax')
    def compute_EoMrhs_bar(n_dof, EoMrhs_bar_lambda, X, Xdot, X_epsed):
        EoMrhs_bar = EoMrhs_bar_lambda(*X[:n_dof], *X[n_dof:], *Xdot[n_dof:], *X_epsed[:])
        return EoMrhs_bar
    compute_batch_EoMrhs_bar = vmap(compute_EoMrhs_bar, in_axes=(None, None, 0, 0, 0), out_axes=2)
    EoMrhs_bar = compute_batch_EoMrhs_bar(n_dof, EoMrhs_bar_lambda, X, Xdot, X_epsed)
    EoMrhs_bar = torch.from_numpy(np.asarray(EoMrhs_bar).copy())
    EoMrhs_bar = torch.flatten(EoMrhs_bar.permute(0,2,1), end_dim=1)

    coeffs_after_norm = torch.linalg.inv(EoMrhs_bar.T @ EoMrhs_bar) @ EoMrhs_bar.T @ (torch.flatten(torch.from_numpy(Tau).T).reshape(-1,1))
    coeffs_before_norm = coeffs_after_norm[:,0] / norm_factor
    xi_L = coeffs_before_norm[:-n_dof]
    D = torch.diag(coeffs_before_norm[-n_dof:])
    print('-Stiffness')
    print(xi_L[-n_dof:])
    print('D:')
    print(D)

    # xi_L = torch.from_numpy(true_coeffs_before_norm)
    # D = B_xi.T @ params['D'] @ B_xi
    # D = torch.from_numpy(np.array(D))
    # coeffs_after_norm = torch.cat((xi_L, torch.diagonal(D)))
    # coeffs_after_norm = coeffs_after_norm.reshape(-1,1)

    # Obtain the inverse dynamics EoM (after multiplying by coefficients)
    inverse_dynamics_expr = EoMrhs_bar_expr @ sympy.Matrix(coeffs_after_norm.detach().cpu().numpy())
    # inverse_dynamics_expr = EoMrhs_expr @ sympy.Matrix(coeffs_after_norm.detach().cpu().numpy())
    inverse_dynamics_expr_lambda = sympy.lambdify([*states_sym[:n_dof], *states_sym[n_dof:], *states_dot_sym[n_dof:], *states_epsed_sym], inverse_dynamics_expr, 'jax')

    #### Training loss ####
    def loss(pred, targ):
        loss = torch.mean((targ - pred)**2) 
        return loss 

    def compute_Tau(inverse_dynamics_expr_lambda, X, Xdot, X_epsed):
        tau = inverse_dynamics_expr_lambda(*X[:n_dof], *X[n_dof:], *Xdot[n_dof:], *X_epsed[:])[:,0]
        return tau
    compute_batch_tau = vmap(compute_Tau, in_axes=(None, 0, 0, 0), out_axes=0)
    tau_pred = compute_batch_tau(inverse_dynamics_expr_lambda, X, Xdot, X_epsed)

    lossval = loss(torch.from_numpy(np.asarray(tau_pred).copy()), torch.from_numpy(Tau))
    print('\nTraining loss:')
    print(lossval)

    fig, ax = plt.subplots(n_dof,1)
    if n_dof == 1:
        ax.plot(Tau[:,0], label='True Model')
        ax.plot(tau_pred[:,0], 'r--',label='Predicted Model')
        ax.set_ylabel('$Tau$')
        ax.grid(True)
    else:
        for i in range(n_dof):
            ax[i].plot(Tau[:,i], label='True Model')
            ax[i].plot(tau_pred[:,i], 'r--',label='Predicted Model')
            ax[i].set_ylabel('$Tau$')
            ax[i].grid(True)
    plt.show()

    # Get estimated stiffness
    stiffness = -2*np.array(xi_L[-n_dof:])
    # Get maximum stiffness
    max_stiffness = np.zeros((n_dof,))
    for i in range(n_dof):
        if string_strains[i] == 'Bending':
            max_stiffness[i] = (0.25*(np.pi*(params['r'][strain_segments[i]])**4)*E_max)
        elif string_strains[i] == 'Shear':
            max_stiffness[i] = ((4/3)*np.pi*((params['r'][strain_segments[i]])**2)*G_max)
        else:
            max_stiffness[i] = (np.pi*((params['r'][strain_segments[i]])**2)*E_max)

    neglect_strain_index = np.nonzero((stiffness > max_stiffness))[0]
    # neglect_strain_index = np.array([])
    if neglect_strain_index.size == 0: # all norms are above the threshold
        convergence = True
        print('\nResult:')
        print('No (more) strains will be deactivated.')
    else:
        count += 1
        print('\nResult:')

        for index in neglect_strain_index:
            print("Strain `" + string_strains[index] + "` will be deactivated.")
            if bending_map[index] == False:
                Lagr_expr = list(
                    sympy.Matrix(Lagr_expr).subs([
                        (states_sym[index], 0),
                        (states_sym[index+n_dof], 0)
                    ])
                )

                # kinetic_energy = kinetic_energy.subs([
                #     (states_sym[index], 0),
                #     (states_sym[index+n_dof], 0),
                #     (states_epsed_sym[index], 0),
                # ])

                # potential_energy = potential_energy.subs([
                #     (states_sym[index], 0),
                #     (states_sym[index+n_dof], 0),
                #     (states_epsed_sym[index], 0),
                # ])
            else:
                Lagr_expr = list(
                    sympy.Matrix(Lagr_expr).subs([
                        (states_sym[index], epsilon_bend),
                        (states_sym[index+n_dof], 0)
                    ])
                )

                # kinetic_energy = kinetic_energy.subs([
                #     (states_sym[index], epsilon_bend),
                #     (states_sym[index+n_dof], 0),
                #     (states_epsed_sym[index], epsilon_bend),
                # ])

                # potential_energy = potential_energy.subs([
                #     (states_sym[index], epsilon_bend),
                #     (states_sym[index+n_dof], 0),
                #     (states_epsed_sym[index], epsilon_bend),
                # ])

        # new_Lagr_expr = []
        # [new_Lagr_expr.append(x) for x in Lagr_expr if x not in new_Lagr_expr and list(x.free_symbols)!=[]]
        # Lagr_expr = new_Lagr_expr

        # Check if there's repeated basis functions or independent terms (Lagrangian is invariant to constants)
        new_Lagr_expr = []
        for expr in Lagr_expr:
            p = expr.as_poly()
            if p != None:
                monom = [sympy.prod(x**k for x, k in zip(p.gens, mon)) for mon in p.monoms()]
                for m in monom:
                    if m not in new_Lagr_expr:
                        new_Lagr_expr.append(m)
        Lagr_expr = new_Lagr_expr


        states_sym = [ele for idx, ele in enumerate(states_sym) if ((idx != neglect_strain_index[0]) and (idx != neglect_strain_index[0]+n_dof))]
        states_dot_sym = [ele for idx, ele in enumerate(states_dot_sym) if ((idx != neglect_strain_index[0]) and (idx != neglect_strain_index[0]+n_dof))]

        states = [ele for idx, ele in enumerate(states) if ((idx != neglect_strain_index[0]) and (idx != neglect_strain_index[0]+n_dof))]
        states_dot = [ele for idx, ele in enumerate(states_dot) if ((idx != neglect_strain_index[0]) and (idx != neglect_strain_index[0]+n_dof))]

        states_epsed = [ele for idx, ele in enumerate(states_epsed) if ((idx != neglect_strain_index[0]) and (idx != neglect_strain_index[0]+n_dof))]
        states_epsed_sym = [ele for idx, ele in enumerate(states_epsed_sym) if ((idx != neglect_strain_index[0]) and (idx != neglect_strain_index[0]+n_dof))]

        string_strains = [ele for idx, ele in enumerate(string_strains) if (idx != neglect_strain_index[0])]
        bending_map = [ele for idx, ele in enumerate(bending_map) if (idx != neglect_strain_index[0])]
        strain_segments = [ele for idx, ele in enumerate(strain_segments) if (idx != neglect_strain_index[0])]

        # Delete neglected strains
        X = np.delete(X, [neglect_strain_index[0], neglect_strain_index[0]+n_dof], 1)
        Xdot = np.delete(Xdot, [neglect_strain_index[0], neglect_strain_index[0]+n_dof], 1)
        X_epsed = np.delete(X_epsed, neglect_strain_index[0], 1)
        Tau = np.delete(Tau, neglect_strain_index[0], 1)
        X_val = np.delete(X_val, [neglect_strain_index[0], neglect_strain_index[0]+n_dof], 1)
        Xdot_val = np.delete(Xdot_val, [neglect_strain_index[0], neglect_strain_index[0]+n_dof], 1)
        Tau_val = np.delete(Tau_val, neglect_strain_index[0], 1)
        X_val_true = np.delete(X_val_true, [neglect_strain_index[0], neglect_strain_index[0]+n_dof], 1)
        Xdot_val_true = np.delete(Xdot_val_true, [neglect_strain_index[0], neglect_strain_index[0]+n_dof], 1)


# ------------------- Validation ---------------------------
# obtain the terms of the Euler-Lagrange EoM
def getEOM(xi_Lcpu, phi_q, phi_qdot2, phi_qdotq):

    delta_expr = sympy.Matrix(phi_q) @ sympy.Matrix(xi_Lcpu)
    eta_expr = (sympy.Matrix(phi_qdotq.reshape(n_dof*n_dof, -1)) @ sympy.Matrix(xi_Lcpu)).reshape(n_dof, n_dof)
    zeta_expr = (sympy.Matrix(phi_qdot2.reshape(n_dof*n_dof, -1)) @ sympy.Matrix(xi_Lcpu)).reshape(n_dof, n_dof)

    delta_expr_lambda = sympy.lambdify([*states_sym[:n_dof], *states_sym[n_dof:], *states_epsed_sym], delta_expr, 'jax')
    eta_expr_lambda = sympy.lambdify([*states_sym[:n_dof], *states_sym[n_dof:], *states_epsed_sym], eta_expr, 'jax')
    zeta_expr_lambda = sympy.lambdify([*states_sym[:n_dof], *states_epsed_sym], zeta_expr, 'jax')

    return delta_expr_lambda, eta_expr_lambda, zeta_expr_lambda

delta_expr_lambda, eta_expr_lambda, zeta_expr_lambda = getEOM(xi_L.detach().cpu().numpy(), phi_q, phi_qdot2, phi_qdotq)

def generate_data(func, time, init_values, Tau):
    # sol = solve_ivp(func,[time[0],time[-1]],init_values,t_eval=time,method='RK45',rtol=1e-10,atol=1e-10)
    dt = time[1]-time[0]
    sol_list = []
    sol_list.append(init_values)

    indexes = np.unique(Tau[:,0], return_index=True)[1]
    tau_unique = [Tau[index,:] for index in sorted(indexes)]

    for count, t in enumerate(time[::10]):
        tau = Tau[count*10, :]
        # tau = tau_unique[count]
        
        if t==0:
            sol = diffeqsolve(
                ODETerm(func),
                solver=Tsit5(),
                t0=time[0],
                t1=time[::10][1],
                dt0=1e-5,
                y0=init_values,
                args=(tau, D.detach().cpu().numpy()),
                max_steps=None,
                saveat=SaveAt(ts=jnp.arange(0.0, time[::10][1]+dt, dt)),
            )
        else:
            sol = diffeqsolve(
                ODETerm(func),
                solver=Tsit5(),
                t0=time[0],
                t1=time[::10][1],
                dt0=1e-5,
                y0=sol_list[-1][-1],
                args=(tau, D.detach().cpu().numpy()),
                max_steps=None,
                saveat=SaveAt(ts=jnp.arange(0.0, time[::10][1]+dt, dt)),
            )
        
        sol_list.append(sol.ys[1:,:])
    
    sol_array = np.vstack(sol_list)
    sol_array = sol_array[:-1,:]

    # acc_list = []
    # for i in range(sol_array.shape[0]):
    #     acc_list.append(func(time[i], sol_array[i,:], (Tau[i,:], D.detach().cpu().numpy())))
    #     print(i)
    # acc_array = np.asarray(acc_list)

    acc_gradient = np.gradient(sol_array[:, (sol_array.shape[1] // 2):], dt, axis=0)
    xdot = np.concatenate((sol_array[:, (sol_array.shape[1] // 2):], acc_gradient), axis=1)

    return sol_array, xdot
    # first output is X array in [x,x_dot] format
    # second output is X_dot array in [x_dot,x_doubledot] format

@jit
def softrobot(t,x,args):
    x_ = x[:x.shape[0]//2]
    x_t = x[x.shape[0]//2:]

    x_epsed_list = []
    for i in range(x.shape[0]//2):
        if bending_map[i] == True:
            q_epsed = apply_eps_to_bend_strains_jnp(x_[i], 6e0)
        else:
            q_epsed = x_[i]
        
        x_epsed_list.append(q_epsed)
    
    x_epsed = jnp.asarray(x_epsed_list)
    
    tau, D = args
    x_tt = jnp.linalg.inv(zeta_expr_lambda(*x_, *x_epsed).T) @ (tau - D @ x_t + delta_expr_lambda(*x_, *x_t, *x_epsed)[:,0] - eta_expr_lambda(*x_, *x_t, *x_epsed).T @ x_t)
    return jnp.concatenate([x_t, x_tt])

## Validation results ##
# true validation dataset
q_tt_true = (Xdot_val_true[:,n_dof:].T).copy()
q_t_true = (Xdot_val_true[:,:n_dof].T).copy()
q_true = (X_val_true[:,:n_dof].T).copy()

# CV validation dataset
q_tt_cv = (Xdot_val[:,n_dof:].T).copy()
q_t_cv = (Xdot_val[:,:n_dof].T).copy()
q_cv = (X_val[:,:n_dof].T).copy()
tau_true = (Tau_val.T).copy()

# prediction results
dt = 1e-3  # time step
time_ = jnp.arange(0.0, 0.5, dt)
y_0 = X_val[0,:]
Xpred, Xdotpred = generate_data(softrobot, time_, y_0, Tau_val)

q_tt_pred = Xdotpred[:,n_dof:].T
q_t_pred = Xdotpred[:,:n_dof].T
q_pred = Xpred[:,:n_dof].T

save = True
if save==True:
    np.save("./Source/Soft Robot/render_data/Xpred.npy" , Xpred)

# Validation loss
X_epsed_val = np.zeros((X_val.shape[0], n_dof))
for i in range(n_dof):
    if bending_map[i] == True:
        q_epsed = apply_eps_to_bend_strains(X_val[:,i], 3e0)
    else:
        q_epsed = X_val[:,i]
    
    X_epsed_val[:,i] = q_epsed

tau_pred = compute_batch_tau(inverse_dynamics_expr_lambda, X_val, Xdot_val, X_epsed_val)
lossval = loss(torch.from_numpy(np.asarray(tau_pred.T).copy()), torch.from_numpy(tau_true))
print('\nValidation loss:')
print(lossval)

fig, ax = plt.subplots(n_dof,1)
if n_dof == 1:
    ax.plot(torch.from_numpy(tau_true)[0,:], label='GT')
    ax.plot(tau_pred[:,0], 'r--',label='Predicted Model')
    ax.set_ylabel('$Tau$')
    ax.grid(True)
else:
    for i in range(n_dof):
        ax[i].plot(torch.from_numpy(tau_true)[i,:], label='GT')
        ax[i].plot(tau_pred[:,i], 'r--',label='Predicted Model')
        ax[i].set_ylabel('$Tau$')
        ax[i].grid(True)
plt.show()

## Plotting
t = time_

for i in range(n_dof):
    fig, ax = plt.subplots(3,1)

    ax[0].plot(t, q_tt_true[i,:], label='GT Data')
    # ax[0].plot(t, q_tt_cv[i,:], label='CV Data')
    ax[0].plot(t, q_tt_pred[i,:], 'r--',label='Simulated Predicted Model')
    ax[0].set_ylabel('$\ddot{q}$')
    ax[0].set_xlim([0,0.5])
    ax[0].grid(True)

    ax[1].plot(t, q_t_true[i,:], label='GT Data')
    # ax[1].plot(t, q_t_cv[i,:], label='CV Data')
    ax[1].plot(t, q_t_pred[i,:], 'r--',label='Simulated Predicted Model')
    ax[1].set_ylabel('$\dot{q}$')
    ax[1].set_xlim([0,0.5])
    ax[1].grid(True)

    ax[2].plot(t, q_true[i,:], label='GT Data')
    # ax[2].plot(t, q_cv[i,:], label='CV Data')
    ax[2].plot(t, q_pred[i,:], 'r--',label='Simulated Predicted Model')
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('$q$')
    ax[2].set_xlim([0,0.5])
    ax[2].grid(True)
    ax[2].legend(loc="upper right")

    # Line, Label = ax[0].get_legend_handles_labels()
    # fig.legend(Line, Label, loc='upper right')
    fig.suptitle('Simulation results xL-SINDY - ' + str(num_segments) + ' segments ' + str(n_dof) + ' DOF => ' + string_strains[i])

    fig.tight_layout()
    plt.show()

fig, ax = plt.subplots(n_dof, 1, sharex=True)
for i in range(n_dof):
    if string_strains[i] == 'Bending':
        ax[i].plot(t, q_pred[i,:], label='Obtained Model')
        ax[i].plot(t, q_true[i,:], label='From kinematic model')
        ax[i].set_ylabel('$\kappa_{be}$')
        ax[i].set_xlim([0,0.5])
        ax[i].grid(True)
        ax[i].legend(loc='lower right')
    elif string_strains[i] == 'Shear':
        ax[i].plot(t, q_pred[i,:], label='Obtained Model')
        ax[i].plot(t, q_true[i,:], label='From kinematic model')
        ax[i].set_ylabel('$\sigma_{sh}$')
        ax[i].set_xlim([0,0.5])
        ax[i].grid(True)
        ax[i].legend(loc='lower right')
    else:
        ax[i].plot(t, q_pred[i,:], label='Obtained Model')
        ax[i].plot(t, q_true[i,:], label='From kinematic model')
        ax[i].set_ylabel('$\sigma_{ax}$')
        ax[i].set_xlim([0,0.5])
        ax[i].grid(True)
        ax[i].legend(loc='lower right')

fig.set_size_inches(5, 5 / 1.618 )
fig.tight_layout()
plt.show()

# np.save("./Source/Soft Robot/render_data/q_cv_task1.npy", q_true)
np.save("./Source/Soft Robot/render_data/q_pred_task4_with_shear.npy", q_pred)


# kinetic_energy_lambda = sympy.lambdify([*states_sym[:n_dof], *states_sym[n_dof:], *states_epsed_sym], kinetic_energy, 'jax')
# potential_energy_lambda = sympy.lambdify([*states_sym[:n_dof], *states_sym[n_dof:], *states_epsed_sym], potential_energy, 'jax')
# def compute_energy(n_dof, lambda_function, X, Xdot, X_epsed):
#     energy = lambda_function(*X[:n_dof], *X[n_dof:], *X_epsed[:])
#     return energy

# compute_batch_energy = vmap(compute_energy, in_axes=(None, None, 0, 0, 0), out_axes=0)
# kinetic_energy = compute_batch_energy(n_dof, kinetic_energy_lambda, X_val, Xdot_val, X_epsed_val)
# potential_energy = compute_batch_energy(n_dof, potential_energy_lambda, X_val, Xdot_val, X_epsed_val)

# fig, ax = plt.subplots(1,1)
# ax.plot(kinetic_energy, label='Kinetic energy')
# ax.plot(potential_energy, label='Potential energy')
# ax.set_ylabel('Energy')
# Line, Label = ax.get_legend_handles_labels()
# fig.legend(Line, Label, loc='upper right')
# ax.grid(True)
# plt.show()