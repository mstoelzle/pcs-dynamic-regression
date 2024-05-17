import numpy as np
import sys 
from sympy import symbols, simplify, derive_by_array, ordered
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from xLSINDy import EulerLagrangeExpressionTensor, LagrangianLibraryTensor, ELforward
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

from utils.math_utils import blk_diag
from utils.utils import compute_planar_stiffness_matrix, compute_strain_basis

rootdir = "./Source/Strain_Regression_Pendulum/"  
noiselevel = 0

# Load dataset
X_all = np.load(rootdir + "X.npy")
Xdot_all = np.load(rootdir + "Xdot.npy")
Tau_all = np.load(rootdir + "Tau.npy")

# Stack variables (from all initial conditions)
X = (X_all[:-1])
Xdot = (Xdot_all[:-1])
Tau = (Tau_all[:-1])
X = np.vstack(X)
Xdot = np.vstack(Xdot)
Tau = np.vstack(Tau)

X_val = np.array(X_all[-1])
Xdot_val = np.array(Xdot_all[-1])
Tau_val = np.array(Tau_all[-1])

# Add dummy strains
num_dummy_strains = 1
q_dummy = np.zeros((X.shape[0] + X_val.shape[0], num_dummy_strains))
q_t_dummy = np.zeros((X.shape[0] + X_val.shape[0], num_dummy_strains))
q_tt_dummy = np.zeros((X.shape[0] + X_val.shape[0], num_dummy_strains))
for i in range(num_dummy_strains):
    noise = np.random.normal(loc=1e-3, scale=0, size=q_dummy.shape[0])
    q_dummy[:,i] = q_dummy[:,i] + noise
    q_t_dummy[:,i] = 1e-3*np.ones(q_t_dummy.shape[0])
    q_tt_dummy[:,i] = 1e-3*np.ones(q_tt_dummy.shape[0])
    # q_dummy[:,i] = savgol_filter(q_dummy[:,i] + noise, 1000, 3)
    # q_t_dummy[:,i] = savgol_filter(q_dummy[:,i], 1000, 3, deriv=1, delta=1e-4)
    # q_tt_dummy[:,i] = savgol_filter(q_dummy[:,i], 1000, 3, deriv=2, delta=1e-4)

# for i in range(num_dummy_strains):
#     fig, ax = plt.subplots(3,1)

#     ax[0].plot(q_tt_dummy[:5000,i])
#     ax[0].set_ylabel('$\ddot{q}$')
#     ax[0].grid(True)

#     ax[1].plot(q_t_dummy[:5000,i])
#     ax[1].set_ylabel('$\dot{q}$')
#     ax[1].grid(True)

#     ax[2].plot(q_dummy[:5000,i])
#     ax[2].set_ylabel('$q$')
#     ax[2].grid(True)

#     fig.suptitle('Data generation - Shear')
#     fig.tight_layout()
#     plt.show()

X = np.insert(X, [1,2], np.concatenate((q_dummy[:X.shape[0],:], q_t_dummy[:X.shape[0],:]), axis=1), axis=1)
X_val = np.insert(X_val, [1,2], np.concatenate((q_dummy[X.shape[0]:,:], q_t_dummy[X.shape[0]:,:]), axis=1), axis=1)
Xdot = np.insert(Xdot, [1,2], np.concatenate((q_t_dummy[:X.shape[0],:], q_tt_dummy[:X.shape[0],:]), axis=1), axis=1)
Xdot_val = np.insert(Xdot_val, [1,2], np.concatenate((q_t_dummy[X.shape[0]:,:], q_tt_dummy[X.shape[0]:,:]), axis=1), axis=1)
Tau = np.insert(Tau, [1], np.zeros((Tau.shape[0], 1)), axis=1)
Tau_val = np.insert(Tau_val, [1], np.zeros((Tau_val.shape[0], 1)), axis=1)


####################################################################
#### Double Pendulum parameters - change based on the use case ####
params = {
    "l1": 1.0,
    "l2": 1.0,
    "m1": 1.0,
    "m2": 1.0,
    "g": 10,
}
####################################################################

# adding noise
mu, sigma = 0, noiselevel
noise = np.random.normal(mu, sigma, X.shape[0])
for i in range(X.shape[1]):
    X[:,i] = X[:,i] + noise
    Xdot[:,i] = Xdot[:,i] + noise

# Create the states nomenclature
n_dof = 2
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

#Turn from sympy to str
states_sym = list(states)
states_dot_sym = list(states_dot)
states = list(str(descr) for descr in states)
states_dot = list(str(descr) for descr in states_dot)

########################################################
############ Create list of basis functions ############
def constructLagrangian(states_sym):
    l1 = sympy.Symbol('l1', nonnegative=True, nonzero=True)
    l2 = sympy.Symbol('l2', nonnegative=True, nonzero=True)
    m1 = sympy.Symbol('m1', nonnegative=True, nonzero=True)
    m2 = sympy.Symbol('m2', nonnegative=True, nonzero=True)
    g = sympy.Symbol('g')

    t1_idx = [idx for idx, ele in enumerate(states_sym) if str(ele)==('x0')]
    if t1_idx != []:
        t1 = states_sym[t1_idx[0]]
    else:
        t1 = 0

    t2_idx = [idx for idx, ele in enumerate(states_sym) if str(ele)==('x1')]
    if t2_idx != []:
        t2 = states_sym[t2_idx[0]]
    else:
        t2 = 0
    
    w1_idx = [idx for idx, ele in enumerate(states_sym) if str(ele)==('x0_t')]
    if w1_idx != []:
        w1 = states_sym[w1_idx[0]]
    else:
        w1 = 0

    w2_idx = [idx for idx, ele in enumerate(states_sym) if str(ele)==('x1_t')]
    if w2_idx != []:
        w2 = states_sym[w2_idx[0]]
    else:
        w2 = 0

    L = 0.5*(m1+m2)*(l1**2)*w1**2 + 0.5*m2*(l2**2)*(w1+w2)**2 + \
        m2*l1*l2*w1*(w1+w2)*(sympy.cos(t2)) + \
        (m1+m2)*g*l1*sympy.cos(t1) + m2*g*l2*sympy.cos(t1+t2)
    
    symbols = list(ordered(list(L.free_symbols))) # g, l1, l2, m1, m2, x0, x0_t, x1, x1_t
    L = L.subs([
        (symbols[0], params['g']),
        (symbols[1], params['l1']),
        (symbols[2], params['l2']),
        (symbols[3], params['m1']),
        (symbols[4], params['m2'])
    ])

    p = L.as_poly()
    coeffs = np.asarray(p.coeffs(), dtype=np.float64)
    monoms = [sympy.prod(x**k for x, k in zip(p.gens, mon)) for mon in p.monoms()]

    return monoms, coeffs

convergence = False
threshold = 2
Lagr_expr, true_coeffs_before_norm = constructLagrangian(states_sym)

while convergence == False:
    n_dof = len(states_sym)//2
    
    # In case there is independent term, remove it from both lists (Lagrangian is invariant to constants)
    # true_coeffs_before_norm = np.asarray([ele for idx, ele in enumerate(true_coeffs_before_norm) if list(Lagr_expr[idx].free_symbols)!=[]])
    Lagr_expr = [ele for idx, ele in enumerate(Lagr_expr) if list(Lagr_expr[idx].free_symbols)!=[]]

    # Compute the coefficient mapping matrix
    mapping_matrix = np.zeros((n_dof, len(Lagr_expr)))
    for i in range(len(Lagr_expr)):
        for q in range(n_dof):
            if (states_sym[q] in list(ordered(list(Lagr_expr[i].free_symbols)))) or (states_sym[q+n_dof] in list(ordered(list(Lagr_expr[i].free_symbols)))):
                mapping_matrix[q,i] = 1

    ########### From Lagrangian expr to EoM expr #############
    phi_q, phi_qdot2, phi_qdotq = EulerLagrangeExpressionTensor(Lagr_expr, states)
    # Turn to mutable objects
    phi_q = np.asarray(phi_q, dtype=object)
    phi_qdot2 = np.asarray(phi_qdot2, dtype=object)
    phi_qdotq = np.asarray(phi_qdotq, dtype=object)

    phi_qdot2_expr = (sympy.Matrix(phi_qdot2.reshape(n_dof,-1)).T @ sympy.Matrix(states_dot_sym[n_dof:])).reshape(n_dof, phi_q.shape[1])
    phi_qdotq_expr = (sympy.Matrix(phi_qdotq.reshape(n_dof,-1)).T @ sympy.Matrix(states_dot_sym[:n_dof])).reshape(n_dof, phi_q.shape[1])
    phi_q_expr = sympy.Matrix(phi_q)
    EoMrhs_expr = phi_qdot2_expr + phi_qdotq_expr - phi_q_expr

    # Evaluate EoM basis functions on the training dataset for normalization
    EoMrhs_lambda = sympy.lambdify([*states_sym[:n_dof], *states_sym[n_dof:], *states_dot_sym[n_dof:]], EoMrhs_expr, 'jax')

    def compute_EoMrhs(n_dof, EoMrhs_lambda, X, Xdot):
        EoMrhs = EoMrhs_lambda(*X[:n_dof], *X[n_dof:], *Xdot[n_dof:])
        return EoMrhs

    compute_batch_EoMrhs = vmap(compute_EoMrhs, in_axes=(None, None, 0, 0), out_axes=2)
    EoMrhs = compute_batch_EoMrhs(n_dof, EoMrhs_lambda, X, Xdot)
    EoMrhs = torch.from_numpy(np.asarray(EoMrhs).copy())
    EoMrhs = torch.flatten(EoMrhs.permute(0,2,1), end_dim=1)

    # Normalize EoM basis functions
    norm_factor = (1/(EoMrhs.shape[0]))*torch.sum(torch.abs(EoMrhs), 0)
    for i in range(norm_factor.shape[0]):
        if norm_factor[i] == 0:
            norm_factor[i] = 1
    EoMrhs_bar_expr = sympy.Matrix(np.asarray(EoMrhs_expr, dtype=object) / np.asarray(norm_factor))

    # Least-square regression
    EoMrhs_bar_lambda = sympy.lambdify([*states_sym[:n_dof], *states_sym[n_dof:], *states_dot_sym[n_dof:]], EoMrhs_bar_expr, 'jax')
    def compute_EoMrhs_bar(n_dof, EoMrhs_bar_lambda, X, Xdot):
        EoMrhs_bar = EoMrhs_bar_lambda(*X[:n_dof], *X[n_dof:], *Xdot[n_dof:])
        return EoMrhs_bar

    compute_batch_EoMrhs_bar = vmap(compute_EoMrhs_bar, in_axes=(None, None, 0, 0), out_axes=2)
    EoMrhs_bar = compute_batch_EoMrhs_bar(n_dof, EoMrhs_bar_lambda, X, Xdot)
    EoMrhs_bar = torch.from_numpy(np.asarray(EoMrhs_bar).copy())
    EoMrhs_bar = torch.flatten(EoMrhs_bar.permute(0,2,1), end_dim=1)

    coeffs_after_norm = torch.linalg.inv(EoMrhs_bar.T @ EoMrhs_bar) @ EoMrhs_bar.T @ (torch.flatten(torch.from_numpy(Tau).T).reshape(-1,1))
    coeffs_before_norm = coeffs_after_norm[:,0] / norm_factor

    # Obtain the inverse dynamics EoM (after multiplying by coefficients)
    inverse_dynamics_expr = EoMrhs_bar_expr @ sympy.Matrix(coeffs_after_norm.detach().cpu().numpy())
    inverse_dynamics_expr_lambda = sympy.lambdify([*states_sym[:n_dof], *states_sym[n_dof:], *states_dot_sym[n_dof:]], inverse_dynamics_expr, 'jax')

    #### Training loss ####
    def loss(pred, targ):
        loss = torch.mean((targ - pred)**2) 
        return loss 

    def compute_Tau(inverse_dynamics_expr_lambda, X, Xdot):
        tau = inverse_dynamics_expr_lambda(*X[:n_dof], *X[n_dof:], *Xdot[n_dof:])[:,0]
        return tau

    compute_batch_tau = vmap(compute_Tau, in_axes=(None, 0, 0), out_axes=0)
    tau_pred = compute_batch_tau(inverse_dynamics_expr_lambda, X, Xdot)
    lossval = loss(torch.from_numpy(np.asarray(tau_pred).copy()), torch.from_numpy(Tau))
    print('\nTraining loss:\n')
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

    ### Compute the p-norm of coefficients associated with each strain
    p = 1
    coeffs_after_norm = coeffs_after_norm.detach().cpu().numpy()
    norm_coefficients = np.power( (mapping_matrix @ np.power(np.abs(coeffs_after_norm), p)), 1./p )[:,0]

    # Neglect strains for which the norm of coefficients < threshold
    neglect_strain_index = np.nonzero((norm_coefficients < threshold))[0]
    if np.any(neglect_strain_index) == False: # all norms are above the threshold
        convergence = True
    else:
        Lagr_expr = list(
            sympy.Matrix(Lagr_expr).subs([
                (states_sym[neglect_strain_index[0]], 0),
                (states_sym[neglect_strain_index[0]+n_dof], 0)
            ])
        )

        Lagr_expr = list(set(Lagr_expr))

        states_sym = [ele for idx, ele in enumerate(states_sym) if ((idx != neglect_strain_index[0]) and (idx != neglect_strain_index[0]+n_dof))]
        states_dot_sym = [ele for idx, ele in enumerate(states_dot_sym) if ((idx != neglect_strain_index[0]) and (idx != neglect_strain_index[0]+n_dof))]

        states = [ele for idx, ele in enumerate(states) if ((idx != neglect_strain_index[0]) and (idx != neglect_strain_index[0]+n_dof))]
        states_dot = [ele for idx, ele in enumerate(states_dot) if ((idx != neglect_strain_index[0]) and (idx != neglect_strain_index[0]+n_dof))]

        # Delete some strains
        X = np.delete(X, [neglect_strain_index[0], neglect_strain_index[0]+n_dof], 1)
        Xdot = np.delete(Xdot, [neglect_strain_index[0], neglect_strain_index[0]+n_dof], 1)
        Tau = np.delete(Tau, neglect_strain_index[0], 1)
        X_val = np.delete(X_val, [neglect_strain_index[0], neglect_strain_index[0]+n_dof], 1)
        Xdot_val = np.delete(Xdot_val, [neglect_strain_index[0], neglect_strain_index[0]+n_dof], 1)
        Tau_val = np.delete(Tau_val, neglect_strain_index[0], 1)



# ------------------- Validation ---------------------------

# Obtain the terms of the Euler-Lagrange EoM
def getEOMterms(xi_Lcpu, phi_q, phi_qdot2, phi_qdotq):

    delta_expr = sympy.Matrix(phi_q) @ sympy.Matrix(xi_Lcpu)
    eta_expr = (sympy.Matrix(phi_qdotq.reshape(n_dof*n_dof, -1)) @ sympy.Matrix(xi_Lcpu)).reshape(n_dof, n_dof)
    zeta_expr = (sympy.Matrix(phi_qdot2.reshape(n_dof*n_dof, -1)) @ sympy.Matrix(xi_Lcpu)).reshape(n_dof, n_dof)

    delta_expr_lambda = sympy.lambdify([*states_sym[:n_dof], *states_sym[n_dof:]], delta_expr, 'jax')
    eta_expr_lambda = sympy.lambdify([*states_sym[:n_dof], *states_sym[n_dof:]], eta_expr, 'jax')
    zeta_expr_lambda = sympy.lambdify([*states_sym[:n_dof]], zeta_expr, 'jax')

    return delta_expr_lambda, eta_expr_lambda, zeta_expr_lambda

delta_expr_lambda, eta_expr_lambda, zeta_expr_lambda = getEOMterms(coeffs_before_norm.detach().cpu().numpy(), phi_q, phi_qdot2, phi_qdotq)

def generate_data(func, time, init_values, Tau):
    # sol = solve_ivp(func,[time[0],time[-1]],init_values,t_eval=time,method='RK45',rtol=1e-10,atol=1e-10)
    dt = time[1]-time[0]
    sol_list = []
    sol_list.append(init_values)

    for count, t in enumerate(time[::10]):
        tau = Tau[count*10, :]
        
        if t==0:
            sol = diffeqsolve(
                ODETerm(func),
                solver=Tsit5(),
                t0=time[0],
                t1=time[::10][1],
                dt0=dt,
                y0=init_values,
                args=tau,
                max_steps=None,
                saveat=SaveAt(ts=jnp.arange(0.0, time[::10][1]+dt, dt)),
            )
        else:
            sol = diffeqsolve(
                ODETerm(func),
                solver=Tsit5(),
                t0=time[0],
                t1=time[::100][1],
                dt0=dt,
                y0=sol_list[-1][-1],
                args=tau,
                max_steps=None,
                saveat=SaveAt(ts=jnp.arange(0.0, time[::10][1]+dt, dt)),
            )
        
        sol_list.append(sol.ys[1:,:])
    
    sol_array = np.vstack(sol_list)
    sol_array = sol_array[:-1,:]

    acc_list = []
    for i in range(sol_array.shape[0]):
        acc_list.append(func(time[i], sol_array[i,:], (Tau[i,:])))
        # print(i)
    acc_array = np.asarray(acc_list)

    # acc_gradient = np.gradient(sol_array[:, (sol_array.shape[1] // 2):], dt, axis=0)
    xdot = np.concatenate((sol_array[:, (sol_array.shape[1] // 2):], acc_array[:,(acc_array.shape[1] // 2):]), axis=1)

    return sol_array, xdot
    # first output is X array in [x,x_dot] format
    # second output is X_dot array in [x_dot,x_doubledot] format

@jit
def doublependulum(t,x,args):
    x_ = x[:x.shape[0]//2]
    x_t = x[x.shape[0]//2:]
    tau = args

    x_tt = jnp.linalg.inv(zeta_expr_lambda(*x_).T) @ (tau + delta_expr_lambda(*x_, *x_t)[:,0] - eta_expr_lambda(*x_, *x_t).T @ x_t)
    return jnp.concatenate([x_t, x_tt])

## Validation results ##
# true results
q_tt_true = (Xdot_val[:,n_dof:].T).copy()
q_t_true = (Xdot_val[:,:n_dof].T).copy()
q_true = (X_val[:,:n_dof].T).copy()
tau_true = (Tau_val.T).copy()

# prediction results
dt = 0.01  # time step
time_ = jnp.arange(0.0, 5.0, dt)
y_0 = X_val[0,:]
Xpred, Xdotpred = generate_data(doublependulum, time_, y_0, Tau_val)

q_tt_pred = Xdotpred[:,n_dof:].T
q_t_pred = Xdotpred[:,:n_dof].T
q_pred = Xpred[:,:n_dof].T

#### Validation loss ####
tau_pred = compute_batch_tau(inverse_dynamics_expr_lambda, X_val, Xdot_val)
lossval = loss(torch.from_numpy(np.asarray(tau_pred.T).copy()), torch.from_numpy(tau_true))
print('Validation loss:\n')
print(lossval)

fig, ax = plt.subplots(n_dof,1)
if n_dof == 1:
    ax.plot(torch.from_numpy(tau_true)[0,:], label='True Model')
    ax.plot(tau_pred[:,0], 'r--',label='Predicted Model')
    ax.set_ylabel('$Tau$')
    ax.grid(True)
else:
    for i in range(n_dof):
        ax[i].plot(torch.from_numpy(tau_true)[i,:], label='True Model')
        ax[i].plot(tau_pred[:,i], 'r--',label='Predicted Model')
        ax[i].set_ylabel('$Tau$')
        ax[i].grid(True)
plt.show()

## Plotting
t = time_

for i in range(n_dof):
    fig, ax = plt.subplots(3,1)

    ax[0].plot(t, q_tt_true[i,:], label='True Data')
    ax[0].plot(t, q_tt_pred[i,:], 'r--',label='Predicted Model')
    ax[0].set_ylabel('$\ddot{q}$')
    ax[0].set_xlim([0,0.5])
    ax[0].grid(True)

    ax[1].plot(t, q_t_true[i,:], label='True Data')
    ax[1].plot(t, q_t_pred[i,:], 'r--',label='Predicted Model')
    ax[1].set_ylabel('$\dot{q}$')
    ax[1].set_xlim([0,0.5])
    ax[1].grid(True)

    ax[2].plot(t, q_true[i,:], label='True Data')
    ax[2].plot(t, q_pred[i,:], 'r--',label='Predicted Model')
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('$q$')
    ax[2].set_xlim([0,0.5])
    ax[2].grid(True)

    Line, Label = ax[0].get_legend_handles_labels()
    fig.legend(Line, Label, loc='upper right')
    fig.suptitle('Simulation results xL-SINDY - Double Pendulum')

    fig.tight_layout()
    plt.show()