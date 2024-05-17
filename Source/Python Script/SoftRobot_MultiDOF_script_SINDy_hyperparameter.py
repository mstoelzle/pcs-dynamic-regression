#%%
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
import optuna
from optuna.trial import TrialState
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

compute_stiffness_matrix_for_all_segments_fn = vmap(
        compute_planar_stiffness_matrix, in_axes=(0, 0, 0, 0), out_axes=0
    )

# rootdir = "./Source/Soft Robot/ns-1_dof-3_random_actuation/"  
rootdir = "./Source/Soft Robot/ns-1_bending/"
# rootdir = "./Source/Soft Robot/ns-2_dof-3_random_actuation/"
# rootdir = "./Source/Soft Robot/ns-2_bsab/"
noiselevel = 0

# Load dataset
X_all = np.load(rootdir + "X.npy")
Xdot_all = np.load(rootdir + "Xdot.npy")
Tau_all = np.load(rootdir + "Tau.npy")

# Stack variables (from all initial conditions)
# X = (X_all[:-1])[:,10:,:]
# Xdot = (Xdot_all[:-1])[:,10:,:]
# Tau = (Tau_all[:-1])[:,10:,:]
X = (X_all[:-1])
Xdot = (Xdot_all[:-1])
Tau = (Tau_all[:-1])

X = np.vstack(X)
Xdot = np.vstack(Xdot)
Tau = np.vstack(Tau)
X_val = np.array(X_all[-1])
Xdot_val = np.array(Xdot_all[-1])
Tau_val = np.array(Tau_all[-1])

# Delete some strains
# X = np.delete(X, [1,4], 1)
# Xdot = np.delete(Xdot, [1,4], 1)
# Tau = np.delete(Tau, 1, 1)
# X_val = np.delete(X_val, [1,4], 1)
# Xdot_val = np.delete(Xdot_val, [1,4], 1)
# Tau_val = np.delete(Tau_val, 1, 1)

# # Add dummy strains
# num_dummy_strains = 1
# q_dummy = np.zeros((X.shape[0] + X_val.shape[0], num_dummy_strains))
# q_t_dummy = np.zeros((X.shape[0] + X_val.shape[0], num_dummy_strains))
# q_tt_dummy = np.zeros((X.shape[0] + X_val.shape[0], num_dummy_strains))
# for i in range(num_dummy_strains):
#     noise = np.random.normal(loc=0, scale=1e-2, size=q_dummy.shape[0])
#     # q_dummy[:,i] = q_dummy[:,i] + noise
#     # q_t_dummy[:,i] = 5e0*np.ones(q_t_dummy.shape[0])
#     # q_tt_dummy[:,i] = 1e-1*np.ones(q_tt_dummy.shape[0])
#     q_dummy[:,i] = savgol_filter(q_dummy[:,i] + noise, 1000, 3)
#     q_t_dummy[:,i] = savgol_filter(q_dummy[:,i], 1000, 3, deriv=1, delta=1e-4)
#     q_tt_dummy[:,i] = savgol_filter(q_dummy[:,i], 1000, 3, deriv=2, delta=1e-4)

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

# X = np.insert(X, [1,3], np.concatenate((q_dummy[:X.shape[0],:], q_t_dummy[:X.shape[0],:]), axis=1), axis=1)
# X_val = np.insert(X_val, [1,3], np.concatenate((q_dummy[X.shape[0]:,:], q_t_dummy[X.shape[0]:,:]), axis=1), axis=1)
# Xdot = np.insert(Xdot, [1,3], np.concatenate((q_t_dummy[:X.shape[0],:], q_tt_dummy[:X.shape[0],:]), axis=1), axis=1)
# Xdot_val = np.insert(Xdot_val, [1,3], np.concatenate((q_t_dummy[X.shape[0]:,:], q_tt_dummy[X.shape[0]:,:]), axis=1), axis=1)
# Tau = np.insert(Tau, [1], np.zeros((Tau.shape[0], 1)), axis=1)
# Tau_val = np.insert(Tau_val, [1], np.zeros((Tau_val.shape[0], 1)), axis=1)

####################################################################
#### Soft manipulator parameters - change based on the use case ####
num_segments = 1
strain_selector = np.array([True, False, False]) # bending, shear and axial
string_strains = ['Bending']

bending_map = [] # from the list of active states says which ones are bending state and which ones are not
for i in range(len(strain_selector)):
    if strain_selector[i]==True:
        if i%3==0:
            bending_map.append(True)
        else:
            bending_map.append(False)
bending_map = np.asarray(bending_map)

n_dof = np.sum(strain_selector)
epsilon_bend = 1e0
params = {
    "th0": jnp.array(0.0),  # initial orientation angle [rad]
    "l": 1e-1 * jnp.ones((num_segments,)),
    "r": 2e-2 * jnp.ones((num_segments,)),
    "rho": 1070 * jnp.ones((num_segments,)),
    "g": jnp.array([0.0, 9.81]), 
    # "E": 1e2 * jnp.ones((num_segments,)),  # Elastic modulus [Pa] # for elongation
    "E": 1e3 * jnp.ones((num_segments,)),  # Elastic modulus [Pa] # for bending
    "G": 1e3 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
    # "E": 1e4 * jnp.ones((num_segments,)),  # Elastic modulus [Pa] # previous values
    # "G": 1e3 * jnp.ones((num_segments,)),  # Shear modulus [Pa] # previous values
    # "D": 5e-6 * jnp.diag(jnp.array([3e0, 1e3, 1e3])),
    "D": 5e-6 * jnp.diag(jnp.array([3e0, 1e3, 1e3])),
}
params["A"] = jnp.pi * params["r"] ** 2
params["Ib"] = params["A"]**2 / (4 * jnp.pi)
# stiffness matrix of shape (num_segments, 3, 3)
S = compute_stiffness_matrix_for_all_segments_fn(params["A"], params["Ib"], params["E"], params["G"])
K = blk_diag(S)
params["K"] = np.array(K)

B_xi = compute_strain_basis(strain_selector)
xi_eq = jnp.zeros((3*num_segments,))
# by default, set the axial rest strain (local y-axis) along the entire rod to 1.0
rest_strain_reshaped = xi_eq.reshape((-1, 3))
rest_strain_reshaped = rest_strain_reshaped.at[:, -1].set(1.0)
xi_eq = rest_strain_reshaped.flatten()
####################################################################

# Remove samples where bending is smaller than a certain threshold => allows better learning
bending_indices = [i for i in range(len(bending_map)) if bending_map[i]==True]
if bending_indices != []:
    mask = True
    for idx in bending_indices:
        mask = mask & (np.abs(X[:,idx]) > 10)
    X = X[mask]
    Xdot = Xdot[mask]
    Tau = Tau[mask]

# adding noise
mu, sigma = 0, noiselevel
noise = np.random.normal(mu, sigma, X.shape[0])
for i in range(X.shape[1]):
    X[:,i] = X[:,i] + noise
    Xdot[:,i] = Xdot[:,i] + noise

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

#Turn from sympy to str
states_sym = states
states_dot_sym = states_dot
states_epsed_sym = states_epsed
states = list(str(descr) for descr in states)
states_dot = list(str(descr) for descr in states_dot)
states_epsed = list(str(descr) for descr in states_epsed)

########################################################
############ Create list of basis functions ############
expr_filepath = f"./Source/Soft Robot/symbolic_expressions/planar_pcs_ns-{num_segments}.dill"
sym_exps = dill.load(open(expr_filepath, 'rb'))

def B_decomp(expr, xi_sym):
    """
    A function dedicated to decompose the mass matrix entries B[i,j] into a 
    list of basis functions and respective coefficients.

    #Params:
    expr                    : symbolic expression of the mass matrix entry
    states_sym              : tuple of the symbolic variables which will be used in the algorithm. Format is (x,x_t)

    #Return:
    coeffs                  : list of coefficients for the basis functions
    monoms                  : list of basis functions
    """
    symbols = list(ordered(list(expr.free_symbols)))

    # replace manipulator parameters by their actual values / variables
    for seg in range(num_segments):
        # find if there is variable 'l' in the symbols
        l_idx = [i for i, j in enumerate(symbols) if str(j)==('l'+str(seg+1))]
        if l_idx != []: # if exists, replace for the value
            expr = expr.subs(symbols[l_idx[0]], params['l'][seg])

        # find if there is variable 'r' in the symbols
        r_idx = [i for i, j in enumerate(symbols) if str(j)==('r'+str(seg+1))]
        if r_idx != []: # if exists, replace for the value
            expr = expr.subs(symbols[r_idx[0]], params['r'][seg])

        # find if there is variable 'rho' in the symbols
        rho_idx = [i for i, j in enumerate(symbols) if str(j)==('rho'+str(seg+1))]
        if rho_idx != []: # if exists, replace for the value
            expr = expr.subs(symbols[rho_idx[0]], params['rho'][seg])

        # find if there is bending variable in the symbols
        bend_idx = [i for i, j in enumerate(symbols) if str(j)==('xi'+str(1+3*seg))]
        if bend_idx != []: # if exists, replace for the value
            expr = expr.subs(symbols[bend_idx[0]], xi_sym[3*seg])

        # find if there is shear variable in the symbols
        shear_idx = [i for i, j in enumerate(symbols) if str(j)==('xi'+str(1+3*seg+1))]
        if shear_idx != []: # if exists, replace for the value
            expr = expr.subs(symbols[shear_idx[0]], xi_sym[3*seg+1])

        # find if there is axial variable in the symbols
        axial_idx = [i for i, j in enumerate(symbols) if str(j)==('xi'+str(1+3*seg+2))]
        if axial_idx != []: # if exists, replace for the value
            expr = expr.subs(symbols[axial_idx[0]], xi_sym[3*seg+2])

    # separate coefficients and basis functions
    p = expr.as_poly(domain='RR[pi]')
    if p==None: # p is only a constant
        coeffs = [expr]
        monoms = [1]
    else:
        coeffs = p.coeffs()
        monoms = [sympy.prod(x**k for x, k in zip(p.gens, mon)) for mon in p.monoms()]
    return coeffs, monoms

def U_decomp(expr, xi_sym):
    """
    A function dedicated to decompose the gravitational potential expression U
    into a list of basis functions and respective coefficients.

    #Params:
    expr                    : symbolic expression of the mass matrix entry
    states_sym              : tuple of the symbolic variables which will be used in the algorithm. Format is (x,x_t)

    #Return:
    coeffs                  : list of coefficients for the basis functions
    monoms                  : list of basis functions
    """
    symbols = list(ordered(list(expr.free_symbols)))

    # replace manipulator parameters by their actual values / variables

    # find if there are variables 'g1' and 'g2' in the symbols
    for k in range(2):
        g_idx = [i for i, j in enumerate(symbols) if str(j)==('g'+str(k+1))]
        if g_idx != []: # if exists, replace for the value
            expr = expr.subs(symbols[g_idx[0]], params['g'][k])
    
    # find if there is variable 'th0' in the symbols
    th0_idx = [i for i, j in enumerate(symbols) if str(j)==('th0')]
    if th0_idx != []: # if exists, replace for the value
        expr = expr.subs(symbols[th0_idx[0]], params['th0'])

    for seg in range(num_segments):
        # find if there is variable 'l' in the symbols
        l_idx = [i for i, j in enumerate(symbols) if str(j)==('l'+str(seg+1))]
        if l_idx != []: # if exists, replace for the value
            expr = expr.subs(symbols[l_idx[0]], params['l'][seg])

        # find if there is variable 'r' in the symbols
        r_idx = [i for i, j in enumerate(symbols) if str(j)==('r'+str(seg+1))]
        if r_idx != []: # if exists, replace for the value
            expr = expr.subs(symbols[r_idx[0]], params['r'][seg])

        # find if there is variable 'rho' in the symbols
        rho_idx = [i for i, j in enumerate(symbols) if str(j)==('rho'+str(seg+1))]
        if rho_idx != []: # if exists, replace for the value
            expr = expr.subs(symbols[rho_idx[0]], params['rho'][seg])

        # find if there is bending variable in the symbols
        bend_idx = [i for i, j in enumerate(symbols) if str(j)==('xi'+str(1+3*seg))]
        if bend_idx != []: # if exists, replace for the value
            expr = expr.subs(symbols[bend_idx[0]], xi_sym[3*seg])

        # find if there is shear variable in the symbols
        shear_idx = [i for i, j in enumerate(symbols) if str(j)==('xi'+str(1+3*seg+1))]
        if shear_idx != []: # if exists, replace for the value
            expr = expr.subs(symbols[shear_idx[0]], xi_sym[3*seg+1])

        # find if there is axial variable in the symbols
        axial_idx = [i for i, j in enumerate(symbols) if str(j)==('xi'+str(1+3*seg+2))]
        if axial_idx != []: # if exists, replace for the value
            expr = expr.subs(symbols[axial_idx[0]], xi_sym[3*seg+2])

    # separate coefficients and basis functions
    p = expr.as_poly(domain='RR[pi]')
    coeffs = p.coeffs()
    monoms = [sympy.prod(x**k for x, k in zip(p.gens, mon)) for mon in p.monoms()]
    return coeffs, monoms

def constructLagrangianExpression(sym_exps, states_sym):
    true_coeffs = []
    expr = []

    xi_sym = sympy.Matrix(xi_eq) + sympy.Matrix(B_xi)*sympy.Matrix(states_sym[:len(states_sym)//2])
    for i in range(num_segments):
        if strain_selector[3*i] == False:
            xi_sym[3*i] = epsilon_bend

    # Extract the basis functions from the mass matrix 
    # Due to the symmetry of the mass matrix, it has (n**2+n)/2 independent entries. In the 
    # Lagrangian expression, each of these entries needs to be mutiplied by 1/2 and 
    # the corresponding q_dot**2
    B = sympy.Matrix(B_xi).T * sym_exps['exps']['B'] * sympy.Matrix(B_xi)

    for i in range(B.shape[1]):
        for j in range(i, B.shape[0]):
            B_entry = B[j,i]
            if B_entry!=0:
                coeffs, monoms = B_decomp(B_entry, xi_sym)
                monoms = [x*states_sym[n_dof+j]*states_sym[n_dof+i] for x in monoms]
                if i==j:
                    coeffs = [0.5*x for x in coeffs]
                
                true_coeffs.append(coeffs)
                expr.append(monoms)

    # Extract the basis functions from the gravitational potential
    U = sym_exps['exps']['U'][0,0]
    coeffs, monoms = U_decomp(U, xi_sym)
    true_coeffs.append(coeffs)
    expr.append(monoms)

    # Add the basis functions for the elastic potential
    K = np.array(B_xi.T) @ params['K'] @ np.array(B_xi)
    for i in range(n_dof):
        true_coeffs.append([-0.5*K[i,i]])
        expr.append([states_sym[i]**2])

    # Flatten out the lists
    true_coeffs = list(chain.from_iterable(true_coeffs))
    expr = list(chain.from_iterable(expr))

    true_coeffs_list = [true_coeffs[i].evalf() for i in range(len(true_coeffs)-n_dof)]
    for i in range(n_dof):
        true_coeffs_list.append(true_coeffs[-n_dof+i])
    true_coeffs = np.asarray(true_coeffs_list, dtype=np.float64)

    return expr, true_coeffs

expr, true_coeffs = constructLagrangianExpression(sym_exps, states_sym)

# In case there is independent term, remove it from both lists (Lagrangian is invariant to constants)
true_coeffs = np.asarray([ele for idx, ele in enumerate(true_coeffs) if list(expr[idx].free_symbols)!=[]])
expr = [ele for idx, ele in enumerate(expr) if list(expr[idx].free_symbols)!=[]]

# Normalize basis functions
expr_lambda = sympy.lambdify([*states_sym[:n_dof], *states_sym[n_dof:]], sympy.Matrix(expr), 'jax')
def evaluate_basis_fcns(n_dof, expr_lambda, X):
    basis_fcns = expr_lambda(*X[:n_dof], *X[n_dof:])
    return basis_fcns

compute_batch_basis_fcns = vmap(evaluate_basis_fcns, in_axes=(None, None, 0), out_axes=0)
basis_fcns = np.asarray(compute_batch_basis_fcns(n_dof, expr_lambda, X))[:,:,0]
mean_basis_fcns = np.abs(np.mean(basis_fcns, axis=0))
expr = [ele/(mean_basis_fcns[idx]) for idx, ele in enumerate(expr)]
# max_basis_fcns = np.max(basis_fcns, axis=0)
# min_basis_fcns = np.min(basis_fcns, axis=0)
# expr = [ele/(max_basis_fcns[idx]-min_basis_fcns[idx]) for idx, ele in enumerate(expr)]
#####################################################################################

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

X_epsed = np.zeros((X.shape[0], n_dof))
for i in range(n_dof):
    if bending_map[i] == True:
        q_epsed = apply_eps_to_bend_strains(X[:,i], epsilon_bend)
    else:
        q_epsed = X[:,i]
    
    X_epsed[:,i] = q_epsed

# Compute symbolic tensors with expression present in the Euler-Lagrange equation
device = 'cpu'
phi_q, phi_qdot2, phi_qdotq = EulerLagrangeExpressionTensor(expr, states, states_epsed, states_epsed_sym)

phi_qdot2_expr = (sympy.Matrix(phi_qdot2.reshape(n_dof,-1)).T @ sympy.Matrix(states_dot_sym[n_dof:])).reshape(n_dof, phi_q.shape[1])
phi_qdotq_expr = (sympy.Matrix(phi_qdotq.reshape(n_dof,-1)).T @ sympy.Matrix(states_dot_sym[:n_dof])).reshape(n_dof, phi_q.shape[1])
phi_q_expr = sympy.Matrix(phi_q)

Zeta, _, _ = LagrangianLibraryTensor(X, Xdot, X_epsed, states, states_dot, states_epsed, phi_q, phi_qdot2, phi_qdotq)

def compute_X_ls(n_dof, X_ls_lambda, X, Xdot, X_epsed):
    X_ls = X_ls_lambda(*X[:n_dof], *X[n_dof:], *Xdot[n_dof:], *X_epsed[:])
    return X_ls

X_ls_lambda = sympy.lambdify([*states_sym[:n_dof], *states_sym[n_dof:], *states_dot_sym[n_dof:], *states_epsed_sym], phi_qdot2_expr + phi_qdotq_expr - phi_q_expr, 'jax')
compute_batch_X_ls = vmap(compute_X_ls, in_axes=(None, None, 0, 0, 0), out_axes=2)

##### Get the true coefficients #####
X_ls = compute_batch_X_ls(n_dof, X_ls_lambda, X, Xdot, X_epsed)
X_ls = torch.from_numpy(np.asarray(X_ls).copy())
q_t = torch.from_numpy(Xdot).to(device)[:, :Xdot.shape[1]//2].T
X_ls = torch.flatten(X_ls.permute(0,2,1), end_dim=1)
# Add basis function for velocity to also find the damping constants
if n_dof == 1:
    X_ls = torch.cat((X_ls, q_t.T), 1)
else:
    list_vel = [torch.flatten(q_t[i,:]).reshape(-1,1) for i in range(n_dof)] # block diagonal matrix 
    X_ls = torch.cat((X_ls, torch.block_diag(*list_vel)), 1)
# Apply least squares normal equation
coef_ls = torch.linalg.inv(X_ls.T @ X_ls) @ X_ls.T @ (torch.flatten(torch.from_numpy(Tau).to(device).T).reshape(-1,1))
xi_L_true = coef_ls[:-n_dof,0]
D_true = torch.diag(coef_ls[-n_dof:,0])
#####################################

X_ls = compute_batch_X_ls(n_dof, X_ls_lambda, X, Xdot, X_epsed)
X_ls = torch.from_numpy(np.asarray(X_ls).copy())

############## SINDy ##############
###################################
def loss(pred, targ):
    loss = torch.mean((targ - pred)**2) 
    return loss 

class xLSINDY(torch.nn.Module):
    def __init__(self, xi_L, D):
        
        super().__init__()
        self.coef = torch.nn.Parameter(xi_L)
        self.D = torch.nn.Parameter(D) # learn the damping constant
        # self.D = D

    def forward(self, n_dof, x_ls, xdot, device):#, tau):
        
        q_t = torch.from_numpy(xdot[:, :n_dof].T).to(device)
        tau_pred = torch.einsum('ikl,k->il', x_ls, self.coef.to(device)) + torch.einsum('ij,il->jl', torch.diag(self.D), q_t)

        # tau_pred = ELforward(self.coef, zeta, eta, delta, x_t, device, self.D)
        # q_tt_pred = lagrangianforward(self.coef, zeta, eta, delta, x_t, device, tau)
        return tau_pred

def objective(trial):
    # Initialize coefficients and model
    torch.manual_seed(1)
    # xi_L = torch.ones(len(expr), dtype = torch.float64, device=device).data.uniform_(-0.01,0.01)
    # D = torch.ones(states_dim//2, dtype=torch.float64, device=device).data.uniform_(0,1e-3)
    # model = xLSINDY(xi_L, D)
    model = xLSINDY(xi_L_true, torch.diagonal(D_true, 0).to(device))

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float('lr', 1e-7, 1e-1, log=True)
    # lam = trial.suggest_float('lam', 0, 1)
    w = trial.suggest_float('w', 0, 5)
    lam = 0
    bs = trial.suggest_int('bs', 100, Xdot.shape[0])
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    if(torch.is_tensor(Tau)==False):
        tau = torch.from_numpy(Tau).to(device).float()

    for epoch in range(Epoch):
        
        tl = Xdot.shape[0]
        loss_list = []
        for i in range(tl//bs):
            
            xdot = Xdot[i*bs:(i+1)*bs,:]
            x_ls = X_ls[:,:,i*bs:(i+1)*bs]
            zeta = Zeta[:,:,:,i*bs:(i+1)*bs]
            n = xdot.shape[1]//2

            # If loss using torque
            tau_pred = model.forward(n, x_ls, xdot, device)
            tau_true = (tau[i*bs:(i+1)*bs,:].T).to(device)
            mse_loss = loss(tau_pred, tau_true)

            DL_qdot2 = torch.einsum('ijkl,k->ijl', zeta, model.coef.to(device))
            min_det = torch.min(torch.linalg.det(DL_qdot2.permute(2,0,1)))

            # If loss using acceleration
            # tau_true = (tau[i*bs:(i+1)*bs,:].T).to(device)
            # q_tt_pred = model.forward(zeta, eta, delta, x_t, device, tau_true)
            # q_tt_true = Xdot[i*bs:(i+1)*bs,n//2:].T
            # mse_loss = loss(q_tt_pred, q_tt_true)

            l1_norm = sum(
                p.abs().sum() for p in model.coef
            )
            lossval = mse_loss + lam * l1_norm + w * torch.nn.functional.relu(-min_det)

            optimizer.zero_grad()
            lossval.backward()
            optimizer.step()

            loss_list.append(lossval.item())
        
        lossitem = torch.tensor(loss_list).mean().item()

        trial.report(lossitem, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return lossitem
    

# Training parameters
Epoch = 250
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=150)

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))