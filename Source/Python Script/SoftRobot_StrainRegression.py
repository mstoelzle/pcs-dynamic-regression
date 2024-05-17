#%%
import numpy as np
import sys 
from sympy import symbols, simplify, derive_by_array, ordered, poly
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

compute_stiffness_matrix_for_all_segments_fn = vmap(
        compute_planar_stiffness_matrix, in_axes=(0, 0, 0, 0), out_axes=0
    )

# rootdir = "./Source/Soft Robot/ns-2_dof-3_random_actuation/"  
# rootdir = "./Source/Soft Robot/ns-1_bending_and_axial/"
rootdir = "./Source/Soft Robot/ns-1_dof-3_random_actuation/"
# rootdir = "./Source/Soft Robot/ns-2_bsab/"
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

# Delete some strains
# X = np.delete(X, [1,4], 1)
# Xdot = np.delete(Xdot, [1,4], 1)
# Tau = np.delete(Tau, 1, 1)
# X_val = np.delete(X_val, [1,4], 1)
# Xdot_val = np.delete(Xdot_val, [1,4], 1)
# Tau_val = np.delete(Tau_val, 1, 1)

# # Add dummy strains
# num_dummy_strains = 3
# q_dummy = np.zeros((X.shape[0] + X_val.shape[0], num_dummy_strains))
# q_t_dummy = np.zeros((X.shape[0] + X_val.shape[0], num_dummy_strains))
# q_tt_dummy = np.zeros((X.shape[0] + X_val.shape[0], num_dummy_strains))
# for i in range(num_dummy_strains):
#     noise = np.random.normal(loc=1e-3, scale=0, size=q_dummy.shape[0])
#     q_dummy[:,i] = q_dummy[:,i] + noise
#     q_t_dummy[:,i] = 1e-4*np.ones(q_t_dummy.shape[0])
#     q_tt_dummy[:,i] = 1e-5*np.ones(q_tt_dummy.shape[0])
#     # q_dummy[:,i] = savgol_filter(q_dummy[:,i] + noise, 1000, 3)
#     # q_t_dummy[:,i] = savgol_filter(q_dummy[:,i], 1000, 3, deriv=1, delta=1e-4)
#     # q_tt_dummy[:,i] = savgol_filter(q_dummy[:,i], 1000, 3, deriv=2, delta=1e-4)

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

# X = np.insert(X, [3,3,3,6,6,6], np.concatenate((q_dummy[:X.shape[0],:], q_t_dummy[:X.shape[0],:]), axis=1), axis=1)
# X_val = np.insert(X_val, [3,3,3,6,6,6], np.concatenate((q_dummy[X.shape[0]:,:], q_t_dummy[X.shape[0]:,:]), axis=1), axis=1)
# Xdot = np.insert(Xdot, [3,3,3,6,6,6], np.concatenate((q_t_dummy[:X.shape[0],:], q_tt_dummy[:X.shape[0],:]), axis=1), axis=1)
# Xdot_val = np.insert(Xdot_val, [3,3,3,6,6,6], np.concatenate((q_t_dummy[X.shape[0]:,:], q_tt_dummy[X.shape[0]:,:]), axis=1), axis=1)
# Tau = np.insert(Tau, [3,3,3], 1e-5*np.ones((Tau.shape[0], 3)), axis=1)
# Tau_val = np.insert(Tau_val, [3,3,3], 1e-5*np.ones((Tau_val.shape[0], 3)), axis=1)

####################################################################
#### Soft manipulator parameters - change based on the use case ####
num_segments = 1
strain_selector = np.array([True, True, False]) # bending, shear and axial
string_strains = ['Bending','Shear']

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
    "E": 1e4 * jnp.ones((num_segments,)),  # Elastic modulus [Pa] # for bending
    # "E": jnp.array([1e4, 1e4]),
    "G": 1e7 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
    "D": 5e-6 * jnp.diag(jnp.array([3e0, 1e3, 1e4])),
    # "D": 5e-6 * jnp.diag(jnp.array([3e1, 1e5, 1e4, 3e1, 1e3, 1e4])),
}
params["length"] = np.sum(params["l"])
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
        mask = mask & (np.abs(X[:,idx]) >= 0)
    X = X[mask]
    Xdot = Xdot[mask]
    Tau = Tau[mask]

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
states_sym = list(states)
states_dot_sym = list(states_dot)
states_epsed_sym = list(states_epsed)
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
    # p = expr.as_poly()
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

    kinetic_energy = HL.generateExpression(
        list(chain.from_iterable(true_coeffs)), list(chain.from_iterable(expr))
    )

    # Extract the basis functions from the gravitational potential
    U = sym_exps['exps']['U'][0,0]
    coeffs, monoms = U_decomp(U, xi_sym)
    true_coeffs.append(coeffs)
    expr.append(monoms)

    g_potential_energy = HL.generateExpression(
        np.asarray(coeffs), monoms
    )

    # Add the basis functions for the elastic potential
    coeffs_elastic_pot = []
    monoms_elastic_pot = []
    K = np.array(B_xi.T) @ params['K'] @ np.array(B_xi)
    for i in range(n_dof):
        true_coeffs.append([-0.5*K[i,i]])
        coeffs_elastic_pot.append(0.5*K[i,i])
        expr.append([states_sym[i]**2])
        monoms_elastic_pot.append(states_sym[i]**2)
    
    elastic_potential_energy = HL.generateExpression(
        coeffs_elastic_pot, monoms_elastic_pot
    )

    def len_symbol(e):
        return len(str(e))
    symbols = list(ordered(list(kinetic_energy.free_symbols), keys=len_symbol))
    for i in range(n_dof):
        kinetic_energy = kinetic_energy.subs([
            (symbols[i], states_epsed_sym[i]),
        ])

        g_potential_energy = g_potential_energy.subs([
            (symbols[i], states_epsed_sym[i]),
        ])

    potential_energy = g_potential_energy + elastic_potential_energy
    # Flatten out the lists
    true_coeffs = list(chain.from_iterable(true_coeffs))
    expr = list(chain.from_iterable(expr))

    true_coeffs_list = [true_coeffs[i].evalf() for i in range(len(true_coeffs)-n_dof)]
    for i in range(n_dof):
        true_coeffs_list.append(true_coeffs[-n_dof+i])
    true_coeffs = np.asarray(true_coeffs_list, dtype=np.float64)

    return expr, true_coeffs, kinetic_energy, potential_energy

Lagr_expr_symbolic_length, true_coeffs_before_norm, kinetic_energy, potential_energy = constructLagrangianExpression(sym_exps, states_sym)
# In case there is independent term, remove it from both lists (Lagrangian is invariant to constants)
true_coeffs_before_norm = np.asarray([ele for idx, ele in enumerate(true_coeffs_before_norm) if list(Lagr_expr_symbolic_length[idx].free_symbols)!=[]])
Lagr_expr_symbolic_length = [ele for idx, ele in enumerate(Lagr_expr_symbolic_length) if list(Lagr_expr_symbolic_length[idx].free_symbols)!=[]]

convergence = False

while convergence == False:
    n_dof = len(states_sym)//2

    symbols = list(ordered(list(sympy.Matrix(Lagr_expr_symbolic_length).free_symbols)))
    length_variables = [word for word in symbols if str(word)[0]=='l']
    print(length_variables)

    Lagr_expr = sympy.Matrix(Lagr_expr_symbolic_length)
    for l_var in length_variables:
        Lagr_expr = Lagr_expr.subs(l_var, params['length']/len(length_variables))
    Lagr_expr = list(Lagr_expr)

    # In case there is independent term, remove it (Lagrangian is invariant to constants)
    Lagr_expr_symbolic_length = [ele for idx, ele in enumerate(Lagr_expr_symbolic_length) if list(Lagr_expr_symbolic_length[idx].free_symbols)!=[]]
    Lagr_expr = [ele for idx, ele in enumerate(Lagr_expr) if list(Lagr_expr[idx].free_symbols)!=[]]

    # Compute the coefficient mapping matrix
    mapping_matrix = np.zeros((n_dof, len(Lagr_expr) + n_dof))
    for i in range(len(Lagr_expr) + n_dof):
        for q in range(n_dof):
            if i < len(Lagr_expr): # coefficients present in the lagrangian
                if (states_sym[q] in list(ordered(list(Lagr_expr[i].free_symbols)))) or (states_sym[q+n_dof] in list(ordered(list(Lagr_expr[i].free_symbols)))):
                    mapping_matrix[q,i] = 1
            else: # coefficients for the damping basis functions
                if (i - len(Lagr_expr)) == q:
                    mapping_matrix[q,i] = 1

    ########### From Lagrangian expr to EoM expr #############
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
    EoMrhs_lambda = sympy.lambdify([*states_sym[:n_dof], *states_sym[n_dof:], *states_dot_sym[n_dof:], *states_epsed_sym], EoMrhs_expr, 'jax')

    def compute_EoMrhs(n_dof, EoMrhs_lambda, X, Xdot, X_epsed):
        EoMrhs = EoMrhs_lambda(*X[:n_dof], *X[n_dof:], *Xdot[n_dof:], *X_epsed[:])
        return EoMrhs

    compute_batch_EoMrhs = vmap(compute_EoMrhs, in_axes=(None, None, 0, 0, 0), out_axes=2)
    EoMrhs = compute_batch_EoMrhs(n_dof, EoMrhs_lambda, X, Xdot, X_epsed)
    EoMrhs = torch.from_numpy(np.asarray(EoMrhs).copy())
    EoMrhs = torch.flatten(EoMrhs.permute(0,2,1), end_dim=1)

    # Normalize EoM basis functions
    norm_factor = (1/(EoMrhs.shape[0]))*torch.sum(torch.abs(EoMrhs), 0)
    # norm_factor = torch.ones(len(Lagr_expr) + n_dof)
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

    ### Compute the p-norm of coefficients associated with each strain
    p = 1
    coeffs_after_norm = coeffs_after_norm.detach().cpu().numpy()
    norm_coefficients = np.power( (mapping_matrix @ np.power(np.abs(coeffs_after_norm), p)), 1./p )[:,0]
    threshold = np.max(norm_coefficients)/10
    threshold = 0.1

    print('\nCoefficient norms:')
    print(norm_coefficients)
    print('(threshold is ' + str(threshold) + ')')

    # Neglect strains for which the norm of coefficients < threshold
    # neglect_strain_index = np.nonzero((norm_coefficients < threshold))[0]
    neglect_strain_index = np.array([0])
    if neglect_strain_index.size == 0: # all norms are above the threshold
        convergence = True
        print('\nResult:')
        print('No (more) strains will be deactivated.')
    else:
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

                Lagr_expr_symbolic_length = list(
                    sympy.Matrix(Lagr_expr_symbolic_length).subs([
                        (states_sym[index], 0),
                        (states_sym[index+n_dof], 0)
                    ])
                )

                kinetic_energy = kinetic_energy.subs([
                    (states_sym[index], 0),
                    (states_sym[index+n_dof], 0),
                    (states_epsed_sym[index], 0),
                ])

                potential_energy = potential_energy.subs([
                    (states_sym[index], 0),
                    (states_sym[index+n_dof], 0),
                    (states_epsed_sym[index], 0),
                ])
            else:
                Lagr_expr = list(
                    sympy.Matrix(Lagr_expr).subs([
                        (states_sym[index], epsilon_bend),
                        (states_sym[index+n_dof], 0)
                    ])
                )

                Lagr_expr_symbolic_length = list(
                    sympy.Matrix(Lagr_expr_symbolic_length).subs([
                        (states_sym[index], epsilon_bend),
                        (states_sym[index+n_dof], 0)
                    ])
                )

                kinetic_energy = kinetic_energy.subs([
                    (states_sym[index], epsilon_bend),
                    (states_sym[index+n_dof], 0),
                    (states_epsed_sym[index], epsilon_bend),
                ])

                potential_energy = potential_energy.subs([
                    (states_sym[index], epsilon_bend),
                    (states_sym[index+n_dof], 0),
                    (states_epsed_sym[index], epsilon_bend),
                ])

        new_Lagr_expr = []
        [new_Lagr_expr.append(x) for x in Lagr_expr if x not in new_Lagr_expr]
        Lagr_expr = new_Lagr_expr

        new_Lagr_expr_symbolic_length = []
        [new_Lagr_expr_symbolic_length.append(x) for x in Lagr_expr_symbolic_length if x not in new_Lagr_expr_symbolic_length]
        Lagr_expr_symbolic_length = new_Lagr_expr_symbolic_length

        states_sym = [ele for idx, ele in enumerate(states_sym) if ((idx != neglect_strain_index[0]) and (idx != neglect_strain_index[0]+n_dof))]
        states_dot_sym = [ele for idx, ele in enumerate(states_dot_sym) if ((idx != neglect_strain_index[0]) and (idx != neglect_strain_index[0]+n_dof))]

        states = [ele for idx, ele in enumerate(states) if ((idx != neglect_strain_index[0]) and (idx != neglect_strain_index[0]+n_dof))]
        states_dot = [ele for idx, ele in enumerate(states_dot) if ((idx != neglect_strain_index[0]) and (idx != neglect_strain_index[0]+n_dof))]

        states_epsed = [ele for idx, ele in enumerate(states_epsed) if ((idx != neglect_strain_index[0]) and (idx != neglect_strain_index[0]+n_dof))]
        states_epsed_sym = [ele for idx, ele in enumerate(states_epsed_sym) if ((idx != neglect_strain_index[0]) and (idx != neglect_strain_index[0]+n_dof))]

        string_strains = [ele for idx, ele in enumerate(string_strains) if (idx != neglect_strain_index[0])]
        bending_map = [ele for idx, ele in enumerate(bending_map) if (idx != neglect_strain_index[0])]

        # Delete neglected strains
        X = np.delete(X, [neglect_strain_index[0], neglect_strain_index[0]+n_dof], 1)
        Xdot = np.delete(Xdot, [neglect_strain_index[0], neglect_strain_index[0]+n_dof], 1)
        X_epsed = np.delete(X_epsed, neglect_strain_index[0], 1)
        Tau = np.delete(Tau, neglect_strain_index[0], 1)
        X_val = np.delete(X_val, [neglect_strain_index[0], neglect_strain_index[0]+n_dof], 1)
        Xdot_val = np.delete(Xdot_val, [neglect_strain_index[0], neglect_strain_index[0]+n_dof], 1)
        Tau_val = np.delete(Tau_val, neglect_strain_index[0], 1)


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

    for count, t in enumerate(time[::100]):
        tau = Tau[count*100, :]
        # tau = tau_unique[count]
        
        if t==0:
            sol = diffeqsolve(
                ODETerm(func),
                solver=Tsit5(),
                t0=time[0],
                t1=time[::100][1],
                dt0=dt,
                y0=init_values,
                args=(tau, D.detach().cpu().numpy()),
                max_steps=None,
                saveat=SaveAt(ts=jnp.arange(0.0, time[::100][1]+dt, dt)),
            )
        else:
            sol = diffeqsolve(
                ODETerm(func),
                solver=Tsit5(),
                t0=time[0],
                t1=time[::100][1],
                dt0=dt,
                y0=sol_list[-1][-1],
                args=(tau, D.detach().cpu().numpy()),
                max_steps=None,
                saveat=SaveAt(ts=jnp.arange(0.0, time[::100][1]+dt, dt)),
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
            q_epsed = apply_eps_to_bend_strains_jnp(x_[i], 1e0)
        else:
            q_epsed = x_[i]
        
        x_epsed_list.append(q_epsed)
    
    x_epsed = jnp.asarray(x_epsed_list)
    
    tau, D = args
    x_tt = jnp.linalg.inv(zeta_expr_lambda(*x_, *x_epsed).T) @ (tau - D @ x_t + delta_expr_lambda(*x_, *x_t, *x_epsed)[:,0] - eta_expr_lambda(*x_, *x_t, *x_epsed).T @ x_t)
    return jnp.concatenate([x_t, x_tt])

## Validation results ##
# true results
q_tt_true = (Xdot_val[:,n_dof:].T).copy()
q_t_true = (Xdot_val[:,:n_dof].T).copy()
q_true = (X_val[:,:n_dof].T).copy()
tau_true = (Tau_val.T).copy()

# prediction results
dt = 1e-4  # time step
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
        q_epsed = apply_eps_to_bend_strains(X_val[:,i], 1e0)
    else:
        q_epsed = X_val[:,i]
    
    X_epsed_val[:,i] = q_epsed

tau_pred = compute_batch_tau(inverse_dynamics_expr_lambda, X_val, Xdot_val, X_epsed_val)
lossval = loss(torch.from_numpy(np.asarray(tau_pred.T).copy()), torch.from_numpy(tau_true))
print('\nValidation loss:')
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
    fig.suptitle('Simulation results xL-SINDY - ' + str(num_segments) + ' segments ' + str(n_dof) + ' DOF => ' + string_strains[i])

    fig.tight_layout()
    plt.show()

kinetic_energy_lambda = sympy.lambdify([*states_sym[:n_dof], *states_sym[n_dof:], *states_epsed_sym], kinetic_energy, 'jax')
potential_energy_lambda = sympy.lambdify([*states_sym[:n_dof], *states_sym[n_dof:], *states_epsed_sym], potential_energy, 'jax')
def compute_energy(n_dof, lambda_function, X, Xdot, X_epsed):
    energy = lambda_function(*X[:n_dof], *X[n_dof:], *X_epsed[:])
    return energy

compute_batch_energy = vmap(compute_energy, in_axes=(None, None, 0, 0, 0), out_axes=0)
kinetic_energy = compute_batch_energy(n_dof, kinetic_energy_lambda, X_val, Xdot_val, X_epsed_val)
potential_energy = compute_batch_energy(n_dof, potential_energy_lambda, X_val, Xdot_val, X_epsed_val)

fig, ax = plt.subplots(1,1)
ax.plot(kinetic_energy, label='Kinetic energy')
ax.plot(potential_energy, label='Potential energy')
ax.set_ylabel('Energy')
Line, Label = ax.get_legend_handles_labels()
fig.legend(Line, Label, loc='upper right')
ax.grid(True)
plt.show()