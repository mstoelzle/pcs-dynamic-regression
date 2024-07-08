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
from utils.basis_functions_utils import constructLagrangianExpression

compute_stiffness_matrix_for_all_segments_fn = vmap(
        compute_planar_stiffness_matrix, in_axes=(0, 0, 0, 0), out_axes=0
    )

####################################################################
#### Soft manipulator parameters - change based on the use case ####
num_segments = 1
strain_selector = np.array([True, True, True]) # bending, shear and axial
epsilon_bend = 5e-2

params = {
    "th0": jnp.array(0.0),  # initial orientation angle [rad]
    "l": 1e-1 * jnp.ones((num_segments,)),
    "r": 2e-2 * jnp.ones((num_segments,)),
    "rho": 1070 * jnp.ones((num_segments,)),
    "g": jnp.array([0.0, 9.81]), 
    "E": 1e4 * jnp.ones((num_segments,)),  # Elastic modulus [Pa] # for bending
    # "E": jnp.array([1e4, 1e4]),
    "G": 1e3 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
    "D": 5e-6 * jnp.diag(jnp.array([1e0, 1e4, 8e3])),
    # "D": 5e-6 * jnp.diag(jnp.array([3e1, 1e5, 1e4, 3e1, 1e3, 1e4])),
    "num_segments": num_segments
}
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

## Create the states nomenclature
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

## Create list of Lagrangian basis functions
expr_filepath = f"./Source/Soft Robot/symbolic_expressions/planar_pcs_ns-{num_segments}.dill"
sym_exps = dill.load(open(expr_filepath, 'rb'))
Lagr_expr, true_coeffs_before_norm, _, _ = constructLagrangianExpression(sym_exps, states_sym, states_epsed_sym, xi_eq, B_xi, strain_selector, params)
# In case there is independent term, remove it from both lists (Lagrangian is invariant to constants)
true_coeffs_before_norm = np.asarray([ele for idx, ele in enumerate(true_coeffs_before_norm) if list(Lagr_expr[idx].free_symbols)!=[]])
Lagr_expr = [ele for idx, ele in enumerate(Lagr_expr) if list(Lagr_expr[idx].free_symbols)!=[]]

## Get derivatives and double derivatives of basis functions (which appear in Euler-Lagrange equation)
phi_q, phi_qdot2, phi_qdotq = EulerLagrangeExpressionTensor(Lagr_expr, states, states_epsed_sym)

## Dictionary with expressions to save
expr_basis_fcns = {
    "true_coeffs_before_norm": true_coeffs_before_norm,
    "Lagr_expr": Lagr_expr,
    "phi_q": phi_q,
    "phi_qdot2": phi_qdot2,
    "phi_qdotq": phi_qdotq
}

basis_fcns_filepath = f"./Source/Soft Robot/symbolic_basis_functions/planar_pcs_ns-{num_segments}.dill"
with open(str(basis_fcns_filepath), "wb") as f:
    dill.dump(expr_basis_fcns, f)

