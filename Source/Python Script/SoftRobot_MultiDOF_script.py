#%%
import numpy as np
import sys 
from sympy import symbols, simplify, derive_by_array
from scipy.integrate import solve_ivp
from xLSINDy import *
from sympy.physics.mechanics import *
from sympy import *
from sympy.solvers import solve
import sympy
import torch
torch.set_printoptions(precision=10)
import math
from math import pi
sys.path.append(r'../../../HLsearch/')
import HLsearch as HL
import time
from itertools import chain
import dill
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, Dopri5, Euler, ODETerm, SaveAt, Tsit5, PIDController
from jax import numpy as jnp
from jax import config, lax, vmap
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

from utils.math_utils import blk_diag
from utils.utils import compute_planar_stiffness_matrix, compute_strain_basis

compute_stiffness_matrix_for_all_segments_fn = vmap(
        compute_planar_stiffness_matrix, in_axes=(0, 0, 0, 0), out_axes=0
    )

# rootdir = "./Soft Robot/ns-1_dof-1_zero_actuation/"
# rootdir = "./Source/Soft Robot/ns-1_dof-1_tau_act/"
# rootdir = "./Source/Soft Robot/ns-1_bending_tau_act/"
# rootdir = "./Source/Soft Robot/ns-1_bending_zero_damping/"
# rootdir = "./Source/Soft Robot/ns-1_bending_random_actuation/"
# rootdir = "./Source/Soft Robot/ns-1_bending_random_actuation_larger_magnitude/"
# rootdir = "./Source/Soft Robot/ns-1_dof-3_random_actuation/"  
# rootdir = "./Source/Soft Robot/ns-1_shear_and_axial/"
rootdir = "./Source/Soft Robot/ns-2_dof-3_random_actuation/"
# rootdir = "./Source/Soft Robot/ns-2_bsab/"
# rootdir = "./Source/Soft Robot/check_if_error/"
noiselevel = 0

# Load dataset
X_all = np.load(rootdir + "X.npy")
Xdot_all = np.load(rootdir + "Xdot.npy")
Tau_all = np.load(rootdir + "Tau.npy")

# Select duration of dataset
X_list, Xdot_list, Tau_list = [], [], []
# X_val, Xdot_val, Tau_val = []
for i in range(len(X_all) - 1):
    X_list.append(X_all[i][:5000,:])
    # X_val.append(X_all[i][400:,:])

    Xdot_list.append(Xdot_all[i][:5000,:])
    # Xdot_val.append(Xdot_all[i][400:,:])

    Tau_list.append(Tau_all[i][:5000])
    # Tau_val.append(Tau_all[i][400:])

X_val = np.array(X_all[-1][:5000,:])
Xdot_val = np.array(Xdot_all[-1][:5000,:])
Tau_val = np.array(Tau_all[-1][:5000])

# time = np.arange(0.0, 0.5, 1e-4)
# mask = np.abs(X_list[0][:,0]) > 0
# X_list[0] = X_list[0][mask]
# Xdot_list[0] = Xdot_list[0][mask]
# time = np.arange(0.0, X_list[0].shape[0]*1e-4, 1e-4)
# fig, ax = plt.subplots(3,1)
# ax[0].plot(time, X_list[0][:,0])
# ax[0].grid(True)
# ax[1].plot(time, X_list[0][:,1])
# ax[1].grid(True)
# ax[2].plot(time, Xdot_list[0][:,1])
# ax[2].grid(True)
# plt.show()

# Stack variables (from all initial conditions)
X = np.vstack(X_list)
Xdot = np.vstack(Xdot_list)
Tau = np.vstack(Tau_list)

####################################################################
#### Soft manipulator parameters - change based on the use case ####
num_segments = 2
strain_selector = np.array([True, True, True, True, True, True]) # bending, shear and axial
string_strains = ['Bending','Shear','Axial','Bending','Shear','Axial']

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
    # "D": 5e-6 * jnp.diag(jnp.array([1e0, 1e3, 1e3])),
    "D": 5e-6 * jnp.diag(jnp.array([1e1, 1e3, 1e3, 1e1, 1e3, 1e3])),
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

bending_indices = [i for i in range(len(bending_map)) if bending_map[i]==True]
if bending_indices != []:
    mask = True
    for idx in bending_indices:
        mask = mask & (np.abs(X[:,idx]) > 10)
    X = X[mask]
    Xdot = Xdot[mask]
    Tau = Tau[mask]
# if bending_active == True:
#     # Remove samples where q is between [-0.1, 0.1]
#     mask = np.abs(X[:,0]) > 10
#     X = X[mask]
#     Xdot = Xdot[mask]
#     Tau = Tau[mask]


# time = np.arange(0.0, 0.5, 1e-4)
# fig, ax = plt.subplots(3,1)
# ax[0].plot(time, X[:5000,0])
# ax[0].grid(True)
# ax[1].plot(time, X[:5000,1])
# ax[1].grid(True)
# ax[2].plot(time, Xdot[:5000,1])
# ax[2].grid(True)

# adding noise
mu, sigma = 0, noiselevel
noise = np.random.normal(mu, sigma, X.shape[0])
for i in range(X.shape[1]):
    X[:,i] = X[:,i] + noise
    Xdot[:,i] = Xdot[:,i] + noise

# ax[0].plot(time, X[:5000,0])
# ax[1].plot(time, X[:5000,1])
# ax[2].plot(time, Xdot[:5000,1])
# plt.show()

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
        monoms = [prod(x**k for x, k in zip(p.gens, mon)) for mon in p.monoms()]
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
    monoms = [prod(x**k for x, k in zip(p.gens, mon)) for mon in p.monoms()]
    return coeffs, monoms

def constructLagrangianExpression(sym_exps, states_sym):
    true_coeffs = []
    expr = []

    xi_sym = Matrix(xi_eq) + Matrix(B_xi)*Matrix(states_sym[:len(states_sym)//2])
    for i in range(num_segments):
        if strain_selector[3*i] == False:
            xi_sym[3*i] = epsilon_bend

    # Extract the basis functions from the mass matrix 
    # Due to the symmetry of the mass matrix, it has (n**2+n)/2 independent entries. In the 
    # Lagrangian expression, each of these entries needs to be mutiplied by 1/2 and 
    # the corresponding q_dot**2
    B = Matrix(B_xi).T * sym_exps['exps']['B'] * Matrix(B_xi)

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

# compute time-series tensors for the lagrangian equation
device = 'cuda:0'
# device = 'cpu'
phi_q, phi_qdot2, phi_qdotq = EulerLagrangeExpressionTensor(expr, states, states_epsed, states_epsed_sym)
Zeta, Eta, Delta = LagrangianLibraryTensor(X, Xdot, X_epsed, states, states_dot, states_epsed, phi_q, phi_qdot2, phi_qdotq, scaling=False)
Eta = Eta.to(device)
Zeta = Zeta.to(device)
Delta = Delta.to(device)

############## Closed form least squares ##############
#######################################################
q_tt = torch.from_numpy(Xdot).to(device)[:, Xdot.shape[1]//2:].T
q_t = torch.from_numpy(Xdot).to(device)[:, :Xdot.shape[1]//2].T
X_ls = torch.einsum('ijkl,il->jkl', Zeta, q_tt) + torch.einsum('ijkl,il->jkl', Eta, q_t) - Delta
X_ls = torch.flatten(X_ls.permute(0,2,1), end_dim=1)
# Add basis function for velocity to also find the damping constants
if n_dof == 1:
    X_ls = torch.cat((X_ls, q_t.T), 1)
else:
    list_vel = [torch.flatten(q_t[i,:]).reshape(-1,1) for i in range(n_dof)] # block diagonal matrix 
    X_ls = torch.cat((X_ls, torch.block_diag(*list_vel)), 1)
# Apply least squares normal equation
coef_ls = torch.linalg.inv(X_ls.T @ X_ls) @ X_ls.T @ (torch.flatten(torch.from_numpy(Tau).to(device).T).reshape(-1,1))

xi_L = coef_ls[:-n_dof,0]
D = torch.diag(coef_ls[-n_dof:,0])

# xi_L = torch.from_numpy(true_coeffs)
# D = B_xi.T @ params['D'] @ B_xi
# D = torch.from_numpy(np.array(D)).to(device)

tau_pred, DL_q, DL_qdot2, DL_qdotq, A, C, B, Tau_NC = ELforward(xi_L, Zeta, Eta, Delta, Xdot, device, D)

def loss(pred, targ):
    loss = torch.mean((targ - pred)**2) 
    return loss 

lossval = loss(tau_pred.T, torch.from_numpy(Tau).to(device))

# fig, ax = plt.subplots(n_dof,1)
# if n_dof == 1:
#     ax.plot(torch.from_numpy(Tau).T[0,:], label='True Model')
#     ax.plot(tau_pred.cpu()[0,:], 'r--',label='Predicted Model')
#     ax.set_ylabel('$Tau$')
#     ax.grid(True)
# else:
#     for i in range(n_dof):
#         ax[i].plot(torch.from_numpy(Tau).T[i,:], label='True Model')
#         ax[i].plot(tau_pred.cpu()[i,:], 'r--',label='Predicted Model')
#         ax[i].set_ylabel('$Tau$')
#         ax[i].grid(True)
# plt.show()

# class xLSINDY(torch.nn.Module):
#     def __init__(self, xi_L, D):
        
#         super().__init__()
#         self.coef = torch.nn.Parameter(xi_L)
#         # self.register_buffer('weight_update_mask', )
#         self.D = torch.nn.Parameter(D) # learn the damping constant
#         # self.D = D

#     def forward(self, zeta, eta, delta, x_t, device):#, tau):

#         tau_pred = ELforward(self.coef, zeta, eta, delta, x_t, device, self.D)
#         # q_tt_pred = lagrangianforward(self.coef, zeta, eta, delta, x_t, device, tau)
#         return tau_pred

# def train(model, epochs, lr, lam, bs, optimizer, Zeta, Eta, Delta, Xdot, Tau, stage, scheduler=None):

#     if(torch.is_tensor(Tau)==False):
#         tau = torch.from_numpy(Tau).to(device)

#     if (torch.is_tensor(Xdot) == False):
#         Xdot = torch.from_numpy(Xdot).to(device)

#     j = 1
#     lossitem_prev = 1e8
#     loss_streak = 0
#     loss_plot = []
#     while (j <= epochs):
#         print("\n")
#         print("Stage " + str(stage))
#         print("Epoch " + str(j) + "/" + str(epochs))
#         print("Learning rate : ", lr)

#         tl = Xdot.shape[0]
#         loss_list = []
#         for i in range(tl//bs):
            
#             zeta = Zeta[:,:,:,i*bs:(i+1)*bs]
#             eta = Eta[:,:,:,i*bs:(i+1)*bs]
#             delta = Delta[:,:,i*bs:(i+1)*bs]
#             x_t = Xdot[i*bs:(i+1)*bs,:]
#             n = x_t.shape[1]

#             # If loss using torque
#             tau_pred = model.forward(zeta, eta, delta, x_t, device)
#             tau_true = (tau[i*bs:(i+1)*bs,:].T).to(device)
#             mse_loss = loss(tau_pred, tau_true)

#             # If loss using acceleration
#             # tau_true = (tau[i*bs:(i+1)*bs,:].T).to(device)
#             # q_tt_pred = model.forward(zeta, eta, delta, x_t, device, tau_true)
#             # q_tt_true = Xdot[i*bs:(i+1)*bs,n//2:].T
#             # mse_loss = loss(q_tt_pred, q_tt_true)

#             l1_norm = sum(
#                 p.abs().sum() for p in model.parameters()
#             )
#             lossval = mse_loss + lam * l1_norm

#             optimizer.zero_grad()
#             lossval.backward()
#             # model.coef.grad[:5] = 0
#             # optimizer.step()

#             loss_list.append(lossval.item())
        
#         lossitem = torch.tensor(loss_list).mean().item()
#         print("Average loss : " , lossitem)
#         loss_plot.append(lossitem)

#         # scheduler.step()
#         lr = scheduler.get_last_lr()
#         # Tolerance (sufficiently good loss)
#         if (lossitem <= 1e-7):
#             break

#         # Early stopping
#         # if (lossitem - lossitem_prev) > -1e-4:
#         #     loss_streak += 1
#         # if loss_streak == 5:
#         #     break
#         # lossitem_prev = lossitem

#         j += 1
    
#     return model, loss_plot


# ## First stage ##

# # Initialize model

# model = xLSINDY(xi_L, D)
# # L = '0.0002241*x0_t**2-6.283185305*x0**2-0.06595270528*x0'

# # Training parameters
# Epoch = 300
# lr = 1e-7 # alpha=learning_rate
# lam = 0 # sparsity promoting parameter (l1 regularization)
# # bs = 2020 # batch size
# bs = Xdot.shape[0]
# optimizer = torch.optim.Adam(
#     model.parameters(), lr=lr
# )
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.01)

# # model, loss_plot = train(model, Epoch, lr, lam, bs, optimizer, Zeta, Eta, Delta, Xdot_train, Tau_train, 1, scheduler=scheduler)
# model, loss_plot = train(model, Epoch, lr, lam, bs, optimizer, Zeta, Eta, Delta, Xdot, Tau, 1, scheduler=scheduler)

# ## Thresholding small indices ##
# xi_L = model.coef
# D = model.D
# threshold = 1e-12
# surv_index = ((torch.abs(xi_L) >= threshold)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
# # surv_index = np.append(surv_index, xi_L.shape[0]-1)
# expr = np.array(expr)[surv_index].tolist()
# xi_L = xi_L[surv_index].clone().detach().requires_grad_(True)

## Obtaining analytical model ##
# xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=10)
xi_Lcpu = xi_L.detach().cpu().numpy()
L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-10)
# print(simplify(L))

# # tau_pred = ELforward(xi_L, Zeta[:,:,:,:X_train_list[0].shape[0]], Eta[:,:,:,:X_train_list[0].shape[0]], Delta[:,:,:X_train_list[0].shape[0]], Xdot_train[:X_train_list[0].shape[0],:], device, D).detach().cpu()
# tau_pred = ELforward(xi_L, Zeta[:,:,:,:5000], Eta[:,:,:,:5000], Delta[:,:,:5000], Xdot[:5000,:], device, D).detach().cpu()
# # time = np.arange(0.0, (X_train_list[0].shape[0])*1e-4, 1e-4)
# time = np.arange(0.0, 0.5, 1e-4)
# # plt.plot(time, tau_pred[0,:], '--r', time, Tau_train_list[0])
# plt.plot(time, tau_pred[0,:], '--r', time, Tau[:5000,0])
# plt.show()
# # loss_tau = loss(tau_pred, torch.from_numpy(Tau_train_list[0]))
# loss_tau = loss(tau_pred, torch.from_numpy(Tau[:5000,0]))

## Next learning stages ##
# for stage in range(2):
    
#     # Redefine computation after thresholding
#     Zeta, Eta, Delta = LagrangianLibraryTensor(X_train, Xdot_train, expr, states, states_dot, scaling=False, x_epsed=q_epsed, states_epsed=states_epsed)
#     # Zeta, Eta, Delta = LagrangianLibraryTensor(X,Xdot,expr,states,states_dot,scaling=False)
#     Eta = Eta.to(device)
#     Zeta = Zeta.to(device)
#     Delta = Delta.to(device)

#     model = xLSINDY(xi_L, D)

#     # Training parameters
#     Epoch = 100
#     lam = lam*0.1
#     lr = lr*0.1
#     bs = 2020
#     optimizer = torch.optim.Adam(
#         model.parameters(), lr=lr
#     )

#     model, loss_plot = train(model, Epoch, lr, lam, bs, optimizer, Zeta, Eta, Delta, Xdot_train, Tau_train, stage+2)

#     ## Thresholding small indices ##
#     xi_L = model.coef
#     D = model.D
#     threshold = 1e-8
#     surv_index = ((torch.abs(xi_L) >= threshold)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
#     # surv_index = np.append(surv_index, xi_L.shape[0]-1)
#     expr = np.array(expr)[surv_index].tolist()
#     xi_L =xi_L[surv_index].clone().detach().requires_grad_(True)

#     ## Obtaining analytical model ##
#     # xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=10)
#     xi_Lcpu = xi_L.detach().cpu().numpy()
#     L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-10)
#     print("Result stage " + str(stage+2) + ":" , simplify(L))

print('\n')
print('------------')
print('\n')
print('L=' + str(L))
print('\n')
print('------------')
# --------------- Validation plots --------------------

# obtain equations of motion
def getEOM(xi_Lcpu, phi_q, phi_qdot2, phi_qdotq):

    delta_expr = sympy.Matrix(phi_q) @ sympy.Matrix(xi_Lcpu)
    eta_expr = (sympy.Matrix(phi_qdotq.reshape(n_dof*n_dof, -1)) @ sympy.Matrix(xi_Lcpu)).reshape(n_dof, n_dof)
    zeta_expr = (sympy.Matrix(phi_qdot2.reshape(n_dof*n_dof, -1)) @ sympy.Matrix(xi_Lcpu)).reshape(n_dof, n_dof)

    return delta_expr, eta_expr, zeta_expr

delta_expr, eta_expr, zeta_expr = getEOM(xi_L.cpu(), phi_q, phi_qdot2, phi_qdotq)

delta_expr_lambda = sympy.lambdify([*states_sym[:n_dof], *states_sym[n_dof:], *states_epsed_sym], delta_expr, 'numpy')
eta_expr_lambda = sympy.lambdify([*states_sym[:n_dof], *states_sym[n_dof:], *states_epsed_sym], eta_expr, 'numpy')
zeta_expr_lambda = sympy.lambdify([*states_sym[:n_dof], *states_sym[n_dof:], *states_epsed_sym], zeta_expr, 'numpy')

delta_expr_lambda_jax = sympy.lambdify([*states_sym[:n_dof], *states_sym[n_dof:], *states_epsed_sym], delta_expr, 'jax')
eta_expr_lambda_jax = sympy.lambdify([*states_sym[:n_dof], *states_sym[n_dof:], *states_epsed_sym], eta_expr, 'jax')
zeta_expr_lambda_jax = sympy.lambdify([*states_sym[:n_dof], *states_sym[n_dof:], *states_epsed_sym], zeta_expr, 'jax')

# tau_pred = []
# Zeta_q_tt_list, Eta_q_t_list, Delta_list, D_q_t_list = [], [], [], []
# DL_qdotq_list, DL_qdot2_list = [], []
# for i in range(X.shape[0]):
#     # zeta = zeta_expr_lambda(X[i,0], X[i,1], X[i,2], X[i,3], X_epsed[i,0], X_epsed[i,1])
#     zeta = zeta_expr_lambda(*X[i,:n_dof], *Xdot[i,:n_dof], *X_epsed[i,:]).T
#     # eta = eta_expr_lambda(X[i,0], X[i,1], X[i,2], X[i,3], X_epsed[i,0], X_epsed[i,1]).T
#     eta = eta_expr_lambda(*X[i,:n_dof], *X[i,n_dof:], *X_epsed[i,:]).T
#     # delta = delta_expr_lambda(X[i,0], X[i,1], X[i,2], X[i,3], X_epsed[i,0], X_epsed[i,1])[:,0]
#     delta = delta_expr_lambda(*X[i,:n_dof], *X[i,n_dof:], *X_epsed[i,:])[:,0]

#     zeta_q_tt = zeta @ Xdot[i,n_dof:]
#     eta_q_t = eta @ Xdot[i,:n_dof]
#     D_q_t = D.cpu().numpy() @ X[i,n_dof:]

#     Zeta_q_tt_list.append(zeta_q_tt)
#     Eta_q_t_list.append(eta_q_t)
#     Delta_list.append(delta)
#     D_q_t_list.append(D_q_t)
#     DL_qdotq_list.append(DL_qdotq[:,:,i].cpu().numpy() @ Xdot[i,:n_dof])
#     DL_qdot2_list.append(DL_qdot2[:,:,i].cpu().numpy() @ Xdot[i,n_dof:])

#     # tau = zeta_expr_lambda(*X[i,:n_dof], *Xdot[i,:n_dof], *X_epsed[i,:]) @ Xdot[i,n_dof:] + eta_expr_lambda(*X[i,:n_dof], *X[i,n_dof:], *X_epsed[i,:]) @ X[i,n_dof:] - delta_expr_lambda(*X[i,:n_dof], *X[i,n_dof:], *X_epsed[i,:])[:,0] + D.detach().cpu().numpy() @ X[i,n_dof:]
#     tau = (zeta @ Xdot[i,n_dof:]) + (eta @ X[i,n_dof:]) - delta + (D.cpu().numpy() @ X[i,n_dof:])
#     # tau = zeta_expr_lambda(X_epsed[i,0], X_epsed[i,1]) @ Xdot[i,n_dof:] + eta_expr_lambda(X_epsed[i,0], X[i,2], X_epsed[i,1], X[i,3]) @ X[i,n_dof:] - delta_expr_lambda(X[i,0], X_epsed[i,0], X[i,2], X[i,1], X_epsed[i,1], X[i,3])[:,0] + D.detach().cpu().numpy() @ X[i,n_dof:]
#     # tau = zeta_expr_lambda(X[i,0], X[i,1]) @ Xdot[i,n_dof:] + eta_expr_lambda(X[i,0], X[i,2], X[i,1], X[i,3]) @ Xdot[i,:n_dof] - delta_expr_lambda(X[i,0], X[i,0], X[i,2], X[i,1], X[i,1], X[i,3])[:,0] + D.detach().cpu().numpy() @ Xdot[i,:n_dof]
#     tau_pred.append(tau)

# tau_pred = np.stack(tau_pred)
# Zeta_q_tt_list = np.stack(Zeta_q_tt_list).T
# Eta_q_t_list = np.stack(Eta_q_t_list).T
# Delta_list = np.stack(Delta_list).T
# D_q_t_list = np.stack(D_q_t_list).T
# DL_qdotq_list = np.stack(DL_qdotq_list)
# DL_qdot2_list = np.stack(DL_qdot2_list)


# fig, ax = plt.subplots(n_dof, 1)
# ax[0].plot(A.cpu()[0,:], label='A')
# ax[0].plot(Zeta_q_tt_list[0,:], 'r--', label='Zeta_q_tt')
# ax[0].plot(DL_qdot2_list[:,0], label='DL_qdot2')
# ax[0].grid(True)
# ax[1].plot(A.cpu()[1,:], label='A')
# ax[1].plot(Zeta_q_tt_list[1,:], 'r--', label='Zeta_q_tt')
# ax[1].plot(DL_qdot2_list[:,1], label='DL_qdot2')
# ax[1].grid(True)
# plt.show()

# fig, ax = plt.subplots(n_dof, 1)
# ax[0].plot(C.cpu()[0,:], label='C')
# ax[0].plot(Eta_q_t_list[0,:], 'r--', label='Eta_q_t')
# # ax[0].plot(DL_qdotq_list[:,0], label='DL_qdotq')
# ax[0].grid(True)
# ax[1].plot(C.cpu()[1,:], label='C')
# ax[1].plot(Eta_q_t_list[1,:], 'r--', label='Eta_q_t')
# # ax[1].plot(DL_qdotq_list[:,1], label='DL_qdotq')
# ax[1].grid(True)
# plt.show()

# fig, ax = plt.subplots(n_dof, 1)
# ax[0].plot(B.cpu()[0,:], label='B')
# ax[0].plot(Delta_list[0,:], 'r--', label='Delta')
# ax[0].grid(True)
# ax[1].plot(B.cpu()[1,:], label='B')
# ax[1].plot(Delta_list[1,:], 'r--', label='Delta')
# ax[1].grid(True)
# plt.show()

# fig, ax = plt.subplots(n_dof, 1)
# ax[0].plot(-Tau_NC.cpu()[0,:], label='Tau_NC')
# ax[0].plot(D_q_t_list[0,:], 'r--', label='D_q_t')
# ax[0].grid(True)
# ax[1].plot(-Tau_NC.cpu()[1,:], label='Tau_NC')
# ax[1].plot(D_q_t_list[1,:], 'r--', label='D_q_t')
# ax[1].grid(True)
# plt.show()

# fig, ax = plt.subplots(n_dof,1)
# if n_dof == 1:
#     ax.plot(Tau[:,0], label='True Model')
#     ax.plot(tau_pred[:,0], 'r--',label='Predicted Model')
#     ax.set_ylabel('$Tau$')
#     ax.grid(True)
# else:
#     for i in range(n_dof):
#         ax[i].plot(Tau[:,i], label='True Model')
#         ax[i].plot(tau_pred[:,i], 'r--',label='Predicted Model')
#         ax[i].set_ylabel('$Tau$')
#         ax[i].grid(True)
# plt.show()

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
    # return jnp.array([x0_t, eval(str(eom[0]))])
    # return jnp.array([x0_t, eval(str(eom))])
    x_tt = jnp.linalg.inv(zeta_expr_lambda_jax(*x_, *x_t, *x_epsed).T) @ (tau - D @ x_t + delta_expr_lambda_jax(*x_, *x_t, *x_epsed)[:,0] - eta_expr_lambda_jax(*x_, *x_t, *x_epsed).T @ x_t)
    # x0_tt = (1/zeta_expr_lambda(x0_epsed))*(tau - D*x0_t + delta_expr_lambda(x0,x0_t,x0_epsed) - eta_expr_lambda(x0_t,x0_epsed)*x0_t )
    # return jnp.array([x0_t, eom_lambda(x0, x0_t, x0_epsed)])
    return jnp.concatenate([x_t, x_tt])
    # return x0_t, eom_lambda(x0, x0_t, x0_epsed)#eval(str(eom[0]))

## Training results ##
# true results
q_tt_true = (Xdot_val[:,states_dim//2:].T).copy()
q_t_true = (Xdot_val[:,:states_dim//2].T).copy()
q_true = (X_val[:,:states_dim//2].T).copy()
tau_true = (Tau_val.T).copy()

# prediction results
dt = 1e-4  # time step
time_ = jnp.arange(0.0, 0.5, dt)
y_0 = X_val[0,:]
Xpred, Xdotpred = generate_data(softrobot, time_, y_0, Tau_val)

# fig, ax = plt.subplots(1,1)
# ax.plot(time_, Xdotpred[:,states_dim//2:], label='Forward acceleration')
# ax.plot(time_, acc_gradient, label='Gradient acceleration')
# ax.set_ylabel('$\ddot{q}$')
# # ax[0].vlines(0.4,0,1,transform=ax[0].get_xaxis_transform(),colors='k')
# ax.set_xlim([0,0.5])
# ax.grid(True)
# Line, Label = ax.get_legend_handles_labels()
# fig.legend(Line, Label, loc='upper right')
# plt.show()

q_tt_pred = Xdotpred[:,states_dim//2:].T
q_t_pred = Xdotpred[:,:states_dim//2].T
q_pred = Xpred[:,:states_dim//2].T

# Validation loss
X_epsed_val = np.zeros((X_val.shape[0], n_dof))
for i in range(n_dof):
    if bending_map[i] == True:
        q_epsed = apply_eps_to_bend_strains(X_val[:,i], 1e0)
    else:
        q_epsed = X_val[:,i]
    
    X_epsed_val[:,i] = q_epsed

# q_epsed_val = apply_eps_to_bend_strains(X_val[:,0], 1e0)
Zeta_val, Eta_val, Delta_val = LagrangianLibraryTensor(X_val, Xdot_val, X_epsed_val, states, states_dot, states_epsed, phi_q, phi_qdot2, phi_qdotq, scaling=False)
Eta_val = Eta_val.to(device)
Zeta_val = Zeta_val.to(device)
Delta_val = Delta_val.to(device)
tau_pred, _, _, _, _, _, _, _ = ELforward(xi_L, Zeta_val, Eta_val, Delta_val, Xdot_val, device, D)
tau_pred = tau_pred.cpu()
loss_val = loss(tau_pred, torch.from_numpy(tau_true))
# tau_pred = ELforward(xi_L, Zeta[:,:,:,:5000], Eta[:,:,:,:5000], Delta[:,:,:5000], Xdot[:5000,:], device, D)
# tau_pred = tau_pred.detach().cpu().numpy()

fig, ax = plt.subplots(n_dof,1)
if n_dof == 1:
    ax.plot(torch.from_numpy(tau_true)[0,:], label='True Model')
    ax.plot(tau_pred.cpu()[0,:], 'r--',label='Predicted Model')
    ax.set_ylabel('$Tau$')
    ax.grid(True)
else:
    for i in range(n_dof):
        ax[i].plot(torch.from_numpy(tau_true)[i,:], label='True Model')
        ax[i].plot(tau_pred.cpu()[i,:], 'r--',label='Predicted Model')
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

