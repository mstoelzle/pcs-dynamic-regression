#%%
import numpy as np
import sys 
from sympy import symbols, simplify, derive_by_array
from scipy.integrate import solve_ivp
from xLSINDy import *
from sympy.physics.mechanics import *
from sympy import *
import sympy
import torch
import math
sys.path.append(r'../../../HLsearch/')
import HLsearch as HL
import time
import matplotlib.pyplot as plt
import csv

rootdir = "./Soft Robot/ns-1_dof-1_zero_actuation/"
save = False
noiselevel = 0

# 60 samples of 0.6 seconds each (1e-3 timestep)
X_all = np.load(rootdir + "X.npy")
Xdot_all = np.load(rootdir + "Xdot.npy")
Tau_all = np.load(rootdir + "Tau.npy")

X_list, Xdot_list, Tau_list = [], [], []
# X_val, Xdot_val, Tau_val = []

for i in range(len(X_all)):
    X_list.append(X_all[i][:2000,:])
    # X_val.append(X_all[i][400:,:])

    Xdot_list.append(Xdot_all[i][:2000,:])
    # Xdot_val.append(Xdot_all[i][400:,:])

    Tau_list.append(Tau_all[i][:2000])
    # Tau_val.append(Tau_all[i][400:])

# variables without noise (wn)
X = np.vstack(X_list)
Xdot = np.vstack(Xdot_list)
Tau = np.vstack(Tau_list)


###### Plots #####
# t = np.arange(0,0.10,1e-4)
# fig, ax = plt.subplots(4,1,figsize=(6,10))

# ax[0].plot(t, X[:1000,0], 'r')
# # ax[0].plot(t, X[27][:,1], 'g')
# ax[0].set_ylabel('$q$')
# ax[0].grid()

# ax[1].plot(t, X[:1000,1], 'r')
# ax[1].plot(t, Xdot[:1000,0], 'g--')
# ax[1].set_ylabel('$\dot{q}$')
# ax[1].grid()

# # ax[2].plot(t, Tau[4])^
# ax[2].plot(t, Xdot[:1000,1], 'r')
# ax[2].set_ylabel('$\ddot{q}$')
# ax[2].set_xlabel('Time (s)')
# ax[2].grid()

# ax[3].plot(t,Tau[:1000])
# ax[3].plot(t,-1e-2*Xdot[:1000,0], 'g--')
# ax[3].grid()
# plt.show()

# variables without noise (wn)
# X = np.vstack(X)
# Xdot = np.vstack(Xdot)
# X_wn = np.copy(X)
# Xdot_wn = np.copy(Xdot)

# Tau = np.vstack(Tau)

#adding noise
# mu, sigma = 0, noiselevel
# noise = np.random.normal(mu, sigma, X_wn.shape[0])
# for i in range(X_wn.shape[1]):
#     X[:,i] = X[:,i] + noise
#     Xdot[:,i] = Xdot_wn[:,i] + noise

states_dim = 2
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
states_sym = states
states_dot_sym = states_dot
states = list(str(descr) for descr in states)
states_dot = list(str(descr) for descr in states_dot)

#build function expression for the library in str
expr= HL.buildFunctionExpressions(2,states_dim,states,use_sine=False)
expr.pop(3)
expr.pop(1)
# expr.pop(1)
# expr.pop(5) #remove theta*theta_dot
# expr.pop(1) # remove theta_dot

# compute time-series tensors for the lagrangian equation
device = 'cuda:0'
Zeta, Eta, Delta = LagrangianLibraryTensor(X,Xdot,expr,states,states_dot,scaling=False)
Eta = Eta.to(device)
Zeta = Zeta.to(device)
Delta = Delta.to(device)

#mask = torch.ones(len(expr),device=device)
# initialize coefficients
torch.manual_seed(1)
xi_L = torch.ones(len(expr), device=device).data.uniform_(-1,1)
xi_L[-1] = abs(xi_L[-1])
# xi_L[-1] = 0.0002241
# xi_L = torch.tensor((-0.06595270528, -6.283185305, 0.0002241), device=device)
prevxi_L = xi_L.clone().detach()

def loss(pred, targ):
    loss = torch.mean((targ - pred)**2) 
    return loss 


def clip(w, alpha):
    clipped = torch.minimum(w,alpha)
    clipped = torch.maximum(clipped,-alpha)
    return clipped

def proxL1norm(w_hat, alpha):
    if(torch.is_tensor(alpha)==False):
        alpha = torch.tensor(alpha)
    w = w_hat - clip(w_hat,alpha)
    return w


def training_loop(coef, prevcoef, Zeta, Eta, Delta, xdot, bs, lr, lam, tau=0):
    loss_list = []
    tl = xdot.shape[0]
    n = xdot.shape[1]

    if(torch.is_tensor(xdot)==False):
        xdot = torch.from_numpy(xdot).to(device).float()

    if(torch.is_tensor(tau)==False):
        tau = torch.from_numpy(tau).to(device).float()
    
    for i in range(tl//bs):
        
        #computing acceleration with momentum
        # v = (coef + ((i-1)/(i+2))*(coef - prevcoef)).clone().detach().requires_grad_(True)
        coef = coef.clone().detach().requires_grad_(True)
        prevcoef = coef.clone().detach()

        #Computing loss
        zeta = Zeta[:,:,:,i*bs:(i+1)*bs]
        eta = Eta[:,:,:,i*bs:(i+1)*bs]
        delta = Delta[:,:,i*bs:(i+1)*bs]
        x_t = xdot[i*bs:(i+1)*bs,:]

        #forward
        # tau_ = (tau[i*bs:(i+1)*bs,:].T).to(device)
        # q_tt_pred = lagrangianforward(coef,zeta,eta,delta,x_t,device,tau=tau_)
        # q_tt_true = xdot[i*bs:(i+1)*bs,n//2:].T
        # lossval = loss(q_tt_pred, q_tt_true)

        # tau_pred = ELforward(v, zeta, eta, delta, x_t, device)
        # weight = torch.cat((coef[:-1].reshape(-1), torch.tensor(coef[-1]**2 + 0.0001).reshape(-1)))
        tau_pred = ELforward(coef, zeta, eta, delta, x_t, device)
        tau_true = (tau[i*bs:(i+1)*bs,:].T).to(device)
        lossval = loss(tau_pred, tau_true)
        l1_norm = sum(abs(c) for c in coef)
        lossval = lossval + lam*l1_norm
        
        #Backpropagation
               
        lossval.backward()
        with torch.no_grad():
            # vhat = v - lr*v.grad
            # coef = (proxL1norm(vhat,lr*lam)).clone().detach()
            coef[0] = coef[0] - lr[0]*coef.grad[0]
            coef[1] = coef[1] - lr[0]*coef.grad[1]
            coef[2] = coef[2] - lr[1]*coef.grad[2]
            # coef[-1] = 0.0002241
            # coef[-1] = coef[-1]**2 + 0.0001
            

        loss_list.append(lossval.item())
    print("Average loss : " , torch.tensor(loss_list).mean().item())
    return coef, prevcoef, torch.tensor(loss_list).mean().item()


Epoch = 100
i = 1
lr = [5e-5, 5e-10] # alpha=learning_rate
lam = 0 # sparsity promoting parameter (l1 regularization)
bs = 1000 # batch size
temp = 1000
lossitem_prev = 1e8
loss_streak = 0
while(i<=Epoch):
    print("\n")
    print("Stage 1")
    print("Epoch "+str(i) + "/" + str(Epoch))
    print("Learning rate : ", lr)
    xi_L, prevxi_L, lossitem= training_loop(xi_L,prevxi_L,Zeta,Eta,Delta,Xdot,bs=bs,lr=lr,lam=lam,tau=Tau)
    if(lossitem <=5e-3):
        break
    if(lossitem <=1):
        lr = lr
    # if (lossitem - lossitem_prev) > -1e-4:
    #     loss_streak+=1
    # if loss_streak == 5:
    #     break
    # lossitem_prev = lossitem
    i+=1


## Thresholding small indices ##
threshold = 1e-3
surv_index = ((torch.abs(xi_L[:-1]) >= threshold)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
surv_index = np.append(surv_index, xi_L.shape[0]-1)
expr = np.array(expr)[surv_index].tolist()

xi_L = xi_L[surv_index].clone().detach().requires_grad_(True)
prevxi_L = xi_L.clone().detach()
#mask = torch.ones(len(expr),device=device)

## obtaining analytical model
xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=6)
L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-10)
print(simplify(L))



## Next round Selection ##
for stage in range(2):
    
    #Redefine computation after thresholding
    Zeta, Eta, Delta = LagrangianLibraryTensor(X,Xdot,expr,states,states_dot,scaling=False)
    Eta = Eta.to(device)
    Zeta = Zeta.to(device)
    Delta = Delta.to(device)

    #Training
    Epoch = 100
    i = 1
    lam = lam*0.1
    lr = lr*1
    lossitem_prev = 1e8
    loss_streak = 0
    # if(stage==1):
    #     lam = 0.001
    #     lr = 4e-5
    # else:
    #     lam = 0.01
    #     lr = 2e-5
    temp = 1000
    while(i<=Epoch):
        print("\n")
        print("Stage " + str(stage+2))
        print("Epoch "+str(i) + "/" + str(Epoch))
        print("Learning rate : ", lr)
        xi_L , prevxi_L, lossitem= training_loop(xi_L,prevxi_L,Zeta,Eta,Delta,Xdot,bs=bs,lr=lr,lam=lam,tau=Tau)
        if(lossitem <=1e-3):
            break
        # if (lossitem - lossitem_prev) > -1e-4:
        #     loss_streak+=1
        # if loss_streak == 5:
        #     break
        # lossitem_prev = lossitem
        i+=1
    
    ## Thresholding small indices ##
    threshold = 1e-3
    surv_index = ((torch.abs(xi_L[:-1]) >= threshold)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
    surv_index = np.append(surv_index, xi_L.shape[0]-1)
    expr = np.array(expr)[surv_index].tolist()

    xi_L =xi_L[surv_index].clone().detach().requires_grad_(True)
    prevxi_L = xi_L.clone().detach()
    mask = torch.ones(len(expr),device=device)

    ## obtaining analytical model
    xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=6)
    L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-10)
    print("Result stage " + str(stage+2) + ":" , simplify(L))

# --------------- Validation plots --------------------

# ## obtaining analytical model
# xi_Lcpu[-1] = xi_Lcpu[-1]**2 + 0.0001
# L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-10)

# obtain equations of motion
x0 = dynamicsymbols(states[0])
x0_t = dynamicsymbols(states[0],1)
L = eval(str(L))
LM = LagrangesMethod(L, [x0])
LM.form_lagranges_equations()
eom = LM.rhs()[1]

#convert to string
eom = str(eom).replace('(t)','')

def generate_data(func, time, init_values):
    sol = solve_ivp(func,[time[0],time[-1]],init_values,t_eval=time,method='RK45',rtol=1e-10,atol=1e-10)
    return sol.y.T, np.array([func(time[i],sol.y.T[i,:]) for i in range(sol.y.T.shape[0])],dtype=np.float64)
    # first output is X array in [x,x_dot] format
    # second output is X_dot array in [x_dot,x_doubledot] format

def softrobot(t,x):
    x0 = x[0]
    x0_t = x[1]
    return x0_t, eval(eom) + (1/(2*np.around(xi_L[-1].detach().cpu().numpy(),decimals=8)))*(-1e-2*x0_t)

## Training results ##
# true results
q_tt_true = (Xdot[:2000,states_dim//2:].T).copy()
q_t_true = (Xdot[:2000,:states_dim//2].T).copy()
q_true = (X[:2000,:states_dim//2].T).copy()
tau_true = (Tau[:2000,:].T).copy()

# prediction results
dt = 1e-4  # time step
time_ = np.arange(0.0, 0.2, dt)
y_0 = np.array([X[0,0], X[0,1]])
Xpred, Xdotpred = generate_data(softrobot, time_, y_0)
q_tt_pred = Xdotpred[:,states_dim//2:].T
q_t_pred = Xdotpred[:,:states_dim//2].T
q_pred = Xpred[:,:states_dim//2].T
tau_pred = ELforward(xi_L, Zeta[:,:,:,:2000], Eta[:,:,:,:2000], Delta[:,:,:2000], Xdot[:2000,:], device)
tau_pred = tau_pred.detach().cpu().numpy()

## Validation Results ##
# true results
# q_tt_true_val = (Xdot[300:600,states_dim//2:].T).copy()
# q_t_true_val = (Xdot[300:600,:states_dim//2].T).copy()
# q_true_val = (X[300:600,:states_dim//2].T).copy()

# # prediction results
# tval = np.arange(0.3, 0.6, dt)
# y_0 = np.array([q_pred[0,-1], q_t_pred[0,-1]])
# Xtestpred, Xdottestpred = generate_data(softrobot, tval, y_0)

# ## Concatenate training and test data ##
# # true results
# t = np.concatenate((time_, tval))
# q_tt_true = np.concatenate((q_tt_true, q_tt_true_val), axis=1)
# q_t_true = np.concatenate((q_t_true, q_t_true_val), axis=1)
# q_true = np.concatenate((q_true, q_true_val), axis=1)

# # prediction results
# q_tt_pred = np.concatenate((q_tt_pred,Xdottestpred[:,states_dim//2:].T), axis=1)
# q_t_pred = np.concatenate((q_t_pred, Xtestpred[:,states_dim//2:].T), axis=1)
# q_pred = np.concatenate((q_pred,Xtestpred[:,:states_dim//2].T), axis=1)

## Plotting
t = time_
fig, ax = plt.subplots(4,1)

ax[0].plot(t, q_tt_true[0,:], label='True Model')
ax[0].plot(t, q_tt_pred[0,:], 'r--',label='Predicted Model')
ax[0].set_ylabel('$\ddot{q}$ (rad)')
# ax[0].vlines(0.4,0,1,transform=ax[0].get_xaxis_transform(),colors='k')
ax[0].set_xlim([0,0.2])
# ax[0].set_ylim([-0.3,0.3])

ax[1].plot(t, q_t_true[0,:], label='True Model')
ax[1].plot(t, q_t_pred[0,:], 'r--',label='Predicted Model')
ax[1].set_ylabel('$\dot{q}$ (rad)')
# ax[1].vlines(0.4,0,1,transform=ax[1].get_xaxis_transform(),colors='k')
ax[1].set_xlim([0,0.2])
# ax[1].set_ylim([-0.3,0.3])

ax[2].plot(t, q_true[0,:], label='True Model')
ax[2].plot(t, q_pred[0,:], 'r--',label='Predicted Model')
ax[2].set_xlabel('Time (s)')
ax[2].set_ylabel('$q$ (rad)')
# ax[2].vlines(0.4,0,1,transform=ax[2].get_xaxis_transform(),colors='k')
ax[2].set_xlim([0,0.2])
# ax[2].set_ylim([-0.5,0.5])

ax[3].plot(t, tau_true[0,:], label='True Model')
ax[3].plot(t, tau_pred[0,:], 'r--', label='Predicted Model')
ax[3].set_xlabel('Time (s)')
ax[3].set_ylabel('$Tau$')
ax[3].set_xlim([0,0.2])

Line, Label = ax[0].get_legend_handles_labels()
fig.legend(Line, Label, loc='upper right', bbox_to_anchor=(1.5, 0.98))

fig.tight_layout()
plt.show()