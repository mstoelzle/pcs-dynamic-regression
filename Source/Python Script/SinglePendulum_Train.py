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


# functions to generate pendulum data
def generate_data(func, time, init_values, a, b, omega):
    sol = solve_ivp(func,[time[0],time[-1]],init_values,t_eval=time,method='RK45',args=[a,b,omega],rtol=1e-10,atol=1e-10)
    return sol.y.T, np.array([func(time[i],sol.y.T[i,:],a,b,omega) for i in range(sol.y.T.shape[0])],dtype=np.float64)
    # first output is X array in [x,x_dot] format
    # second output is X_dot array in [x_dot,x_doubledot] format

def pendulum(t,x,a,b,omega):
    return x[1],-9.81*np.sin(x[0]) + a*math.sin(omega*t) + b*math.cos(omega*t) # the 'a*math.sin(omega*t) + b*math.cos(omega*t)' is a sinusoidal external torque


#Saving Directory
rootdir = "../Single Pendulum/Data/"

num_sample = 100
create_data = True
save = False
noiselevel = 0


if(create_data):
    X, Xdot, Tau = [], [], []
    Tau_charact = []
    for i in range(num_sample):
        t = np.arange(0,5,0.01)

        # initial conditions
        theta = np.random.uniform(-np.pi, np.pi)
        #thetadot = np.random.uniform(-2.1,2.1)
        thetadot = 0
        #cond = 0.5*thetadot**2 - np.cos(theta)
        #checking condition so that it does not go full loop
        # while(cond>0.99):
        #     theta = np.random.uniform(-np.pi, np.pi)
        #     thetadot = np.random.uniform(-2.1,2.1)

        
        #     cond = 0.5*thetadot**2 - np.cos(theta)
        
        # define the external actuation
        def external_torque(t, a, b, omega): return a*np.sin(omega*t).reshape(-1,1) + b*np.cos(omega*t).reshape(-1,1)
        omega = np.random.uniform(1.0, 4.0) # generate random frequency
        a = np.random.uniform(-0.2, 0.2)
        b = np.random.uniform(-0.2, 0.2)
        tau = external_torque(t, a, b, omega)
        
        y_0 = np.array([theta, thetadot])
        x,xdot = generate_data(pendulum, t, y_0, a, b, omega)
        X.append(x)
        Xdot.append(xdot)
        Tau.append(tau)
        Tau_charact.append([a,b,omega])
    
    if(save==True):
        np.save(rootdir + "Active/X.npy", X)
        np.save(rootdir + "Active/Xdot.npy", Xdot)
        np.save(rootdir + "Active/Tau.npy", Tau)
else:
    X = np.load(rootdir + "Active/X.npy")
    Xdot = np.load(rootdir + "Active/Xdot.npy")
    Tau = np.load(rootdir + "Active/Tau.npy")

# Plots
fig, ax = plt.subplots(3,1,figsize=(5,6))

ax[0].plot(t, X[0][:,0], 'r')
ax[0].plot(t, X[0][:,1], 'g')
ax[0].set_ylabel('$\\theta$ (rad)')

ax[1].plot(t, Xdot[0][:,0], 'r')
# ax[1].plot(t, X[0][:,1], 'g')
ax[1].plot(t, Xdot[0][:,1], 'g')
ax[1].set_ylabel('$\\theta$ (rad)')

ax[2].plot(t, Tau[0])
plt.show()

# variables without noise (wn)
X = np.vstack(X)
Xdot = np.vstack(Xdot)
X_wn = np.copy(X)
Xdot_wn = np.copy(Xdot)

Tau = np.vstack(Tau)

#adding noise
mu, sigma = 0, noiselevel
noise = np.random.normal(mu, sigma, X_wn.shape[0])
for i in range(X_wn.shape[1]):
    X[:,i] = X[:,i] + noise
    Xdot[:,i] = Xdot_wn[:,i] + noise

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
expr= HL.buildFunctionExpressions(2,states_dim,states,use_sine=True)
expr.pop(5) #remove theta*theta_dot
expr.pop(1) # remove theta_dot
expr = ['x0_t**2', 'cos(x0)']

# compute time-series tensors for the lagrangian equation
device = 'cuda:0'
Zeta, Eta, Delta = LagrangianLibraryTensor(X,Xdot,expr,states,states_dot,scaling=False)
Eta = Eta.to(device)
Zeta = Zeta.to(device)
Delta = Delta.to(device)

#mask = torch.ones(len(expr),device=device)
# initialize coefficients
# xi_L = torch.ones(len(expr), device=device).data.uniform_(-10,10)
xi_L = torch.tensor((0.5, 9.81), device=device)
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
        v = (coef + ((i-1)/(i+2))*(coef - prevcoef)).clone().detach().requires_grad_(True)
        prevcoef = coef.clone().detach()


        #Computing loss
        zeta = Zeta[:,:,:,i*bs:(i+1)*bs]
        eta = Eta[:,:,:,i*bs:(i+1)*bs]
        delta = Delta[:,:,i*bs:(i+1)*bs]
        x_t = xdot[i*bs:(i+1)*bs,:]

        #forward
        tau_true = (tau[i*bs:(i+1)*bs,:].T).to(device)
        q_tt_pred = lagrangianforward(v,zeta,eta,delta,x_t,device,tau_true)
        q_tt_true = xdot[i*bs:(i+1)*bs,n//2:].T
        lossval = loss(q_tt_pred, q_tt_true)

        # tau_pred = ELforward(v, zeta, eta, delta, x_t, device)
        # tau_true = (tau[i*bs:(i+1)*bs,:].T).to(device)
        # lossval = loss(tau_pred, tau_true)
        
        #Backpropagation
               
        lossval.backward()
        with torch.no_grad():
            vhat = v - lr*v.grad
            coef = (proxL1norm(vhat,lr*lam)).clone().detach()

        loss_list.append(lossval.item())
    print("Average loss : " , torch.tensor(loss_list).mean().item())
    return coef, prevcoef, torch.tensor(loss_list).mean().item()


Epoch = 100
i = 1
lr = 1e-5 # alpha=learning_rate
lam = 0 # sparsity promoting parameter (l1 regularization)
temp = 1000
while(i<=Epoch):
    print("\n")
    print("Stage 1")
    print("Epoch "+str(i) + "/" + str(Epoch))
    print("Learning rate : ", lr)
    xi_L, prevxi_L, lossitem= training_loop(xi_L,prevxi_L,Zeta,Eta,Delta,Xdot,128,lr=lr,lam=lam,tau=Tau)
    # if(temp <=5e-3):
    #     break
    # if(temp <=1e-1):
    #     lr = 1e-5
    temp = lossitem
    i+=1


## Thresholding small indices ##
threshold = 1e-2
surv_index = ((torch.abs(xi_L) >= threshold)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
expr = np.array(expr)[surv_index].tolist()

xi_L =xi_L[surv_index].clone().detach().requires_grad_(True)
prevxi_L = xi_L.clone().detach()
#mask = torch.ones(len(expr),device=device)

## obtaining analytical model
xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=2)
L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-2)
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
    lam = lam/10
    lr = lr*2
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
        xi_L , prevxi_L, lossitem= training_loop(xi_L,prevxi_L,Zeta,Eta,Delta,Xdot,128,lr=lr,lam=lam,tau=Tau)
        temp = lossitem
        if(temp <=1e-3):
            break
        i+=1
    
    ## Thresholding small indices ##
    threshold = 1e-1
    surv_index = ((torch.abs(xi_L) >= threshold)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
    expr = np.array(expr)[surv_index].tolist()

    xi_L =xi_L[surv_index].clone().detach().requires_grad_(True)
    prevxi_L = xi_L.clone().detach()
    mask = torch.ones(len(expr),device=device)

    ## obtaining analytical model
    xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=3)
    L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-1)
    print("Result stage " + str(stage+2) + ":" , simplify(L))

if(save==True):
    #Saving Equation in string
    text_file = open(rootdir + "lagrangian_" + str(noiselevel)+ "_noise.txt", "w")
    text_file.write(L)
    text_file.close()

# --------------- Validation plots --------------------

# obtain equations of motion
x0 = dynamicsymbols(states[0])
x0_t = dynamicsymbols(states[0],1)
L = eval(str(L))
LM = LagrangesMethod(L, [x0])
LM.form_lagranges_equations()
eom = LM.rhs()[1]

#convert to string
eom = str(eom).replace('(t)','')

def predictedpendulum(t,x,a,b,omega):
    from numpy import sin, cos
    x0 = x[0]
    x0_t = x[1]
    return x0_t,eval(eom) + a*math.sin(omega*t) + b*math.cos(omega*t)

## Training results ##
# true results
q_tt_true = (Xdot[:500,states_dim//2:].T).copy()
q_t_true = (Xdot[:500,:states_dim//2].T).copy()
q_true = (X[:500,:states_dim//2].T).copy()

# prediction results
t = np.arange(0,5,0.01)
y_0 = X_wn[0,:]
Xpred, Xdotpred = generate_data(predictedpendulum, t, y_0, Tau_charact[0][0], Tau_charact[0][1], Tau_charact[0][2])
q_tt_pred = Xdotpred[:,states_dim//2:].T
q_t_pred = Xdotpred[:,:states_dim//2].T
q_pred = Xpred[:,:states_dim//2].T

## Validation Results ##
tval = np.arange(5,10,0.01)
y_0 = np.array([X_wn[499,0], X_wn[499,1]])
Xtest, Xdottest = generate_data(pendulum, tval, y_0, Tau_charact[0][0], Tau_charact[0][1], Tau_charact[0][2])

y_0 = np.array([q_pred[0,-1], q_t_pred[0,-1]])
Xtestpred, Xdottestpred = generate_data(predictedpendulum, tval, y_0, Tau_charact[0][0], Tau_charact[0][1], Tau_charact[0][2])

## Concatenate training and test data ##
# true results
t = np.concatenate((t, tval))
q_tt_true = np.concatenate((q_tt_true, Xdottest[:,states_dim//2:].T), axis=1)
q_t_true = np.concatenate((q_t_true, Xtest[:,states_dim//2:].T), axis=1)
q_true = np.concatenate((q_true,Xtest[:,:states_dim//2].T), axis=1)

# prediction results
q_tt_pred = np.concatenate((q_tt_pred,Xdottestpred[:,states_dim//2:].T), axis=1)
q_t_pred = np.concatenate((q_t_pred, Xtestpred[:,states_dim//2:].T), axis=1)
q_pred = np.concatenate((q_pred,Xtestpred[:,:states_dim//2].T), axis=1)

## Plotting
fig, ax = plt.subplots(3,1,figsize=(5,4))

ax[0].plot(t, q_tt_true[0,:], label='True Model')
ax[0].plot(t, q_tt_pred[0,:], 'r--',label='Predicted Model')
ax[0].set_ylabel('$\ddot{\\theta}$ (rad)')
ax[0].vlines(5,0,1,transform=ax[0].get_xaxis_transform(),colors='k')
ax[0].set_xlim([0,10])

ax[1].plot(t, q_t_true[0,:], label='True Model')
ax[1].plot(t, q_t_pred[0,:], 'r--',label='Predicted Model')
ax[1].set_ylabel('$\dot{\\theta}$ (rad)')
ax[1].vlines(5,0,1,transform=ax[1].get_xaxis_transform(),colors='k')
ax[1].set_xlim([0,10])

ax[2].plot(t, q_true[0,:], label='True Model')
ax[2].plot(t, q_pred[0,:], 'r--',label='Predicted Model')
ax[2].set_xlabel('Time (s)')
ax[2].set_ylabel('$\\theta$ (rad)')
ax[2].vlines(5,0,1,transform=ax[2].get_xaxis_transform(),colors='k')
ax[2].set_xlim([0,10])

Line, Label = ax[0].get_legend_handles_labels()
fig.legend(Line, Label, loc='upper right', bbox_to_anchor=(1.5, 0.98))

fig.tight_layout()