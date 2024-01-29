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
from datetime import datetime

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

def loss(pred, targ, coef):
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
        coef = coef.clone().detach().requires_grad_(True)
        prevcoef = coef.clone().detach()


        #Computing loss
        zeta = Zeta[:,:,:,i*bs:(i+1)*bs]
        eta = Eta[:,:,:,i*bs:(i+1)*bs]
        delta = Delta[:,:,i*bs:(i+1)*bs]
        x_t = xdot[i*bs:(i+1)*bs,:]

        #forward
        # tau_ = (tau[i*bs:(i+1)*bs,:].T).to(device)
        # q_tt_pred = lagrangianforward(v,zeta,eta,delta,x_t,device,tau=tau_)
        # q_tt_true = xdot[i*bs:(i+1)*bs,n//2:].T
        # lossval = loss(q_tt_pred, q_tt_true, coef)

        # tau_pred = ELforward(v, zeta, eta, delta, x_t, device)
        tau_pred = ELforward(coef, zeta, eta, delta, x_t, device)
        tau_true = (tau[i*bs:(i+1)*bs,:].T).to(device)
        lossval = loss(tau_pred, tau_true, coef)
        l1_norm = sum(abs(c) for c in coef)
        lossval = lossval + lam*l1_norm
        
        #Backpropagation
            
        lossval.backward()
        with torch.no_grad():
            # vhat = v - lr*v.grad
            # coef = (proxL1norm(vhat,lr*lam)).clone().detach()
            coef = coef - lr*coef.grad
            # coef[-1] = coef[-1]**2 + 0.0001

        loss_list.append(lossval.item())
    print("Average loss : " , torch.tensor(loss_list).mean().item())
    return coef, prevcoef, torch.tensor(loss_list).mean().item()

# Combinations to test
batch_size_list = [1000, 500, 200, 100]
lr_list = [1e-12, 1e-11, 1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6]
lambda_list = [0.1, 1, 2, 5]
lr_increments = [1, 2, 10]
lambda_increments = [1, 0.5, 0.1]

counter = 0
hyper_perm = []
for bs in batch_size_list:
    for lr in lr_list:
        for lam in lambda_list:
            for lr_inc in lr_increments:
                for lambda_inc in lambda_increments:
                    for seed in range(2):
                        hyper_perm.append([bs, lr, lam, lr_inc, lambda_inc, seed])


# Create log file
log_file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_ns-1_dof-1.csv"
with open(f"./Soft Robot/results/{log_file_name}",'w') as log_file:

    # Create logging file
    log_writer = csv.writer(log_file)
    log_writer.writerow(
        [
            'batch_size',
            'learning_rate',
            'lambda',
            'lr_increment',
            'lambda_increment',
            'seed',
            'loss_stage_1',
            'L_stage_1',
            'loss_stage_2',
            'L_stage_2',
            'loss_stage_3',
            'L_stage_3'
        ]
    )

    counter = 0
    for experiment in hyper_perm:
                 
        bs = experiment[0]
        lr = experiment[1]
        lam = experiment[2]
        lr_inc = experiment[3]
        lambda_inc = experiment[4]
        seed = experiment[5]

        counter += 1
        print('\n')
        print(f"Experiment progress: {counter}/3168")

        #build function expression for the library in str
        expr= HL.buildFunctionExpressions(2,states_dim,states,use_sine=False)
        expr.pop(3)
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
        torch.manual_seed(seed)
        xi_L = torch.ones(len(expr), device=device).data.uniform_(-1,1)
        # xi_L = torch.tensor((-0.06595270528, -6.283185305, 0.0002241), device=device)
        # xi_L = torch.tensor((-0.08, -6.7, 0.0008), device=device)
        prevxi_L = xi_L.clone().detach()

        Epoch = 100
        i = 1
        lossitem_prev = 1e8
        loss_streak = 0
        while(i<=Epoch):
            print("\n")
            print("Stage 1")
            print("Epoch "+str(i) + "/" + str(Epoch))
            print("Batch size : ", bs)
            print("Learning rate : ", lr)
            print("Lambda : ", lam)
            xi_L, prevxi_L, lossitem= training_loop(xi_L,prevxi_L,Zeta,Eta,Delta,Xdot,bs=bs,lr=lr,lam=lam,tau=Tau)
            if(lossitem <=5e-3):
                break
            if (lossitem - lossitem_prev) > -1e-3:
                loss_streak+=1
            if loss_streak == 6:
                break
            if math.isnan(lossitem) or math.isinf(lossitem):
                break
            lossitem_prev = lossitem
            i+=1

        if math.isnan(lossitem) or math.isinf(lossitem):
            rowlist = [
                bs,
                lr,
                lam,
                lr_inc,
                lambda_inc,
                seed,
                '',
                '',
                '',
                '',
                '',
                ''
            ]
            log_writer.writerow(rowlist)
        else:

            ## Thresholding small indices ##
            threshold = 1e-3
            surv_index = ((torch.abs(xi_L[:-1]) >= threshold)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
            surv_index = np.append(surv_index, xi_L.shape[0]-1)
            expr = np.array(expr)[surv_index].tolist()

            xi_L =xi_L[surv_index].clone().detach().requires_grad_(True)
            prevxi_L = xi_L.clone().detach()

            ## obtaining analytical model
            xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=6)
            L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-6)

            rowlist = [
                bs,
                lr,
                lam,
                lr_inc,
                lambda_inc,
                seed,
                lossitem,
                simplify(L)
            ]

            ## Next round Selection ##
            lr_next = lr
            lam_next = lam
            for stage in range(2):
                
                #Redefine computation after thresholding
                Zeta, Eta, Delta = LagrangianLibraryTensor(X,Xdot,expr,states,states_dot,scaling=False)
                Eta = Eta.to(device)
                Zeta = Zeta.to(device)
                Delta = Delta.to(device)

                #Training
                Epoch = 100
                i = 1
                lam_next = lam_next*lambda_inc
                lr_next = lr_next*lr_inc
                lossitem_prev = 1e8
                loss_streak = 0
                while(i<=Epoch):
                    print("\n")
                    print("Stage " + str(stage+2))
                    print("Epoch "+str(i) + "/" + str(Epoch))
                    print("Batch size : ", bs)
                    print("Learning rate : ", lr_next)
                    print("Lambda : ", lam_next)
                    xi_L , prevxi_L, lossitem= training_loop(xi_L,prevxi_L,Zeta,Eta,Delta,Xdot,bs=bs,lr=lr_next,lam=lam_next,tau=Tau)
                    if(lossitem <=1e-3):
                        break
                    if (lossitem - lossitem_prev) > -1e-3:
                        loss_streak+=1
                    if loss_streak == 6:
                        break
                    if math.isnan(lossitem) or math.isinf(lossitem):
                        break
                    lossitem_prev = lossitem
                    i+=1
                
                if math.isnan(lossitem) or math.isinf(lossitem):
                    rowlist.append('')
                    rowlist.append('')
                else:
                    ## Thresholding small indices ##
                    threshold = 1e-3
                    surv_index = ((torch.abs(xi_L[:-1]) >= threshold)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
                    surv_index = np.append(surv_index, xi_L.shape[0]-1)
                    expr = np.array(expr)[surv_index].tolist()

                    xi_L =xi_L[surv_index].clone().detach().requires_grad_(True)
                    prevxi_L = xi_L.clone().detach()

                    ## obtaining analytical model
                    xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=6)
                    L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-6)

                    rowlist.append(lossitem)
                    rowlist.append(simplify(L))

            log_writer.writerow(rowlist)

