'''written by Adam Purnomo'''

import HLsearch as HL
import numpy as np

from sympy import symbols, simplify, derive_by_array, ordered

import sympy
import torch
import sys
sys.path.append(r'../../HLsearch/')

def EulerLagrangeExpressionTensor(expr, states, states_epsed_sym=None):
    """
    A function dedicated to build a tensor with the expressions which are present on the Euler-Lagrange equation.
    The lagrangian equation is described as follow
    L = sum(c_k*phi_k)
    tau = (D^2L_qdot2)*q_tt + (D^2L_qdotq)*q_t - DL_q 

    #Params:
    expr                    : list of basis function (str) (d,)
    states                  : list states variable description (str) (n,)
    states_epsed            : list states variable with a small value added to avoid singularity (this just applies to bending) (str) (n,)
    states_epsed sym        : list of symbolic states variable with a small value added to avoid singularity (this just applies to bending) (sympy) (n,)

    #Return:
    phi_q                   : derivative of basis functions w.r.t q
    phi_qdot2               : double derivative of basis functions w.r.t qdot and qdot  
    phi_qdotq               : double derivative of basis functions w.r.t qdot and q
    """
    n = len(states)

    #define symbolically
    q = sympy.Array(np.array(sympy.Matrix(states[:n//2])).squeeze().tolist())
    qdot = sympy.Array(np.array(sympy.Matrix(states[n//2:])).squeeze().tolist())
    phi = sympy.Array(np.array(sympy.Matrix(expr)).squeeze().tolist())
    phi_q = derive_by_array(phi, q).reshape(n//2, len(phi)) #Delta
    phi_qdot = derive_by_array(phi, qdot).reshape(n//2, len(phi))
    phi_qdot2 = derive_by_array(phi_qdot, qdot).reshape(n//2, n//2, len(phi)) #Zeta
    phi_qdotq = derive_by_array(phi_qdot, q).reshape(n//2, n//2, len(phi)) #Eta

    if states_epsed_sym != None:
        def len_symbol(e):
            return len(str(e))
        symbols = list(ordered(list(phi_q.free_symbols), keys=len_symbol)) # x0, x1, x2, ..., x0_t, x1_t, x2_t, ...

        # The entries corresponding to the Elastic potential basis functions 
        # should be kept as original configuration variables (not epsed) - these entries are the last
        # (n//2) entries from the expr list
        phi_q_not_epsed = phi_q[:,-(n//2):]
        phi_qdot2_not_epsed = phi_qdot2[:,:,-(n//2):]
        phi_qdotq_not_epsed = phi_qdotq[:,:,-(n//2):]
        
        # Replace the configuration variables by the epsed configuration variables
        for i in range(n//2):
            phi_q = phi_q.subs([
                (symbols[i], states_epsed_sym[i]),
            ])
            phi_qdot2 = phi_qdot2.subs([
                (symbols[i], states_epsed_sym[i]),
            ])
            phi_qdotq = phi_qdotq.subs([
                (symbols[i], states_epsed_sym[i]),
            ])

        phi_q_assemble = np.empty(phi_q.shape, dtype=object)
        phi_q_assemble[:,:-(n//2)] = phi_q[:,:-(n//2)]
        phi_q_assemble[:,-(n//2):] = phi_q_not_epsed
        phi_q = phi_q_assemble

        phi_qdot2_assemble = np.empty(phi_qdot2.shape, dtype=object)
        phi_qdot2_assemble[:,:,:-(n//2)] = phi_qdot2[:,:,:-(n//2)]
        phi_qdot2_assemble[:,:,-(n//2):] = phi_qdot2_not_epsed
        phi_qdot2 = phi_qdot2_assemble
        # phi_qdot2 = np.transpose(phi_qdot2, (1,0,2))

        phi_qdotq_assemble = np.empty(phi_qdotq.shape, dtype=object)
        phi_qdotq_assemble[:,:,:-(n//2)] = phi_qdotq[:,:,:-(n//2)]
        phi_qdotq_assemble[:,:,-(n//2):] = phi_qdotq_not_epsed
        phi_qdotq = phi_qdotq_assemble
        # phi_qdotq = np.transpose(phi_qdotq, (1,0,2))

    return phi_q, phi_qdot2, phi_qdotq

def LagrangianLibraryTensor(x, xdot, x_epsed, states, states_dot, states_epsed, phi_q, phi_qdot2, phi_qdotq, scaling=False):
    """
    A function dedicated to build time-series tensor for the lagrangian equation.
    The lagrangian equation is described as follow
    L = sum(c_k*phi_k)
    q_tt = (D^2L_qdot2)^-1*(tau + DL_q - D^2L_qdotq)

    #Params:
    x                       : values of state variables in torch tensor. In [x,x_dot] format. Each row presents states at one time
    xdot                    : values of states_dot variables in torch tensor. In [x_dot,x_doubledot] format. Each row presents states at one time
    expr                    : list of basis function (str) (d,)
    states                  : list states variable description (str) (n,)
    states_dot              : time derivative state_variable (str) (n,)

    #Return:
    Zeta                    : time-series of double derivative of basis functions w.r.t qdot and qdot  
    Eta                     : time-series of double derivative of basis functions w.r.t qdot and q 
    Delta                   : time-series of derivative of basis functions w.r.t q 
    """
    from torch import cos, sin
    x = torch.from_numpy(x)
    xdot = torch.from_numpy(xdot)
    x_epsed = torch.from_numpy(x_epsed)

    i, j, k = np.array(phi_qdot2).shape
    l = x.shape[0]
    Delta = torch.ones(j, k, l, dtype=torch.float64)
    Zeta = torch.ones(i, j, k, l, dtype=torch.float64)
    Eta = torch.ones(i, j, k, l, dtype=torch.float64)

    for idx in range(len(states)):
        locals()[states[idx]] = x[:, idx]
        locals()[states_dot[idx]] = xdot[:, idx]
    
    for idx in range(len(states_epsed)):
        locals()[states_epsed[idx]] = x_epsed[:,idx]

    for n in range(j):
        for o in range(k):
            delta = eval(str(phi_q[n, o]))
            if(isinstance(delta, int)):
                Delta[n, o, :] = delta*Delta[n, o, :]
            else:
                # Feature Scaling
                if(scaling == True):
                    scales = torch.max(delta) - torch.min(delta)
                    delta = delta/scales
                Delta[n, o, :] = delta

    for m in range(i):
        for n in range(j):
            for o in range(k):
                zeta = eval(str(phi_qdot2[m, n, o]))
                eta = eval(str(phi_qdotq[m, n, o]))

                if(isinstance(zeta, int)):
                    Zeta[m, n, o, :] = zeta*Zeta[m, n, o, :]
                else:
                    # Feature Scaling
                    if(scaling == True):
                        scales = torch.max(zeta) - torch.min(zeta)
                        zeta = zeta/scales
                    Zeta[m, n, o, :] = zeta

                if(isinstance(eta, int)):
                    Eta[m, n, o, :] = eta*Eta[m, n, o, :]
                else:
                    # Feature Scaling
                    if(scaling == True):
                        scales = torch.max(eta) - torch.min(eta)
                        eta = eta/scales
                    Eta[m, n, o, :] = eta
    return Zeta, Eta, Delta


def lagrangianforward(coef, Zeta, Eta, Delta, xdot, device, tau, D):
    """
    Computing time series of q_tt (q double dot) prediction
    #Params:
    coef        : Coefficient corresponding to each basis function
    mask        : filter for coefficient below a certain threshold
    Zeta        : time-series of double derivative of basis functions w.r.t qdot and qdot  
    Eta         : time-series of double derivative of basis functions w.r.t qdot and q 
    Delta       : time-series of derivative of basis functions w.r.t q 
    xdot        : Time-series of states_dot data  
    """
    weight = coef
    DL_q = torch.einsum('jkl,k->jl', Delta, weight)
    DL_qdot2 = torch.einsum('ijkl,k->ijl', Zeta, weight)
    DL_qdotq = torch.einsum('ijkl,k->ijl', Eta, weight)
    n = xdot.shape[1]

    if(torch.is_tensor(xdot) == False):
        xdot = torch.from_numpy(xdot).to(device).float()
    if(torch.is_tensor(tau) == False):
        tau = torch.from_numpy(tau).to(device).float()
    q_t = xdot[:, :n//2].T

    C = torch.einsum('ijl,il->jl', DL_qdotq, q_t)
    B = DL_q
    A = torch.einsum('ijl->lij', DL_qdot2)
    Tau_NC = torch.einsum('ij,il->jl', -D, q_t)
    invA = torch.linalg.pinv(A)
    invA = torch.einsum('lij->ijl', invA)
    q_tt = torch.einsum('ijl,jl->il', invA, tau + Tau_NC + B - C)

    return q_tt


def ELforward(coef, Zeta, Eta, Delta, xdot, device, D):
    """
    Computing time series of total sum of Euler-Lagrange equation
    #Params:
    coef        : Coefficient corresponding to each basis function
    Zeta        : time-series of double derivative of basis functions w.r.t qdot and qdot  
    Eta         : time-series of double derivative of basis functions w.r.t qdot and q 
    Delta       : time-series of derivative of basis functions w.r.t q 
    xdot        : Time-series of states_dot data  

    #Returns:
    El          : Time series of the left hand side of Euler's Lagranges equation (n, time-series)
    """

    weight = coef.to(device)

    DL_q = torch.einsum('jkl,k->jl', Delta, weight)
    DL_qdot2 = torch.einsum('ijkl,k->ijl', Zeta, weight)
    DL_qdotq = torch.einsum('ijkl,k->ijl', Eta, weight)
    n = xdot.shape[1]

    if(torch.is_tensor(xdot) == False):
        xdot = torch.from_numpy(xdot).to(device)
    q_t = xdot[:, :n//2].T
    q_tt = xdot[:, n//2:].T

    C = torch.einsum('ijl,il->jl', DL_qdotq, q_t)
    B = DL_q
    A = torch.einsum('ijl,il->jl', DL_qdot2, q_tt)
    Tau_NC = torch.einsum('ij,il->jl', -D, q_t)
    EL = A + C - B - Tau_NC
    return EL#, DL_q, DL_qdot2, DL_qdotq, A, C, B, Tau_NC


def Upsilonforward(Zeta, Eta, Delta, xdot, device):
    """
    Computing time series of total sum of Euler-Lagrange equation
    #Params:
    Zeta        : time-series of double derivative of basis functions w.r.t qdot and qdot  
    Eta         : time-series of double derivative of basis functions w.r.t qdot and q 
    Delta       : time-series of derivative of basis functions w.r.t q 
    xdot        : Time-series of states_dot data  

    #Returns:
    Upsilon          : Time series of the left hand side of Euler's Lagranges equation before multiplied with weight (n, time-series)
    """
    n = xdot.shape[1]

    if(torch.is_tensor(xdot) == False):
        xdot = torch.from_numpy(xdot).to(device).float()
    q_t = xdot[:, :n//2].T
    q_tt = xdot[:, n//2:].T

    A = torch.einsum('ijkl,il->jkl', Zeta, q_tt)
    B = torch.einsum('ijkl,il->jkl', Eta, q_t)
    C = Delta

    Upsilon = A + B - C
    return Upsilon


def tauforward(coef, Zeta, Eta, Delta, xdot, device):
    '''
    Computing time series of tau (external input) prediction
    #Params:
    coef        : Coefficient corresponding to each basis function
    mask        : filter for coefficient below a certain threshold
    Zeta        : time-series of double derivative of basis functions w.r.t qdot and qdot  
    Eta         : time-series of double derivative of basis functions w.r.t qdot and q 
    Delta       : time-series of derivative of basis functions w.r.t q 
    xdot        : Time-series of states_dot data  
    '''
    weight = coef
    DL_q = torch.einsum('jkl,k->jl', Delta, weight)
    DL_qdot2 = torch.einsum('ijkl,k->ijl', Zeta, weight)
    DL_qdotq = torch.einsum('ijkl,k->ijl', Eta, weight)

    if(torch.is_tensor(xdot) == False):
        xdot = torch.from_numpy(xdot).to(device).float()
    q_t = xdot[:, :2].T
    q_tt = xdot[:, 2:].T

    C = torch.einsum('ijl,il->jl', DL_qdotq, q_t)
    B = DL_q
    A = torch.einsum('ijl,il->jl', DL_qdot2, q_tt)
    tau = A + C - B
    return tau


def SymGradient(func_description, q):
    '''
    Symbolic gradient of list of basis function w.r.t quantity q where q is subset of the states (can be position, velocity or acceleration)
    #Params:
    func_description    : list of basis functions (str) (d,)
    q                   : list of a quantity subset of the states (str) (d,)

    #Retuns:
    dfunc_dq                : gradient of basis functions w.r.t q (sympy matrix) (d,n)
    '''
    q = sympy.Matrix(q)
    func_description = sympy.Matrix(func_description)
    dfunc_dq = simplify(func_description.jacobian(q))
    return dfunc_dq


def TimeDerivativeSym(func_description, states, states_dot):
    '''
    Symbolic time derivative of basis function

    #Params:
    func_description     : list basis functions (str) (d,)
    states               : list states variable description (str) (n,)
    states_dot           : time derivative state_variable (str) (n,)

    #Return
    dfunc_dt             : symbolic time derivative of basis functions list (sympy matrix) (d,)
    '''
    func = sympy.Matrix(func_description)
    x = sympy.Matrix(states)
    x_dot = sympy.Matrix(states_dot)
    grad = func.jacobian(x)
    dfunc_dt = grad*x_dot
    return dfunc_dt


def TimeDerivativeSymGradient(gradfunc_description, states, states_dot):
    '''
    Symbolic time derivative of gradient of basis function w.r.t. quantity q which is a subset of the states

    #Params:
    gradfunc_description : gradient of basis function w.r.t. quantity q (sympy matrix) (d,n)
    states               : list states variable description (str) (n,)
    states_dot           : time derivative state_variable (str) (n,)


    #Return
    dgradfunc_description_dt : Symbolic time derivative of gradient of basis function w.r.t. quantity q (sympy matrix) (d,n)
    '''
    x = sympy.Matrix(states)
    x_dot = sympy.Matrix(states_dot)

    temp = gradfunc_description[:, 0].jacobian(x)*x_dot
    for i in range(1, len(states)//2):
        temp = temp.row_join(gradfunc_description[:, i].jacobian(x)*x_dot)
    dgradfunc_description_dt = temp
    return dgradfunc_description_dt


def SymVectorFuncSumOverStates(matrix_func):
    '''
    Sum of gradient of symbolic basis function over states
    #Params
    matrix_fun : gradient of symbolic basis function (sympy matrix) (d,n)

    #Return
    Sigma      : sum of gradeitn of symbolic basis function over states (sympy matrix) (d) 
    '''

    p, m = matrix_func.shape
    sigma = matrix_func[:, 0]
    for i in range(1, m):
        sigma += matrix_func[:, i]
    return sigma


def timeDerivativeLibraryMatrix(x, xdot, function_description, states, states_dot):
    """
    #Params:
    x                       : values of state variables in torch tensor. In [x,x_dot] format. Each row presents states at one time
    xdot                    : values of states_dot variables in torch tensor. In [x_dot,x_doubledot] format. Each row presents states at one time
    function_description    : list of basis functions (str) (d,)
    states                  : list states variable description (str) (n,)
    states_dot              : time derivative state_variable (str) (n,)

    #Return:
    time-series of time-derivative functions in torch.tensor
    """
    df_dt = TimeDerivativeSym(function_description, states, states_dot)
    df_dt = [str(f) for f in df_dt]
    from torch import cos, sin
    if((torch.is_tensor(x) == False) or (torch.is_tensor(xdot) == False)):
        x = torch.from_numpy(x)
        xdot = torch.from_numpy(xdot)

    column = []
    n = len(states)
    # Assign data to states and states dot
    for j in range(n):
        locals()[states[j]] = x[:, j]
        locals()[states_dot[j]] = xdot[:, j]
    # evaluate each function in function expression with data
    for func in df_dt:
        column.append(eval(func))
    column = torch.stack(column)
    column = column.T
    return column


def LibraryMatrix(x, function_description, states, scaling=True):
    """
    #Params:
    x                       : values of variables in torch tensor. In [x,x_dot] format. Each row presents states at one time
    function_description    : list of basis functions (str) (d,)
    states                  : symbolic states' names (str)

    #Return:
    time-serie of calculated functions in torch.tensor
    """

    from torch import cos, sin
    if(torch.is_tensor(x) == False):
        x = torch.from_numpy(x)

    column = []
    n = len(states)
    # Assign data to data_description (states)
    for j in range(n):
        locals()[states[j]] = x[:, j]
    # evaluate each function in function expression with data
    for func in function_description:
        k = eval(func)
        if(isinstance(k, int)):
            column.append(k*torch.ones(x.shape[0]))
        else:
            # Feature Scaling
            if(scaling == True):
                scales = torch.max(k) - torch.min(k)
                k = k/scales
            column.append(k)
    column = torch.stack(column)
    column = column.T
    return column


def timeDerivativeLibraryTensor(x, xdot, matrix_func, states, states_dot):
    """
    #Params:
    x                       : values of state variables in torch tensor. In [x,x_dot] format. Each row presents states at one time
    xdot                    : values of states_dot variables in torch tensor. In [x_dot,x_doubledot] format. Each row presents states at one time
    matrix_func             : matrix of basis functions (str) (d,n)
    states                  : list states variable description (str) (n,)
    states_dot              : time derivative state_variable (str) (n,)

    #Return:
    time-series of time-derivative functions in torch.tensor
    """
    from torch import cos, sin
    if((torch.is_tensor(x) == False) or (torch.is_tensor(xdot) == False)):
        x = torch.from_numpy(x)
        xdot = torch.from_numpy(xdot)

    d, n = matrix_func.shape[0], len(states)
    b = x.shape[0]
    Eta = torch.ones(d, n//2, b)

    # Assign data to states and states dot
    for j in range(n):
        locals()[states[j]] = x[:, j]
        locals()[states_dot[j]] = xdot[:, j]

    # evaluate each function in function expression with data
    for i in range(matrix_func.shape[0]):
        for j in range(matrix_func.shape[1]):
            k = eval(str(matrix_func[i, j]))
            if(isinstance(k, int)):
                Eta[i, j, :] = k*Eta[i, j, :]
            else:
                Eta[i, j, :] = k
    return Eta
