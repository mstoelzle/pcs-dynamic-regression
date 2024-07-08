import numpy as np
import sympy
import HLsearch as HL
from itertools import chain
from sympy import symbols, simplify, derive_by_array, ordered, poly

def B_decomp(expr, xi_sym, params):
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
    num_segments = params['num_segments']

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
    if p == None: # p is only a constant
        coeffs = [expr]
        monoms = [1]
    else:
        coeffs = p.coeffs()
        monoms = [sympy.prod(x**k for x, k in zip(p.gens, mon)) for mon in p.monoms()]
    return coeffs, monoms

def U_decomp(expr, xi_sym, params):
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
    num_segments = params['num_segments']

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

def constructLagrangianExpression(sym_exps, states_sym, states_epsed_sym, xi_eq, B_xi, strain_selector, params):
    true_coeffs = []
    expr = []
    num_segments = params['num_segments']
    n_dof = np.sum(strain_selector)

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
                coeffs, monoms = B_decomp(B_entry, xi_sym, params)
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
    coeffs, monoms = U_decomp(U, xi_sym, params)
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