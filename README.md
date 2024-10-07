# Dynamic Regression
This repository is based on the Extended Lagrangian SINDy (xL-SINDy) (check the original [here](https://github.com/AdamPurnomo/Extended-Lagrangian-SINDy-xL-SINDy-)). xL-SINDy is a learning based algorithm designed to obtain Lagrangian function of nonlinear dynamical systems from noisy measurement data. The Lagrangian function is modeled as a linear combination of nonlinear candidate functions, and Euler-Lagrange’s equation is used to formulate the objective cost function.

The approach in this repository deviates from xL-SINDy because it doesn't use a "random" library of candidate nonlinear functions nor resorts to sparsifying that library. Instead, it uses all the basis functions that the Lagrangian of a planar PCS soft robot parametrization contains. The Euler-Lagrange’s equation is still used to formulate the objective cost function, and a normal regression is performed by applying least-squares to obtain the optimal coefficients.

## Installation
* Clone this repository
* Install dependencies (see below)

## How to Use
The two main folders are `Source/Soft Robot` and `Source/Python Script`.
1) `Source/Soft Robot` stores all the datasets used for the Dynamic Regression. The folders state the type of soft robot used in the simulation to obtain those datasets (e.g `Source/Soft Robot/ns-1_dof-3`). The datasets include three files: `X.npy` hold the data for (strain) position and velocity; `Xdot.npy` contain data for (strain) velocity and acceleration; `Tau.npy` have the actuation data. 
`Source/Soft Robot/symbolic_expressions` contain the files with the basis functions for the planar PCS robots.
`Source/Soft Robot/render_data` contain variables for post-processing plots.

2) `Source/Python Script` contain the scripts used in the Dynamic Regression. The **main script** is `Source/Python Script/SoftRobot_DynamicRegression.py` (the other soft robot scripts are deprecated versions, and the other scripts for different systems are from the original repository). Run this file to do the dynamic regression. Auxiliary functions used in this script are in the `Source/Python Script/xLSINDy.py` file and `Source/Python Script/utils` folder.

## Dependencies
* numpy 1.19.2
* scipy 1.6.1
* pytorch 1.9.0
* sympy 1.7.1
