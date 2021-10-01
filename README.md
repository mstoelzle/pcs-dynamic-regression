# Extended-Lagrangian-SINDy-xL-SINDy-

## Overview
Extended Lagrangian SINDy (xL-SINDy) is an algorithm designed to obtain Lagrangian function of nonlinear dynamical systems from noisy measurement data. The Lagrangian function is modelled as a linear combination of nonlinear candidate functions, and Euler-
Lagrange’s equation is used to formulate the objective cost function. The optimization of the learning process is done with proximal gradient method. For more detail about the derivation and the problem formulation, please take a look at this paper (in the review process for RAL with ICRA option 2022). [Sparse Identification of Lagrangian for Nonlinear Dynamical Systems via Proximal Gradient Method](https://drive.google.com/file/d/14FqbwIONE2wfZJqi2hgxNPylV5eYjNQv/view?usp=sharing)

![overview](/images/overview.png)



## Examples
The effectiveness of xL-SINDy  is demonstrated against different noise levels in physical simulation with four dynamical systems: A single pendulum, a cart-pendulum, a double pendulum, and a spherical pendulum.

<p align="center">
  <img width=25% height=25% src="https://github.com/AdamPurnomo/Extended-Lagrangian-SINDy-xL-SINDy-/blob/main/images/systems.png?raw=true">
</p>

<p align="center">
  <img width=75% height=75% src="https://raw.githubusercontent.com/AdamPurnomo/Extended-Lagrangian-SINDy-xL-SINDy-/main/images/resfull.png">
</p>

## Installation
* Clone this repository `git clone https://github.com/AdamPurnomo/Extended-Lagrangian-SINDy-xL-SINDy-.git`
* Install the environment and dependencies `conda env create -f environment.yml`

## How to Use
Please take a look at `Source/Notebook/[System Name] - Train.ipynb` or `Source/Python Script/[System Name]_Train.py` and run the code.

## Dependencies
* numpy 1.19.2
* scipy 1.6.1
* pytorch 1.9.0
* sympy 1.7.1
