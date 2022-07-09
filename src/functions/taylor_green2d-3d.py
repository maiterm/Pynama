from math import exp, sin, cos, pi
import numpy as np 

def alpha(nu, t):
    Uref = 1
    Lx = 1
    Ly = 1
    return  Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2))

def velocity(coord, alpha):
    vel = np.zeros(coord.shape)
    Lx= 1
    Ly= 1
    x_ = 2 * pi * coord[:,0] / Lx
    y_ = 2 * pi * coord[:,1] / Ly
    vel[:,0] = np.cos(x_) * np.sin(y_) *Lx* alpha 
    vel[:,1] = np.sin(x_) * np.cos(y_) *Ly *alpha 
    vel[:,2] = 0
    return vel.flatten()

def vorticity(coord, alpha):
    vort = np.zeros(coord.shape)
    Lx, Ly= (1, 1)
    x_ = 2 * pi * coord[:,0] / Lx
    y_ = 2 * pi * coord[:,1] / Ly
    vort[:,0] = 0
    vort[:,1] = 0 
    vort[:,2] =  -2 * pi * (Ly / Lx - Lx / Ly) * np.cos(x_) * np.cos(y_) * alpha 
    return vort.flatten()