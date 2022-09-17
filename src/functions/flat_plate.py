from math import pi, sin, cos, exp, sqrt
import numpy as np
from scipy.special import  erf

    # @staticmethod
    # def flatplateVel(coord, nu , t=None):

def alpha(nu, t):
    """tau"""
    return  sqrt(4*nu*t)

def velocity(coord, tau):
    vel = np.zeros(coord.shape)
    U_ref = 1
    vel[:,0] = U_ref * erf(coord[:,1]/ tau)
    vel[:,1] = 0
    vel[:,2] = 0 #U_ref * erf(coord[:,1]/ tau)
    return vel.flatten()

    # @staticmethod
    # def flatplateVort(coord, nu, t=None):
def vorticity(coord, tau):
    vort = np.zeros(coord.shape)
    #if tau ==0 : 
    #    vort[:,0] = 0
    #    vort[:,1] = 0
    #    vort[:,2] = 0
    #    return vort.flatten()
    vort[:,0] = 0 # (-2/(tau * sqrt(pi))) * exp(-(coord[:,1]/tau)**2)
    vort[:,1] = 0
    vort[:,2] = (-2/(tau * sqrt(pi))) * np.exp(-(coord[:,1]/tau)**2)
    return vort.flatten()


#@staticmethod
def flatplateConvective(coord, nu, t=None):
    c = 1
    tau = sqrt(4*nu*t)
    alpha = 4 * c * coord[1] / ( sqrt(pi) * tau**3 )
    convective = alpha * exp( -(coord[1]/tau)**2 )
    return [convective]

#@staticmethod
def flatplateDiffusive(coord, nu, t=None):
    tau = sqrt(4*nu*t)
    alpha = 4 / (sqrt(pi)* tau**3)
    beta = ( 1 - 2 * coord[1]**2 / tau**2 )
    diffusive = nu * alpha * beta * exp( -(coord[1]/tau)**2 )
    return [diffusive]