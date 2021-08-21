from math import exp, sin, cos, pi
import numpy as np


    # @staticmethod
    # def flatplateVel(coord, nu , t=None):
def velocity(coords, nu,t=None):
    U_ref = 1
    vx = U_ref * erf(coord[1]/ sqrt(4*nu*t))
    vz = U_ref * erf(coord[1]/ sqrt(4*nu*t))
    vy = 0
    return [vx, vy,vz]

    # @staticmethod
    # def flatplateVort(coord, nu, t=None):
def vorticity(coord, nu, t=None):
    tau = sqrt(4*nu*t)
    vort = (-2/(tau * sqrt(pi))) * exp(-(coord[1]/tau)**2)
    return [vort,0,-vort]

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