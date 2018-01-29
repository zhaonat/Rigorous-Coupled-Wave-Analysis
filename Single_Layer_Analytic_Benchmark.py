import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import cmath
from scipy.sparse import linalg
## analytic expressions for the transmissoin
## and reflection of a single dielectric slab

def fsr(ng, l1, theta, lambda_0):
    return lambda_0**2/(2*ng*l1*np.cos(theta)+lambda_0)

degrees = np.pi/180;
L0 = 1e-6; #units of microns;
eps0 = 8.854e-12;
mu0 = 4*np.pi-7;
c0 = 1/(np.sqrt(mu0*eps0))

## we can specify TE and TM angle depencies
#right now we assume theta = 0;

## reflection and transmission materials
eta_a = np.sqrt(mu0/eps0);
eta_b = np.sqrt(mu0/eps0);

# slab specifications
l1 = 0.1e-6;
e_r = 12; # dielectric constant of the slab
eta_1 = np.sqrt(mu0/eps0)*(1/e_r)**0.5

## define intermediate elementary coefs
rho1 = (eta_1-eta_a)/(eta_1+eta_a);
rho2 = (eta_b-eta_1)/(eta_b+eta_1);
tau1 = 1+rho1;
tau2 = 1+rho2;

im = cmath.sqrt(-1);
ref = list();
trans = list();
## specify wavelength range to do plotting
wavelengths = 1e-6*np.linspace(0.05, 0.1,1000)
for lam0 in wavelengths:
    k1 = 2*np.pi/lam0;
    R = rho1+rho2*np.exp(-2*im*k1*l1)/(1+rho1*rho2*np.exp(-2*im*k1*l1))
    T = tau1*tau2*np.exp(-im*k1*l1)/((1+rho1*rho2*np.exp(-2*im*k1*l1)))
    ref.append(abs(R)**2);
    trans.append(abs(T)**2);

lambda_0 = 0.075e-6;
theta = 0;
print('free spectral range: '+str(fsr(np.sqrt(e_r), l1, theta, lambda_0)))
plt.figure()
plt.plot(wavelengths*1e6, ref);
plt.plot(wavelengths*1e6, trans);
plt.xlabel('wavelength (um)')
plt.legend(('reflection', 'transmission'))
plt.show()