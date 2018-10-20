import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import matplotlib.pyplot as plt;
import cmath;
from TMM_functions import run_TMM_simulation as rTMM

## GOAL: DEMONSTRATE SCATTERING MATRICES WORK BY SIMULATING A METAL SLAB

#%% DEFINE SIMULATION PARAMETers
#% General Units
degrees = np.pi/180;
L0 = 1e-6; #units of microns;
eps0 = 8.854e-12;
mu0 = 4*np.pi*10**-7;
c0 = 1/(np.sqrt(mu0*eps0))
I = np.matrix(np.eye(2,2)); #unit 2x2 matrix


## normalized units
#z' = k0*z;
#k = k/k0;

## REFLECTION AND TRANSMSSION SPACE epsilon and mu PARAMETERS
m_r = 1; e_r = 1; incident_medium = [e_r, m_r];
m_t = 1; e_t = 1; transmission_medium = [e_t, m_t];

## set wavelength scanning range
wavelengths = L0*np.linspace(0.2,2,1000); #500 nm to 1000 nm
kmagnitude_scan = 2 * np.pi / wavelengths; #no
omega = c0 * kmagnitude_scan; #using the dispersion wavelengths

#source parameters
theta = 0 * degrees; #%elevation angle
phi = 0 * degrees; #%azimuthal angle

## incident wave properties, at this point, everything is in units of k_0
n_i = np.sqrt(e_r*m_r);

#k0 = np.sqrt(kx**2+ky**2+kz**2); we know k0, theta, and phi

#actually, in the definitions here, kx = k0*sin(theta)*cos(phi), so kx, ky here are normalized
kx = n_i*np.sin(theta)*np.cos(phi); #constant in ALL LAYERS; kx = 0 for normal incidence
ky = n_i*np.sin(theta)*np.sin(phi); #constant in ALL LAYERS; ky = 0 for normal incidence
kz_inc = cmath.sqrt(e_r * m_r - kx ** 2 - ky ** 2);

normal_vector = np.array([0, 0, -1]) #positive z points down;
ate_vector = np.matrix([0, 1, 0]); #vector for the out of plane E-field
#ampltidue of the te vs tm modes (which are decoupled)
pte = 1; #1/np.sqrt(2);
ptm = 0; #cmath.sqrt(-1)/np.sqrt(2);
polarization_amplitudes = [pte, ptm]
k_inc = [kx, ky];
print('--------incident wave paramters----------------')
print('incident n_i: '+str(n_i))
print('kx_inc: '+str(kx)+' ky_inc: '+str(ky))
print('kz_inc: ' + str(kz_inc));
print('-----------------------------------------------')

omega_p = 3e15; lambda_p = 2*np.pi*c0/omega_p;
gamma = 5.5e12;

drude_eps = 1 - omega_p**2/(omega**2 + cmath.sqrt(-1)*omega*gamma)
ER = np.array([drude_eps]);
UR = np.ones_like(ER);
layer_thicknesses = [0.3*L0]
## run simulation
Ref, Tran = rTMM.run_TMM_dispersive(wavelengths, polarization_amplitudes, theta, phi, ER, UR, layer_thicknesses,\
                       transmission_medium, incident_medium)

plt.figure();
plt.plot(wavelengths/L0, Ref);
plt.plot(wavelengths/L0, Tran);
plt.show();