'''
TMM applied to a single uniform layer
should recover the analytic fabry perot solution
'''

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import matplotlib.pyplot as plt;
import cmath;
from TMM_functions import run_TMM_simulation as rTMM

## GOAL: simulate a BRAGG MIRROR at some wavelength (1 micron)

#%% DEFINE SIMULATION PARAMETers
#% General Units
degrees = np.pi/180;
L0 = 1e-6; #units of microns;
eps0 = 8.854e-12;
mu0 = 4*np.pi*10**-7;
c0 = 1/(np.sqrt(mu0*eps0))

## normalized units
#z' = k0*z;
#k = k/k0;

## REFLECTION AND TRANSMSSION SPACE epsilon and mu PARAMETERS
m_r = 1; e_r = 1; incident_medium = [e_r, m_r];
m_t = 1; e_t = 1; transmission_medium = [e_t, m_t];

## set wavelength scanning range
wavelengths = np.linspace(0.5,1.6,500); #500 nm to 1000 nm
kmagnitude_scan = 2 * np.pi / wavelengths; #no
omega = c0 * kmagnitude_scan; #using the dispersion wavelengths

#source parameters
theta = 0 * degrees; #%elevation angle; #off -normal incidence does not excite guided resonances...
phi = 0 * degrees; #%azimuthal angle

## incident wave properties, at this point, everything is in units of k_0
n_i = np.sqrt(e_r*m_r);

#k0 = np.sqrt(kx**2+ky**2+kz**2); we know k0, theta, and phi

#actually, in the definitions here, kx = k0*sin(theta)*cos(phi), so kx, ky here are normalized
kx = n_i*np.sin(theta)*np.cos(phi); #constant in ALL LAYERS; kx = 0 for normal incidence
ky = n_i*np.sin(theta)*np.sin(phi); #constant in ALL LAYERS; ky = 0 for normal incidence
print((n_i**2, kx**2+ky**2))
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


#thickness 0 means L = 0, which only pops up in the xponential part of the expression
ER = [12]
UR = [1]
layer_thicknesses = [0.6]
## run simulation
Ref, Tran = rTMM.run_TMM_simulation(wavelengths, polarization_amplitudes, theta, phi, ER, UR, layer_thicknesses,\
                       transmission_medium, incident_medium)

plt.figure();
plt.plot(wavelengths, Ref);
plt.plot(wavelengths, Tran);
plt.title('Spectrum of a Bragg Mirror')
plt.xlabel('wavelength ($\mu m$)')
plt.ylabel('R/T')
plt.legend(('Ref','Tran'))
plt.savefig('bragg_TMM.png');
plt.show();