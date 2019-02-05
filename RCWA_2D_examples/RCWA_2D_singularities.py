from RCWA_functions import K_matrix as km
from RCWA_functions import PQ_matrices as pq
from RCWA_functions import homogeneous_layer as hl
from TMM_functions import eigen_modes as em
from TMM_functions import scatter_matrices as sm
from RCWA_functions import redheffer_star as rs
from scipy import linalg as LA
from numpy.linalg import cond
import numpy as np;
from RCWA_functions import rcwa_initial_conditions as ic

import matplotlib.pyplot as plt
from convolution_matrices import convmat2D as cm
import cmath
from RCWA_functions import run_RCWA_simulation as rrs

'''
spectra with the EMLab triangle
'''

meters = 1;
centimeters= 1e-2*meters;
degrees = np.pi/180;

eps0 = 8.854e-12*centimeters;
mu0 = 4*np.pi*10**-7*centimeters;
c0 = 1/(np.sqrt(mu0*eps0))

# Source parameters
lam0 = 2*centimeters;
theta = 0
phi = 0
pte = 1; #te polarized
ptm = 0;
normal_vector = np.array([0, 0, -1]) #positive z points down;
ate_vector = np.array([0, 1, 0]); #vector for the out of plane E-field

k0 = 2*np.pi/lam0;
print('k0: '+str(k0))
# structure parameters
#reflection 1 and transmission 2
ur1 = 1; er1 = 1; n_i = np.sqrt(ur1*er1)
ur2 = 1; er2 = 1;

urd = 1;
erd = 6;

#dimensions of the unit cell
Lx = 1.75*centimeters;
Ly = 1.5*centimeters;

#thickness of layers
d1 = 0.5*centimeters
d2 = 0.3*centimeters
w = 0.5*Ly; #side length of equilateral triangle
layer_thicknesses = [d1,d2];  # this retains SI unit convention


#RCWA parameters
Nx = 512;
Ny = round(Nx*Ly/Lx);
N= 3;
M = 3;
PQ = [N,M]; #number of spatial harmonics

NH = (2*(PQ[0])+1)*(2*(PQ[1])+1);

## =========================== BUILD DEVICE ON GRID ==================================##
dx = Lx/Nx;
dy = Ly/Ny;
xa = np.linspace(0,Lx,Nx);
xa = xa - np.mean(xa);
ya = np.linspace(0,Ly,Ny);
ya = ya - np.mean(ya);

#initialize layers

UR = np.ones((Nx,Ny,2)); #interestin
ER = erd*np.ones((Nx,Ny,2))

L = [d1,d2];

# Build the triangle
h = 0.5*np.sqrt(3)*w;
ny = int(np.round(h/dy)); #discrete height
ny1 = np.round((Ny-ny)/2)
ny2 = ny1+ny-1;
print(str(ny1)+', '+str(ny2))

for ny_ind  in np.arange(ny1, ny2+1):
    #build the triangle slice wise
    f = (ny_ind-ny1)/(ny2-ny1);
    #fractional occupation;

    nx = int(round(f*(w/Lx)*Nx)); #x width
    nx1 = 1+int(np.floor((Nx-nx)/2));
    nx2 = int(nx1+nx);
    #print(str(nx1)+', '+str(nx2))
    ER[nx1:nx2+1, int(ny_ind), 0] = er1;

# plt.imshow(ER[:,:,0])
# plt.colorbar()
# plt.show()
Af = np.fft.fftshift(np.fft.fft2(ER[:,:,0]));
plt.figure();
plt.subplot(121)
plt.imshow(np.log(abs(Af))); #note that the fft HAS A HUGE RANGE OF VALUES, which can become a source of inaccuracy
plt.colorbar()
plt.subplot(122)
plt.imshow(np.abs(ER[:,:,0]))
plt.show();
## conv matrices of the 1st
E_conv = (cm.convmat2D(ER[:, :, 0], PQ[0], PQ[1]));
np.set_printoptions(precision = 4)
print(E_conv)
mu_conv = (np.identity(NH));

## Build the second layer (uniform)
URC2 = (np.identity(NH))
ERC2= erd*(np.identity(NH));
ER = [E_conv, ERC2];
UR = [mu_conv, URC2];

wavelengths = np.linspace(2,4, 501)*centimeters
ref = list(); trans = list();

ref = list();
tran = list();
for wvlen in wavelengths:
    print('wvlen: ' + str(wvlen));
    omega = 2 * np.pi * c0 / (wvlen);  # must be in SI for eps_drude

    # source parameters
    theta = 0 * degrees;  # %elevation angle
    phi = 0 * degrees;  # %azimuthal angle

    ## incident wave polarization
    normal_vector = np.array([0, 0, -1])  # positive z points down;
    ate_vector = np.matrix([0, 1, 0]);  # vector for the out of plane E-field
    # ampltidue of the te vs tm modes (which are decoupled)
    pte = 1 / np.sqrt(2);
    ptm = cmath.sqrt(-1) / np.sqrt(2);

    lattice_constants = [Lx, Ly];
    e_half = [1, 1];
    R, T = rrs.run_RCWA_2D(wvlen, theta, phi, ER, UR, layer_thicknesses, lattice_constants, pte, ptm, N, M, e_half)
    ref.append(R);
    trans.append(T)
    print(R);

ref = np.array(ref);
trans = np.array(trans);
plt.figure();
plt.plot(wavelengths/centimeters, ref);
plt.plot(wavelengths/centimeters, trans);
plt.plot(wavelengths/centimeters, ref+trans)
plt.legend(('ref', 'tran', 'r+t'))
plt.show()
