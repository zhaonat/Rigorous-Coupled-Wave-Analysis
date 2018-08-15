from RCWA_functions import K_matrix as km
from RCWA_functions import PQ_matrices as pq
from RCWA_functions import homogeneous_layer as hl
from TMM_functions import eigen_modes as em
from TMM_functions import scatter_matrices as sm
from RCWA_functions import redheffer_star as rs
from scipy import linalg as LA
from numpy.linalg import cond
import time
import numpy as np;
from RCWA_functions import rcwa_initial_conditions as ic

import matplotlib.pyplot as plt
from convolution_matrices import convmat2D as cm
import cmath
'''
used in conjunction with CEM EMLab triangle to check for correctness
3x3 spatial harmonics
Currently 2nd Homogeneous layer giving us some problems
'''
t0 = time.time()

meters = 1;
centimeters= 1e-2*meters;
degrees = np.pi/180;

# Source parameters
lam0 = 2*centimeters;
theta = 0
phi = 0
pte = 1; #te polarized
ptm = 0;
normal_vector = np.array([0, 0, -1]) #positive z points down;
ate_vector = np.matrix([0, 1, 0]); #vector for the out of plane E-field

k0 = 2*np.pi/lam0;
print('k0: '+str(k0))
# structure parameters
#reflection 1 and transmission 2
ur1 = 1; er1 = 2; n_i = np.sqrt(ur1*er1)
ur2 = 1; er2 = 9;

## second layer
urd = 1;
erd = 6;

#dimensions of the unit cell
Lx = 1.75*centimeters;
Ly = 1.5*centimeters;

#thickness of layers
d1 = 0.5*centimeters
d2 = 0.3*centimeters
w = 0.8*Ly;

#RCWA parameters
Nx = 512;
Ny = round(Nx*Ly/Lx);
PQ = [1,1]; #number of spatial harmonics
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
# Af = np.fft.fftshift(np.fft.fft2(ER[:,:,0]));
# plt.figure();
# plt.subplot(121)
# plt.imshow(np.log(abs(Af))); #note that the fft HAS A HUGE RANGE OF VALUES, which can become a source of inaccuracy
# plt.colorbar()
# plt.subplot(122)
# plt.imshow(np.abs(ER[:,:,0]))
# plt.show();
## conv matrices of the 1st
E_conv = np.matrix(cm.convmat2D(ER[:, :, 0], PQ[0], PQ[1]));
np.set_printoptions(precision = 4)
print(E_conv)
mu_conv = np.matrix(np.identity(NH));

## Build the second layer (uniform)
URC2 = np.matrix(np.identity(NH))
ERC2= erd*np.matrix(np.identity(NH));


## BUILD THE K_MATRIX
kx_inc = n_i * np.sin(theta) * np.cos(phi);
ky_inc = n_i * np.sin(theta) * np.sin(phi);  # constant in ALL LAYERS; ky = 0 for normal incidence
kz_inc = cmath.sqrt(n_i**2 - kx_inc ** 2 - ky_inc ** 2);

Kx, Ky = km.K_matrix_cubic_2D(kx_inc, ky_inc, k0, Lx, Ly,  PQ[0], PQ[1]);
Kx = Kx.todense();
Ky = Ky.todense();

#gap media
Wg, Vg, Kzg = hl.homogeneous_module(Kx, Ky, 1);

## Get Kzr and Kztrans
Wr, Vr, Kzr = hl.homogeneous_module(Kx, Ky, er1);
Ar, Br = sm.A_B_matrices_half_space(Wr, Wg, Vr, Vg); #make sure this order is right
Sr, Sr_dict = sm.S_R(Ar, Br)
Sg = Sr_dict;


## =================================================================##
##               First LAYER (homogeneous)
## =======================================================================##
P, Q, Kzl = pq.P_Q_kz(Kx, Ky, E_conv, mu_conv)
omega_sq =  P * Q; ## no gaurantees this is hermitian or symmetric
W1, lambda_matrix = em.eigen_W(omega_sq)
V1 = em.eigen_V(Q, W1, lambda_matrix)
A1, B1 = sm.A_B_matrices(W1, Wg, V1, Vg);
S1, S1_dict = sm.S_layer(A1, B1, d1, k0, lambda_matrix)
Sg_matrix, Sg = rs.RedhefferStar(Sg, S1_dict)

## =================================================================##
##               SECOND LAYER (homogeneous)
## =======================================================================##

##check with PQ formalism, which is unnecessary
P2, Q2, Kz2_check = pq.P_Q_kz(Kx, Ky, ERC2, URC2)
omega_sq_2 =  P2 * Q2;
W2, lambda_matrix_2 = em.eigen_W(omega_sq_2) #somehow lambda_matrix is fine but W is full of errors
V2 = em.eigen_V(Q2,W2,lambda_matrix_2);
A2, B2 = sm.A_B_matrices(W2, Wg,V2, Vg);
S2, S2_dict = sm.S_layer(A2,B2, d2, k0, lambda_matrix_2)
Sg_matrix, Sg = rs.RedhefferStar(Sg, S2_dict);

## TRANSMISSION LAYER
# #create ST
Wt, Vt, Kzt = hl.homogeneous_module(Kx, Ky, er2);
At, Bt = sm.A_B_matrices_half_space(Wt,Wg,  Vt, Vg); #make sure this order is right
St, St_dict = sm.S_T(At, Bt); ### FUCKKKKKKKKKKKKKKKK
Sg_matrix, Sg = rs.RedhefferStar(Sg, St_dict);

print('final Sg')
print(Sg['S11'])


## ================START THE SSCATTERING CALCULATION ==========================##

K_inc_vector = n_i * np.matrix([np.sin(theta) * np.cos(phi), \
                                     np.sin(theta) * np.sin(phi), np.cos(theta)]);
E_inc, cinc, Polarization = ic.initial_conditions(K_inc_vector, theta, normal_vector, pte, ptm, PQ[0], PQ[1])


## COMPUTE FIELDS: similar idea but more complex for RCWA since you have individual modes each contributing
reflected = Wr * Sg['S11'] * cinc;  # reflection coefficients for every mode...
transmitted = Wt * Sg['S21'] * cinc;

## these include only (rx, ry), (tx, ty), which is okay as these are the only components for normal incidence in LHI
rx = reflected[0:NH, :];
ry = reflected[NH:, :];
tx = transmitted[0:NH, :];
ty = transmitted[NH:, :];

# longitudinal components; should be 0
rz = Kzr.I * (Kx * rx + Ky * ry);
tz = Kzt.I * (Kx * tx + Ky * ty)

print('rx')
print(rx)
print('ry')
print(ry)
print('rz'); print(rz)

## apparently we're not done...now we need to compute 'diffraction efficiency'
r_sq = np.square(np.abs(rx)) + np.square(np.abs(ry)) + np.square(np.abs(rz));
t_sq = np.square(np.abs(tx)) + np.square(np.abs(ty)) + np.square(np.abs(tz));
R = np.real(Kzr) * r_sq / np.real(kz_inc);
T = np.real(Kzt) * t_sq / (np.real(kz_inc));

print('final R vector-> matrix')
print(np.reshape(R,(3,3))); #should be 3x3
print('final T vector/matrix')
print(np.reshape(T,(3,3)))
print('final reflection: '+str(np.sum(R)))
print('final transmission: '+str(np.sum(T)))
print('sum of R and T: '+str(np.sum(R)+np.sum(T)))

## if the sum isn't 1, that's a PROBLEM
t1 = time.time()

print('time: '+str(abs(t1-t0)))

## times
# 08/15/2018: 0.3 down to 0.19
