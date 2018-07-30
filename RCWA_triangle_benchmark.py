from RCWA_functions import K_matrix as km
from RCWA_functions import PQ_matrices as pq
from RCWA_functions import homogeneous_layer as hl
from TMM_functions import eigen_modes as em
from TMM_functions import scatter_matrices as sm
from TMM_functions import scatter_matrices as sm
from RCWA_functions import redheffer_star as rs
from scipy import linalg as LA
import numpy as np;
from RCWA_functions import rcwa_initial_conditions as ic

import matplotlib.pyplot as plt
from convolution_matrices import convmat2D as cm

'''
used in conjunction with CEM EMLab triangle to check for correectness
'''

meters = 1;
centimeters= 1e-2;
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
Af = np.fft.fftshift(np.fft.fft2(ER[:,:,0]));
plt.figure();
plt.imshow(np.log(abs(Af))); #note that the fft HAS A HUGE RANGE OF VALUES, which can become a source of inaccuracy
plt.colorbar()
plt.show();
## conv matrices of the 1st
E_r = np.matrix(cm.convmat2D(ER[:,:,0], PQ[0], PQ[1]));#already has errors
np.set_printoptions(precision = 4)
print(E_r)
mu_conv = np.matrix(np.identity(NH));

## Build the second layer (uniform)
URC = np.matrix(np.identity(NH))
ERC = 6*np.matrix(np.identity(NH));


## BUILD THE K_MATRIX
Kx, Ky = km.K_matrix_cubic_2D(0,0, k0, Lx, Ly,  PQ[0], PQ[1]);
print(Kx.todense())
print(Ky.todense());

#gap media
Wg, Vg, Kzg = hl.homogeneous_module(Kx.todense(), Ky.todense(), 1);

## Get Kzr and Kztrans
Wr, Vr, Kzr = hl.homogeneous_module(Kx.todense(), Ky.todense(), er1);
print(Kzr)
print(Vr)
Wt, Vt, Kzt = hl.homogeneous_module(Kx.todense(), Ky.todense(), er2);


#create SR
Ar, Br = sm.A_B_matrices(Wg, Wr, Vg, Vr); #make sure this order is right
Sr, Sr_dict = sm.S_R(Ar, Br)

#create ST
At, Bt = sm.A_B_matrices(Wg, Wt, Vg, Vt); #make sure this order is right
St, St_dict = sm.S_R(At, Bt)

#somehow vg is supposed to be real.

## global scattering matrix
NM = NH;
Sg11 = np.matrix(np.zeros((2 * NM, 2 * NM)));
Sg12 = np.matrix(np.eye(2 * NM, 2 * NM));
Sg21 = np.matrix(np.eye(2 * NM, 2 * NM));
Sg22 = np.matrix(np.zeros((2 * NM, 2 * NM)));  # matrices
Sg = {'S11': Sg11, 'S12': Sg12, 'S21': Sg21,
      'S22': Sg22};  # initialization is equivelant as that for S_reflection side matrix
Sg0 = Sg;
print(Sg['S11'].shape)

## main loop Layer 1
P, Q, Kzl = pq.P_Q_kz(Kx.todense(), Ky.todense(), E_r, mu_conv)
omega_sq =  P*Q;

#print(omega_sq)

#intentionally zero out small elements?
#omega_sq[omega_sq < 1e-8] = 0

## get eigenmodes of the layer
W, lambda_matrix = em.eigen_W(omega_sq) #somehow lambda_matrix is fine but W is full of errors
print('================================================')
print(W); #this is close but not quite really accurate...could becuase of near degnerate eigenvalues...

print('=================================================')
print(lambda_matrix)
V = em.eigen_V(Q,W,lambda_matrix)
#print(V)

print(np.linalg.norm(Wg - np.matrix(np.identity(2*NH))))
## compute A and B -- at this point the matrices start diverging from the benchmark


A, B = sm.A_B_matrices(W, Wg, V, Vg);
print('=================================================')
print('A matrix')
print(A)

print('=================================================')
print('B matrix')
print(B)

X_i = LA.expm(lambda_matrix * d1 * k0);  # k and L are in Si Units
print(X_i)

## finally get scattering matrix of the layer
S1, S1_dict = sm.S_layer(B,A, d1, k0, lambda_matrix)

print(S1_dict['S11'])


## second layer (homogeneous)
## main loop Layer 1
W2, V2, Kz2 = hl.homogeneous_module(Kx.todense(), Ky.todense(), 6);
print(W2)
print(V2)

A2, B2 = sm.A_B_matrices(W2, Wg, V2, Vg);
lambda_matrix = LA.block_diag(Kz2, Kz2);
S2, S2_dict = sm.S_layer(B2,A2, d2, k0, lambda_matrix)

##global scatterming matrix
Sg, Sg_dict = rs.RedhefferStar(S1_dict, S2_dict)
St, Stotal_dict = rs.RedhefferStar(Sr_dict, Sg_dict);
St, Stotal_dict = rs.RedhefferStar(Stotal_dict, St_dict);
print(Stotal_dict['S11'])


##
K_inc_vector = n_i * k0 * np.matrix([np.sin(theta) * np.cos(phi), \
                                     np.sin(theta) * np.sin(phi), np.cos(theta)]);
kz_inc = K_inc_vector[0,2];
E_inc, cinc, Polarization = ic.initial_conditions(K_inc_vector, theta, normal_vector, pte, ptm, PQ[0], PQ[1])
# print(cinc.shape)
# print(cinc)

cinc = Wr.I * cinc;  # All W's are identity matrices, so they do nothing
## COMPUTE FIELDS: similar idea but more complex for RCWA since you have individual modes each contributing
reflected = Wr * Stotal_dict['S11'] * cinc;  # reflection coefficients for every mode...
transmitted = Wt * Stotal_dict['S21'] * cinc;

## these include only (rx, ry), (tx, ty), which is okay as these are the only components for normal incidence in LHI
rx = reflected[0:NM, :];
ry = reflected[NM:, :];
tx = transmitted[0:NM, :];
ty = transmitted[NM:, :];

# longitudinal components; should be 0
rz = -Kzr.I * (Kx * rx + Ky * ry);
tz = -Kzt.I * (Kx * tx + Ky * ty)

print('rx')
print(rx)
print('ry')
print(ry)
print('rz'); print(rz)

## apparently we're not done...now we need to compute 'diffraction efficiency'
r_sq = np.linalg.norm(reflected) ** 2 +np.linalg.norm(rz)**2
t_sq = np.linalg.norm(transmitted) ** 2 +np.linalg.norm(tz)**2;
R = np.real(Kzr) * r_sq / np.real(kz_inc/k0);
print(R); #should be 3x3

print('final reflection: '+str(np.sum(R)))
