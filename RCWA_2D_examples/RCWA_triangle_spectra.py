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
'''
spectra with the EMLab triangle
'''

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
ER = 12*np.ones((Nx,Ny,2))

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
E_conv = np.matrix(cm.convmat2D(ER[:, :, 0], PQ[0], PQ[1]));
np.set_printoptions(precision = 4)
print(E_conv)
mu_conv = np.matrix(np.identity(NH));

## Build the second layer (uniform)
URC2 = np.matrix(np.identity(NH))
ERC2= erd*np.matrix(np.identity(NH));
ER = [E_conv, ERC2];
UR = [mu_conv, URC2];

wavelengths = np.linspace(0.5, 5, 501)*centimeters
ref = list(); trans = list();

for i in range(len(wavelengths)): #in SI units

    # define vacuum wavevector k0
    lam0 = wavelengths[i]; #k0 and lam0 are related by 2*pi/lam0 = k0
    k0 = 2*np.pi/lam0;
    ## BUILD THE K_MATRIX
    kx_inc = n_i * np.sin(theta) * np.cos(phi);
    ky_inc = n_i * np.sin(theta) * np.sin(phi);  # constant in ALL LAYERS; ky = 0 for normal incidence
    kz_inc = cmath.sqrt(n_i**2 - kx_inc ** 2 - ky_inc ** 2);

    Kx, Ky = km.K_matrix_cubic_2D(kx_inc, ky_inc, k0, Lx, Ly,  PQ[0], PQ[1]);
    Kx = Kx.todense();
    Ky = Ky.todense();


    ## directly insert photonic circle
    # define vacuum wavevector k0

    print('wavelength: ' + str(lam0))
    ## ============== values to keep track of =======================##
    S_matrices = list();
    kz_storage = list();
    ## ==============================================================##


    ## =============== K Matrices for gap medium =========================
    ## specify gap media (this is an LHI so no eigenvalue problem should be solved
    e_h = 1;
    m_h = 1;
    Wg, Vg, Kzg = hl.homogeneous_module(Kx, Ky, e_h)

    ### ================= Working on the Reflection Side =========== ##
    Wr, Vr, kzr = hl.homogeneous_module(Kx, Ky, er1);
    kz_storage.append(kzr)

    ## calculating A and B matrices for scattering matrix
    Ar, Br = sm.A_B_matrices_half_space(Wr, Wg, Vr, Vg); #make sure this order is right

    ## s_ref is a matrix, Sr_dict is a dictionary
    S_ref, Sr_dict = sm.S_R(Ar, Br);  # scatter matrix for the reflection region
    Sg = Sr_dict;

    Q_storage = list();
    P_storage = list();
    ## go through the layers
    for i in range(len(ER)):
        # ith layer material parameters
        e_conv = ER[i];
        mu_conv = UR[i];

        # longitudinal k_vector
        P, Q, kzl = pq.P_Q_kz(Kx, Ky, e_conv, mu_conv)
        kz_storage.append(kzl)
        Gamma_squared = P * Q;

        ## E-field modes that can propagate in the medium, these are well-conditioned
        W_i, lambda_matrix = em.eigen_W(Gamma_squared);
        V_i = em.eigen_V(Q, W_i, lambda_matrix);

        # now defIne A and B, slightly worse conditoined than W and V
        A, B = sm.A_B_matrices(W_i, Wg, V_i, Vg);  # ORDER HERE MATTERS A LOT because W_i is not diagonal

        # calculate scattering matrix
        Li = L[i];
        S_layer, Sl_dict = sm.S_layer(A, B, Li, k0, lambda_matrix)
        S_matrices.append(Sl_dict);

        ## update global scattering matrix using redheffer star
        Sg_matrix, Sg = rs.RedhefferStar(Sg, Sl_dict);

    ##========= Working on the Transmission Side==============##

    Wt, Vt, kz_trans = hl.homogeneous_module(Kx, Ky, er2)

    # get At, Bt
    # since transmission is the same as gap, order does not matter
    At, Bt = sm.A_B_matrices_half_space(Wt,Wg,  Vt, Vg); #make sure this order is right

    ST, ST_dict = sm.S_T(At, Bt)
    #S_matrices.append(ST);
    # update global scattering matrix
    Sg_matrix, Sg = rs.RedhefferStar(Sg, ST_dict);

    ## finally CONVERT THE GLOBAL SCATTERING MATRIX BACK TO A MATRIX

    K_inc_vector = n_i * np.matrix([np.sin(theta) * np.cos(phi), \
                                    np.sin(theta) * np.sin(phi), np.cos(theta)]);

    E_inc, cinc, Polarization = ic.initial_conditions(K_inc_vector, theta, normal_vector, pte, ptm, PQ[0], PQ[1])
    # print(cinc.shape)
    # print(cinc)

    ## COMPUTE FIELDS: similar idea but more complex for RCWA since you have individual modes each contributing
    reflected = Wr * Sg['S11'] * cinc;
    transmitted = Wt * Sg['S21'] * cinc;

    rx = reflected[0:NH, :];  # rx is the Ex component.
    ry = reflected[NH:, :];  #
    tx = transmitted[0:NH, :];
    ty = transmitted[NH:, :];

    # longitudinal components; should be 0
    rz = kzr.I * (Kx * rx + Ky * ry);
    tz = kz_trans.I * (Kx * tx + Ky * ty)

    ## we need to do some reshaping at some point

    ## apparently we're not done...now we need to compute 'diffraction efficiency'
    r_sq = np.square(np.abs(rx)) + np.square(np.abs(ry)) + np.square(np.abs(rz));
    t_sq = np.square(np.abs(tx)) + np.square(np.abs(ty)) + np.square(np.abs(tz));
    R = np.real(kzr) * r_sq / np.real(kz_inc);
    T = np.real(kz_trans) * t_sq / (np.real(kz_inc));
    ref.append(np.sum(R));
    trans.append(np.sum(T))


ref = np.array(ref);
trans = np.array(trans);
plt.figure();
plt.plot(wavelengths, ref);
plt.plot(wavelengths, trans);
plt.plot(wavelengths, ref+trans)

plt.show()
