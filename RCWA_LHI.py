import numpy as np
import matplotlib.pyplot as plt
from convolution_matrices import convmat2D as cm
from RCWA_functions import K_matrix as km
from RCWA_functions import PQ_matrices as pq
from TMM_functions import eigen_modes as em
from TMM_functions import scatter_matrices as sm
from RCWA_functions import redheffer_star as rs
from TMM_functions import generate_initial_conditions as ic
from RCWA_functions import homogeneous_layer as hl
import cmath

'''
solve RCWA for a simple circular structure in a square unit cell
and generate band structure
compare with CEM EMLab

In essence, there is almost nothing new that needs to be done
We do have to modify scatter_matrices and redheffer star
## normalized units
#z' = k0*z;
#k = k/k0;
'''

#% General Units
degrees = np.pi/180;
L0 = 1e-6; #units of microns;
eps0 = 8.854e-12;
mu0 = 4*np.pi*10**-7;
c0 = 1/(np.sqrt(mu0*eps0))

## lattice and material parameters
a = 1;
radius = 0.35;
e_r = 9;

#generate irreducible BZ sample
T1 = 2*np.pi/a;
T2 = 2*np.pi/a;

## Specify number of fourier orders to use:
N = 10;
M = 10;
NM = N*M;


## ================== GEOMETRY OF THE LAYERS AND CONVOLUTIONS ==================##
thickness_slab = 0.76*L0; #100 nm;
ER = [12];
UR = [1];
layer_thicknesses = [thickness_slab]; #this retains SI unit convention

## =============== Simulation Parameters =========================
## set wavelength scanning range
wavelengths = L0*np.linspace(1,2,1000); #500 nm to 1000 nm
kmagnitude_scan = 2 * np.pi / wavelengths; #no
omega = c0 * kmagnitude_scan; #using the dispersion wavelengths

#source parameters
theta = 0 * degrees; #%elevation angle
phi = 0 * degrees; #%azimuthal angle

## incident wave properties, at this point, everything is in units of k_0
n_i = 1;

#actually, in the definitions here, kx = k0*sin(theta)*cos(phi), so kx, ky here are normalized
beta_x = beta_y = 0; #does beta still correspond to incident wavevector?

Kx, Ky = km.K_matrix_cubic_2D(beta_x, beta_y, a, a, N,M);
## density Kx and Ky (for now)
Kx = Kx.todense();
Ky = Ky.todense();


## incident wave polarization
normal_vector = np.array([0, 0, -1]) #positive z points down;
ate_vector = np.matrix([0, 1, 0]); #vector for the out of plane E-field
#ampltidue of the te vs tm modes (which are decoupled)
pte = 1/np.sqrt(2);
ptm = cmath.sqrt(-1)/np.sqrt(2);

## =============== K Matrices for gap medium =========================
## specify gap media (this is an LHI so no eigenvalue problem should be solved
e_h = 1; m_h = 1;
Wg,Vg,Kzg = hl.homogeneous_module(Kx, Ky, e_h)

## ======================= RUN SIMULATION =========================
# GET SPECTRAL REFLECTION FOR THE PHOTONIC CRYSTAL, SHOULD SEE FANO RESONANCES
ref = list(); trans = list();

for i in range(len(wavelengths)): #in SI units

    ## ============== values to keep track of =======================##
    S_matrices = list();
    kz_storage = list();
    X_storage = list();
    ## ==============================================================##

    ## Initialize global scattering matrix
    Sg11 = np.matrix(np.zeros((2*NM,2*NM)));
    Sg12 = np.matrix(np.eye(2*NM,2*NM));
    Sg21 = np.matrix(np.eye(2*NM,2*NM));
    Sg22 = np.matrix(np.zeros((2*NM,2*NM)));  # matrices
    Sg = {'S11':Sg11, 'S12':Sg12, 'S21':Sg21,'S22': Sg22};  # initialization is equivelant as that for S_reflection side matrix
    Sg0 = Sg;

    ### ================= Working on the Reflection Side =========== ##
    m_r = 1; e_r = 1;
    Wr, Vr, kzr = hl.homogeneous_module(Kx, Ky, e_r); kz_storage.append(kzr)

    # define vacuum wavevector k0
    k0 = kmagnitude_scan[i]; #this is in SI units, it is the normalization constant for the k-vector
    lam0 = wavelengths[i]; #k0 and lam0 are related by 2*pi/lam0 = k0

    ## calculating A and B matrices for scattering matrix
    Ar, Br = sm.A_B_matrices(Wg, Wr, Vg, Vr);

    S_ref, Sr_dict = sm.S_R(Ar, Br); #scatter matrix for the reflection region
    S_matrices.append(S_ref);
    Sg_matrix, Sg = rs.RedhefferStar(Sg, Sr_dict);

    Q_storage = list(); P_storage=  list();
    ## go through the layers
    for i in range(len(ER)):
        #ith layer material parameters
        e_conv = ER[i]*np.matrix(np.eye(NM,NM)); m = UR[i];

        #longitudinal k_vector
        P, Q, kzl = pq.P_Q_kz(Kx, Ky, e_conv)
        kz_storage.append(kzl)
        Gamma_squared = P*Q;

        ## E-field modes that can propagate in the medium
        W_i, lambda_matrix = em.eigen_W(Gamma_squared);
        V_i = em.eigen_V(Q, W_i, lambda_matrix);

        #now defIne A and B
        A,B = sm.A_B_matrices(Wg, W_i, Vg, V_i);

        #calculate scattering matrix
        Li = layer_thicknesses[i];
        S_layer, Sl_dict = sm.S_layer(A, B, Li, k0, lambda_matrix)
        S_matrices.append(S_layer);

        ## update global scattering matrix using redheffer star
        Sg_matrix, Sg = rs.RedhefferStar(Sg, Sl_dict);

    ##========= Working on the Transmission Side==============##
    m_t = 1;    e_t = 1;
    Wt, Vt, kz_trans = hl.homogeneous_module(Kx, Ky, e_t)

    #get At, Bt
    At, Bt = sm.A_B_matrices(Wg, I, Vg, Vt)

    ST, ST_dict = sm.S_T(At, Bt)
    S_matrices.append(ST);
    #update global scattering matrix
    Sg_matrix, Sg = rs.RedhefferStar(Sg, ST_dict);

    ## finally CONVERT THE GLOBAL SCATTERING MATRIX BACK TO A MATRIX


    K_inc_vector = n_i * k0*np.matrix([np.sin(theta) * np.cos(phi), \
                                    np.sin(theta) * np.sin(phi), np.cos(theta)]);

    # cinc is the c1+
    E_inc, cinc, Polarization = ic.initial_conditions(K_inc_vector, theta,  normal_vector, pte, ptm)

    ## COMPUTE FIELDS: similar idea but more complex for RCWA since you have individual modes each contributing


