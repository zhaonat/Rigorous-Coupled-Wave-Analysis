import numpy as np
import matplotlib.pyplot as plt
from convolution_matrices import convmat2D as cm
from RCWA_functions import K_matrix as km
from RCWA_functions import PQ_matrices as pq
from TMM_functions import eigen_modes as em
from TMM_functions import scatter_matrices as sm
from RCWA_functions import redheffer_star as rs
from RCWA_functions import rcwa_initial_conditions as ic
from RCWA_functions import homogeneous_layer as hl
import cmath
from numpy.linalg import cond

'''
solve RCWA for a simple circular structure in a square unit cell

In essence, there is almost nothing new that needs to be done
We do have to modify scatter_matrices and redheffer star
## normalized units
#z' = k0*z;
#k = k/k0;

FINALLY WORKS 8/11/2018
'''

#% General Units
degrees = np.pi/180;
L0 = 1e-6; #units of microns;
eps0 = 8.854e-12*L0;
mu0 = 4*np.pi*10**-7*L0;
c0 = 1/(np.sqrt(mu0*eps0))

## lattice and material parameters
a = 1;
radius = 0.2; #0.4;
e_r = 12;

#generate irreducible BZ sample
T1 = 2*np.pi/a;
T2 = 2*np.pi/a;

## Specify number of fourier orders to use:
N = 1; M = 1;
NM = (2*N+1)*(2*M+1);

# ============== build high resolution circle ==================
Nx = 512; Ny = 512;
A = e_r*np.ones((Nx,Ny));
ci = int(Nx/2); cj= int(Ny/2);
cr = (radius/a)*Nx;
I,J=np.meshgrid(np.arange(A.shape[0]),np.arange(A.shape[1]));

dist = np.sqrt((I-ci)**2 + (J-cj)**2);
A[np.where(dist<cr)] = 1;

#visualize structure
# plt.imshow(A);
# plt.show()

## =============== Convolution Matrices ==============
E_r = cm.convmat2D(A, N,M)
print(type(E_r))
E_r = np.matrix(E_r)
plt.figure();
plt.imshow(abs(E_r), cmap = 'jet');
plt.colorbar()
plt.show()

## ================== GEOMETRY OF THE LAYERS AND CONVOLUTIONS ==================##
thickness_slab = 0.76; # in units of L0;
ER = [E_r];
UR = [np.matrix(np.identity(NM))];
layer_thicknesses = [thickness_slab]; #this retains SI unit convention

## =============== Simulation Parameters =========================
## set wavelength scanning range
wavelengths = np.linspace(0.5,2,401); #500 nm to 1000 nm #be aware of Wood's Anomalies
kmagnitude_scan = 2 * np.pi / wavelengths; #no
omega = c0 * kmagnitude_scan; #using the dispersion wavelengths

#source parameters
theta = 0 * degrees; #%elevation angle
phi = 0 * degrees; #%azimuthal angle

## incident wave polarization
normal_vector = np.array([0, 0, -1]) #positive z points down;
ate_vector = np.matrix([0, 1, 0]); #vector for the out of plane E-field
#ampltidue of the te vs tm modes (which are decoupled)
pte = 1/np.sqrt(2);
ptm = cmath.sqrt(-1)/np.sqrt(2);


## ======================= RUN SIMULATION =========================
# GET SPECTRAL REFLECTION FOR THE PHOTONIC CRYSTAL, SHOULD SEE FANO RESONANCES
ref = list(); trans = list();

for i in range(len(wavelengths)): #in SI units

    # define vacuum wavevector k0
    k0 = kmagnitude_scan[i]; #this is in SI units, it is the normalization constant for the k-vector
    lam0 = wavelengths[i]; #k0 and lam0 are related by 2*pi/lam0 = k0
    print('wavelength: '+ str(lam0))
    ## ============== values to keep track of =======================##
    S_matrices = list();
    kz_storage = list();
    X_storage = list();
    ## ==============================================================##

    m_r = 1; e_r = 1;
    ## incident wave properties, at this point, everything is in units of k_0
    n_i =  np.sqrt(e_r*m_r);

    # actually, in the definitions here, kx = k0*sin(theta)*cos(phi), so kx, ky here are normalized
    kx_inc = n_i * np.sin(theta) * np.cos(phi);
    ky_inc = n_i * np.sin(theta) * np.sin(phi);  # constant in ALL LAYERS; ky = 0 for normal incidence
    kz_inc = cmath.sqrt(e_r * 1 - kx_inc ** 2 - ky_inc ** 2);

    #remember, these Kx and Ky come out already normalized
    Kx, Ky = km.K_matrix_cubic_2D(kx_inc, ky_inc, k0, a, a, N, M); #Kx and Ky are diagonal but have a 0 on it
    ## density Kx and Ky (for now)
    Kx = Kx.todense();  Ky = Ky.todense();

    ## =============== K Matrices for gap medium =========================
    ## specify gap media (this is an LHI so no eigenvalue problem should be solved
    e_h = 1; m_h = 1;
    Wg, Vg, Kzg = hl.homogeneous_module(Kx, Ky, e_h)

    ### ================= Working on the Reflection Side =========== ##
    Wr, Vr, kzr = hl.homogeneous_module(Kx, Ky, e_r); kz_storage.append(kzr)

    ## calculating A and B matrices for scattering matrix
    # since gap medium and reflection media are the same, this doesn't affect anything
    Ar, Br = sm.A_B_matrices(Wg, Wr, Vg, Vr);

    ## s_ref is a matrix, Sr_dict is a dictionary
    S_ref, Sr_dict = sm.S_R(Ar, Br); #scatter matrix for the reflection region
    S_matrices.append(S_ref);
    Sg = Sr_dict;

    Q_storage = list(); P_storage=  list();
    ## go through the layers
    for i in range(len(ER)):
        #ith layer material parameters
        e_conv = ER[i]; mu_conv = UR[i];

        #longitudinal k_vector
        P, Q, kzl = pq.P_Q_kz(Kx, Ky, e_conv, mu_conv)
        kz_storage.append(kzl)
        Gamma_squared = P*Q;

        ## E-field modes that can propagate in the medium, these are well-conditioned
        W_i, lambda_matrix = em.eigen_W(Gamma_squared);
        V_i = em.eigen_V(Q, W_i, lambda_matrix);

        #now defIne A and B, slightly worse conditoined than W and V
        A,B = sm.A_B_matrices(W_i, Wg, V_i, Vg); #ORDER HERE MATTERS A LOT because W_i is not diagonal

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
    # since transmission is the same as gap, order does not matter
    At, Bt = sm.A_B_matrices(Wg, Wt, Vg, Vt)

    ST, ST_dict = sm.S_T(At, Bt)
    S_matrices.append(ST);
    #update global scattering matrix
    Sg_matrix, Sg = rs.RedhefferStar(Sg, ST_dict);

    ## finally CONVERT THE GLOBAL SCATTERING MATRIX BACK TO A MATRIX

    K_inc_vector = n_i *np.matrix([np.sin(theta) * np.cos(phi), \
                                    np.sin(theta) * np.sin(phi), np.cos(theta)]);

    E_inc, cinc, Polarization = ic.initial_conditions(K_inc_vector, theta,  normal_vector, pte, ptm, N,M)
    # print(cinc.shape)
    # print(cinc)

    cinc = Wr.I*cinc;
    ## COMPUTE FIELDS: similar idea but more complex for RCWA since you have individual modes each contributing
    reflected = Wr*Sg['S11']*cinc;
    transmitted = Wt*Sg['S21']*cinc;

    rx = reflected[0:NM, :]; # rx is the Ex component.
    ry = reflected[NM:, :];  #
    tx = transmitted[0:NM,:];
    ty = transmitted[NM:, :];

    # longitudinal components; should be 0
    rz = kzr.I * (Kx * rx + Ky * ry);
    tz = kz_trans.I * (Kx * tx + Ky * ty)

    ## we need to do some reshaping at some point

    ## apparently we're not done...now we need to compute 'diffraction efficiency'
    r_sq = np.square(np.abs(rx)) +  np.square(np.abs(ry))+ np.square(np.abs(rz));
    t_sq = np.square(np.abs(tx)) +  np.square(np.abs(ty))+ np.square(np.abs(tz));
    R = np.real(kzr) * r_sq / np.real(kz_inc);
    T = np.real(kz_trans)*t_sq/(np.real(kz_inc));
    ref.append(np.sum(R));
    trans.append(np.sum(T))

    print('final R vector-> matrix')
    print(np.reshape(R,(3,3))); #should be 3x3
    print('final T vector/matrix')
    print(np.reshape(T,(3,3)))

ref = np.array(ref);
trans = np.array(trans);
plt.figure();
plt.plot(wavelengths, ref);
plt.plot(wavelengths, trans);
plt.plot(wavelengths, ref+trans)

plt.show()
