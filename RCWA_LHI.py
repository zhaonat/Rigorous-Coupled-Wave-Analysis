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
from scipy.linalg import block_diag
import cmath

'''
solve RCWA for a LHI
#FINALLY WORKS: problem occurs at the eigendecomposition of Gamma_squared, probably the same issue

#k = k/k0;
# SIGN CONVENTION IS NEGATIVE
# exp(-1i*k*x) is forward prop in x

'it appears there are pathological cases where bad conditioning is unavoidable'
'EMPy seems to avoid it by injecting noise into the K matrices...


'''

#% General Units
degrees = np.pi/180;
L0 = 1e-6; #units of microns;
eps0 = 8.854e-12;
mu0 = 4*np.pi*10**-7;
c0 = 1/(np.sqrt(mu0*eps0))

## lattice and material parameters
a = 1*L0;
e_layer= 9;
#generate irreducible BZ sample
T1 = 2*np.pi/a;
T2 = 2*np.pi/a;

## Specify number of fourier orders to use:
#sometimes can get orders which make the system singular

N = 2; M = 2;
NM = (2*N+1)*(2*M+1);

# ============== build high resolution circle ==================

E_r = e_layer*np.matrix(np.identity(NM))
U_r = np.matrix(np.identity(NM))

#check that the convmat returns an identity matrix
eps_grid = np.ones((512, 512))
er_check = e_layer*cm.convmat2D(eps_grid, N,M)
er_fft = np.fft.fftshift(np.fft.fft(np.ones((3,3))))/(9)

## ================== GEOMETRY OF THE LAYERS AND CONVOLUTIONS ==================##
thickness_slab = 0.76*L0; #100 nm;
ER = [E_r];
UR = [U_r];
layer_thicknesses = [thickness_slab]; #this retains SI unit convention

## =============== Simulation Parameters =========================
## set wavelength scanning range
wavelengths = L0*np.linspace(0.25,1,1000); #500 nm to 1000 nm #LHI fails for small wavelengths?
kmagnitude_scan = 2 * np.pi / wavelengths; #no
omega = c0 * kmagnitude_scan; #using the dispersion wavelengths

#source parameters
theta = 0 * degrees; #%elevation angle
phi = 0 * degrees; #%azimuthal angle

## incident wave polarization
normal_vector = np.array([0, 0, -1]) #positive z points down;
ate_vector = np.matrix([0, 1, 0]); #vector for the out of plane E-field

#ampltidue of the te vs tm modes (shouldn't affect spectrum for a fabry perot stack)
pte = 1/np.sqrt(2);
ptm = cmath.sqrt(-1)/np.sqrt(2);


## ======================= RUN SIMULATION =========================
# GET SPECTRAL REFLECTION FOR THE PHOTONIC CRYSTAL, SHOULD SEE FANO RESONANCES
ref = list(); trans = list();

for i in range(len(wavelengths)): #in SI units

    # define vacuum wavevector k0
    k0 = kmagnitude_scan[i]; #this is in SI units, it is the normalization constant for the k-vector
    lam0 = wavelengths[i]; #k0 and lam0 are related by 2*pi/lam0 = k0
    #print('wavelength: '+ str(1e6*lam0))
    ## ============== values to keep track of =======================##
    S_matrices = list();
    kz_storage = list();
    X_storage = list();
    ## ==============================================================##


    m_r = 1; e_r = 1+cmath.sqrt(-1)*1e-12;
    ## incident wave properties, at this point, everything is in units of k_0
    n_i =  np.sqrt(e_r*m_r);

    # actually, in the definitions here, kx = k0*sin(theta)*cos(phi), so kx, ky here are normalized
    kx_inc = n_i * np.sin(theta) * np.cos(phi);  # actually, in the definitions here, kx = k0*sin(theta)*cos(phi), so kx, ky here are normalized
    ky_inc = n_i * np.sin(theta) * np.sin(phi);  # constant in ALL LAYERS; ky = 0 for normal incidence
    kz_inc = cmath.sqrt(e_r * 1 - kx_inc ** 2 - ky_inc ** 2);

    Kx, Ky = km.K_matrix_cubic_2D(kx_inc, ky_inc, k0, a, a, N, M); # a and k0 must be in SI units!!!!
    ## density Kx and Ky (for now)
    Kx = Kx.todense();  Ky = Ky.todense();

    ## =============== K Matrices for gap medium =========================
    ## specify gap media (this is an LHI so no eigenvalue problem should be solved
    e_h = 1+cmath.sqrt(-1)*1e-12; m_h = 1;
    Wg, Vg, Kzg = hl.homogeneous_module(Kx, Ky, e_h)

    ### ================= Working on the Reflection Side =========== ##
    Wr, Vr, kzr = hl.homogeneous_module(Kx, Ky, e_r); kz_storage.append(kzr)

    ## calculating A and B matrices for scattering matrix
    Ar, Br = sm.A_B_matrices(Wg, Wr, Vg, Vr);

    S_ref, Sr_dict = sm.S_R(Ar, Br); #scatter matrix for the reflection region
    S_matrices.append(S_ref);
    Sg = Sr_dict;

    Q_storage = list(); P_storage=  list();

    ## go through the layers
    for i in range(len(ER)):
        #ith layer material parameters
        e_conv = ER[i]; mu_conv = UR[i]; #should be identity matrices

        #longitudinal k_vector
        P, Q, kzl = pq.P_Q_kz(Kx, Ky, e_conv, mu_conv)
        #there appear to be pathological cases where kz is imaginary, which from a physical standpoint, makes sense...
        # there are inevitably kx and ky orders that are totally internally reflected
        kz_storage.append(kzl)
        Gamma_squared = P*Q; #condition number starts getting bad at high frequency?,
        #zero out non diagonal elements intentionally!!
        Gamma_squared = np.diag(np.diag(Gamma_squared)); #apparently the errors are strong enough that not doing this fucks it up

        #check that P*Q's diagonals are just kzl
        #print(np.linalg.norm(Gamma_squared - (-block_diag(kzl*kzl,kzl*kzl)))) #surprisingly, the error on this is small

        ## E-field modes that can propagate in the medium
        eigenvalues, W_i = np.linalg.eig(Gamma_squared);
        lambda_matrix = np.matrix(np.diag(np.sqrt(eigenvalues.astype('complex'))));
        W_i = np.matrix(W_i);
        #check that W_i is the identity matrix
        #print(np.linalg.norm(W_i -np.matrix(np.identity(2*NM))))
        V_i = em.eigen_V(Q, W_i, lambda_matrix); #lambda_matrix is singular...

        #now defIne A and B; FOR INTERMEDIATE LAYERS, WG and VG come AFTER, but for LHI, the order actually doesn't matter
        A,B = sm.A_B_matrices(Wg, W_i, Vg, V_i);

        #calculate scattering matrix
        Li = layer_thicknesses[i];
        S_layer, Sl_dict = sm.S_layer(A, B, Li, k0, lambda_matrix)
        S_matrices.append(S_layer);

        ## update global scattering matrix using redheffer star
        Sg_matrix, Sg = rs.RedhefferStar(Sg, Sl_dict);

    ##========= Working on the Transmission Side==============##
    m_t = 1; e_t = 1+cmath.sqrt(-1)*1e-12;
    Wt, Vt, kz_trans = hl.homogeneous_module(Kx, Ky, e_t)

    #get At, Bt
    At, Bt = sm.A_B_matrices(Wg, Wt, Vg, Vt)

    ST, ST_dict = sm.S_T(At, Bt)
    S_matrices.append(ST);
    #update global scattering matrix
    Sg_matrix, Sg = rs.RedhefferStar(Sg, ST_dict);

    ## finally CONVERT THE GLOBAL SCATTERING MATRIX BACK TO A MATRIX

    K_inc_vector = n_i * k0*np.matrix([np.sin(theta) * np.cos(phi), \
                                    np.sin(theta) * np.sin(phi), np.cos(theta)]);

    E_inc, cinc, Polarization = ic.initial_conditions(K_inc_vector, theta,  normal_vector, pte, ptm, N,M)
    # print(cinc.shape)
    # print(cinc)

    cinc = Wr.I*cinc; #All W's are identity matrices, so they do nothing
    ## COMPUTE FIELDS: similar idea but more complex for RCWA since you have individual modes each contributing
    reflected = Wr*Sg['S11']*cinc; #reflection coefficients for every mode...
    transmitted = Wt*Sg['S21']*cinc;

    ## these include only (rx, ry), (tx, ty), which is okay as these are the only components for normal incidence in LHI
    rx = reflected[0:NM, :];
    ry = reflected[NM:, :];
    tx = transmitted[0:NM,:];
    ty = transmitted[NM:, :];

    #longitudinal components; should be 0
    rz = -kzr.I*(Kx*rx + Ky*ry);
    tz = -kz_trans.I*(Kx*tx + Ky*ty)

    ## apparently we're not done...now we need to compute 'diffraction efficiency'
    r_sq = np.linalg.norm(reflected)**2 +np.linalg.norm(rz)**2
    t_sq = np.linalg.norm(transmitted)**2+np.linalg.norm(tz)**2;

    ## calculate the R 'matrix'... error exists here...
    R = np.real(kzr)*r_sq/np.real(kz_inc);

    # ref.append(np.sum(R));
    #  trans.append(1-np.sum(R))

    ref.append(r_sq);
    trans.append(1-r_sq);
plt.figure();
plt.plot(1e6*wavelengths, ref);
plt.plot(1e6*wavelengths, trans);

plt.show()

print(Wr.shape)
