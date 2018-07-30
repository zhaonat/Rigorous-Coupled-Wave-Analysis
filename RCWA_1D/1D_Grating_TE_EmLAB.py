import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import cond
import cmath;
from TMM_functions import eigen_modes as em
from TMM_functions import scatter_matrices as sm
from RCWA_functions import redheffer_star as rs
from RCWA_functions import rcwa_initial_conditions as ic
from RCWA_functions import homogeneous_layer as hl
from scipy import linalg as LA
# Moharam et. al Formulation for stable and efficient implementation for RCWA
plt.close("all")
'''
1D TE implementation of PLANAR DIFFRACTiON...the easy case
only: sign convention is exp(-ikr) (is the positive propagating wave), so loss is +  not - 
'''

def grating_fourier_harmonics(order, fill_factor, n_ridge, n_groove):
    """ function comes from analytic solution of a step function in a finite unit cell"""
    if(order == 0):
        return n_ridge**2*fill_factor + n_groove**2*(1-fill_factor);
    else:
        #should it be 1-fill_factor or fill_factor?
        return(n_ridge**2 - n_groove**2)*np.sin(np.pi*order*(1-fill_factor))/(np.pi*order);

def grating_fourier_array(num_ord, fill_factor, n_ridge, n_groove):
    fourier_comps = list();
    for i in range(-num_ord, num_ord+1):
        fourier_comps.append(grating_fourier_harmonics(i, fill_factor, n_ridge, n_groove));
    return fourier_comps;

def fourier_reconstruction(x, period, num_ord, n_ridge, n_groove, fill_factor = 0.5):
    index = np.arange(-num_ord, num_ord+1);
    f = 0;
    for n in index:
        coef = grating_fourier_harmonics(n, fill_factor, n_ridge, n_groove);
        f+= coef*np.exp(cmath.sqrt(-1)*np.pi*n*x/period);
    return f;

x = np.linspace(0,1,1000);
period = 1;
# plt.plot(x, np.real(fourier_reconstruction(x, period, 1000, 1,np.sqrt(12), fill_factor = 0.1)));
# plt.title('check that the analytic fourier series works')
# #'note that the lattice constant tells you the length of the ridge'
# plt.show()


L0 = 1e-6;
e0 = 8.854e-12;
mu0 = 4*np.pi*1e-8;

num_ord = 20;
indices = np.arange(-num_ord, num_ord+1)

n_ridge = 1; #np.sqrt(12);       # ridge
n_groove = np.sqrt(12); # groove
lattice_constant = 0.5* L0;  # SI units
d = 0.2* L0;  # thickness

theta_inc = 0;
spectra = list();
spectra_T = list();

wavelength_scan = np.linspace(0.5,2,200)
## construct permittivity harmonic components E
Nx = 256;
#fill factor = 0 is complete dielectric, 1 is air
fill_factor = 0.5 # 50% of the unit cell is the ridge material
fourer_orders = 2 * Nx + 1;
fourier_array = grating_fourier_array(Nx, fill_factor, n_ridge, n_groove);

##construct convolution matrix
E = np.zeros((2 * num_ord + 1, 2 * num_ord + 1))
p0 = Nx;
q0 = Nx;  # we don't need these centering offsets (or it's equivelant if we don't)
p_index = np.arange(-num_ord, num_ord + 1);
q_index = np.arange(-num_ord, num_ord + 1);

for prow in range(2 * num_ord + 1):
    # first term locates z plane, 2nd locates y coumn, prow locates x
    for pcol in range(2 * num_ord + 1):
        pfft = p_index[prow] - p_index[pcol];
        E[prow, pcol] = fourier_array[p0 + pfft];  # fill conv matrix from top left to top right

# E is now the convolution of fourier amplitudes

for wave in wavelength_scan:
    lam0 = wave*L0; #free space wavelength in SI units
    print('wavelength: '+str(wave));
    k0 = 2*np.pi/lam0;
    ## =====================STRUCTURE======================##

    ## Region I
    n1 = 1;#cmath.sqrt(-1)*1e-12; #apparently small complex perturbations are bad in Region 1, these shouldn't be necessary

    ## Region 2;
    n2 = 1+cmath.sqrt(-1)*1e-12;

    #
    kx_array = k0*(n1*np.sin(theta_inc)- indices*(lam0 / lattice_constant)); #0 is one of them, k0*lam0 = 2*pi

    ## IMPLEMENT SCALING: these are the fourier orders of the x-direction decomposition.
    KX = np.diag(kx_array/k0); #singular since we have a n=0, m= 0 order and incidence is normal

    ## construct matrix of Gamma^2 ('constant' term in ODE):
    PQ = KX**2 - E; #conditioning of this matrix is not bad, A SHOULD BE SYMMETRIC
    #sum of a symmetric matrix and a diagonal matrix should be symmetric;

    eigenvals, W = LA.eig(PQ); #A is negative symmetric...but eigh only works on positive symmetric

    #conditioning of q is not good
    q = np.conj(np.sqrt((eigenvals.astype('complex')))); #in the paper, it says 'positive square root', as if it expects the numbers to be real > 0
                                 # typically, it is this array which has a huge range of values, which makes life shitty

    # plt.figure()
    # plt.plot(q);
    # plt.title('eigenvalues')
    # plt.show()

    ## ================================================================================================##
    #exp(iQr), negative sign for negative sign convention
    Q = np.diag(-q); #SIGN OF THE EIGENVALUES IS HUGELY IMPORTANT, but why is it negative for this?
    ## ================================================================================================##


    V = np.matmul(W,Q); #H field modes

    # print('conditioning analysis');
    # print(lam0)
    # print(cond(W))
    # print(cond(V)) #bad conditioning
    # print(cond(Q)) #bad

    # plt.figure();
    # plt.imshow(abs(W));
    # plt.title('E-field modes') #sort of has a messed up cross shape...
    # plt.show()

    kz_inc = n1;

    ## scattering matrix needed for 'gap medium'
    #if calculations shift with changing selection of gap media, this is BAD; it should not shift with choice of gap
    Wg,Vg, Kzg = hl.homogeneous_1D(KX, 1.1, m_r = 1)
    ## reflection medium
    Wr,Vr, Kzr = hl.homogeneous_1D(KX, 1, m_r = 1)

    ## transmission medium;
    Wt,Vt, Kzt = hl.homogeneous_1D(KX, 1, m_r = 1)

    ## S matrices for the reflection region
    # the order of W and Vg are important
    Ar, Br = sm.A_B_matrices(Wg, Wr, Vg, Vr);
    S_ref, Sr_dict = sm.S_R(Ar, Br);  # scatter matrix for the reflection region    ## calculating A and B matrices for scattering matrix

    ## define S matrix for the grating
    #A, B = sm.A_B_matrices(Wg, np.matrix(W), Vg, np.matrix(V));
    A, B = sm.A_B_matrices(np.matrix(W),Wg,  np.matrix(V), Vg);

    S, S_dict = sm.S_layer(A, B, d, k0, Q)

    ## define S matrices for the Transmission region
    At, Bt = sm.A_B_matrices(Wg, Wt, Vg, Vt);
    St, St_dict = sm.S_T(At, Bt); #scatter matrix for the reflection region

    ## construct global scattering matrices
    Sg = Sr_dict;
    Sg_matrix, Sg = rs.RedhefferStar(Sg, S_dict);
    Sg_m, Sg = rs.RedhefferStar(Sg, St_dict)

    #check scattering matrix is unitary
    print(np.linalg.norm(Sg_m*Sg_m.I - np.matrix(np.eye(2*(2*num_ord+1)))))

    ## ======================== CALCULATE R AND T ===============================##
    K_inc_vector = k0 * np.matrix([np.sin(theta_inc), \
                                         0, np.cos(theta_inc)]);

    # print(cinc.shape)
    # print(cinc)
    cinc = np.zeros((2*num_ord+1, ));
    cinc[num_ord] = 1;
    cinc = np.matrix(cinc).T
    cinc = Wr.I * cinc;
    ## COMPUTE FIELDS: similar idea but more complex for RCWA since you have individual modes each contributing
    reflected = Wr * Sg['S11'] * cinc;
    transmitted = Wt * Sg['S21'] * cinc;

    ## reflected is already ry or Ey
    rsq = np.power(abs(reflected),2);

    ## compute final reflectivity
    Rdiff = Kzr*rsq/kz_inc; #reshape this?
    R = np.sum(Rdiff);

    print(R);
    spectra.append(R); #spectra_T.append(T);
    spectra_T.append(1-R)
plt.figure();
plt.plot(wavelength_scan, spectra);
plt.plot(wavelength_scan, spectra_T)
plt.legend(['reflection', 'transmission'])
plt.show()



# ## now we can form the linear system to solve for the amplitude coeffs
# WV1 = np.block([[W, np.matmul(W,X)],[V, -np.matmul(V,X)]]);
#
# WV2 = np.block([[np.matmul(W,X), W],[np.matmul(V,X), -V]])
#
# print(WV1.shape)
# '''
# the conditioning of WV1 and WV2 cannot suck because we're going to invert one of them.
# '''
# print('condition of WV1: '+str(np.linalg.cond(WV1))) #condition number of this is HUGE!!! caused by X
# print('condition of WV2: '+str(np.linalg.cond(WV2))) #condition number of this is HUGE!!!
#
# F1 = np.linalg.inv(WV1);


