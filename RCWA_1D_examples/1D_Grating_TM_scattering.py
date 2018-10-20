import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import cond
import cmath;
from numpy.linalg import solve as bslash
from scipy.fftpack import fft, fftfreq, fftshift
from TMM_functions import eigen_modes as em
from TMM_functions import scatter_matrices as sm
from RCWA_functions import redheffer_star as rs
from RCWA_functions import rcwa_initial_conditions as ic
from RCWA_functions import homogeneous_layer as hl
from scipy import linalg as LA
# Moharam et. al Formulation for stable and efficient implementation for RCWA
plt.close("all")


np.set_printoptions(precision = 4)
def grating_fourier_harmonics(order, fill_factor, n_ridge, n_groove):
    """ function comes from analytic solution of a step function in a finite unit cell"""
    #n_ridge = index of refraction of ridge (should be dielectric)
    #n_ridge = index of refraction of groove (air)
    #n_ridge has fill_factor
    #n_groove has (1-fill_factor)
    # there is no lattice constant here, so it implicitly assumes that the lattice constant is 1...which is not good

    if(order == 0):
        return n_ridge**2*fill_factor + n_groove**2*(1-fill_factor);
    else:
        #should it be 1-fill_factor or fill_factor?, should be fill_factor
        return(n_ridge**2 - n_groove**2)*np.sin(np.pi*order*(fill_factor))/(np.pi*order);

def grating_fourier_array(num_ord, fill_factor, n_ridge, n_groove):
    """ what is a convolution in 1D """
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
        #f+=coef*np.cos(np.pi*n*x/period)
    return f;

def fourier_reconstruction_general(x, period, num_ord, coefs):
    '''
    overloading odesn't work in python...fun fact, since it is dynamically typed (vs statically typed)
    :param x:
    :param period:
    :param num_ord:
    :param coefs:
    :return:
    '''
    index = np.arange(-num_ord, num_ord+1);
    f = 0; center = int(len(coefs)/2); #no offset
    for n in index:
        coef = coefs[center+n];
        f+= coef*np.exp(cmath.sqrt(-1)*2*np.pi*n*x/period);
    return f;

def grating_fft(eps_r):
    assert len(eps_r.shape) == 2
    assert eps_r.shape[1] == 1;
    #eps_r: discrete 1D grid of the epsilon profile of the structure
    fourier_comp = np.fft.fftshift(np.fft.fft(eps_r, axis = 0)/eps_r.shape[0]);
    #ortho norm in fft will do a 1/sqrt(n) scaling
    return np.squeeze(fourier_comp);

# plt.plot(x, np.real(fourier_reconstruction(x, period, 1000, 1,np.sqrt(12), fill_factor = 0.1)));
# plt.title('check that the analytic fourier series works')
# #'note that the lattice constant tells you the length of the ridge'
# plt.show()

L0 = 1e-6;
e0 = 8.854e-12;
mu0 = 4*np.pi*1e-8;
fill_factor = 0.3; # 50% of the unit cell is the ridge material


num_ord = 10; #INCREASING NUMBER OF ORDERS SEEMS TO CAUSE THIS THING TO FAIL, to many orders induce evanescence...particularly
               # when there is a small fill factor
PQ = 2*num_ord+1;
indices = np.arange(-num_ord, num_ord+1)

n_ridge = 3.48; #3.48;              # ridge
n_groove = 1;                # groove (unit-less)
lattice_constant = 0.7;  # SI units
# we need to be careful about what lattice constant means
# in the gaylord paper, lattice constant exactly means (0, L) is one unit cell


d = 0.46;               # thickness, SI units

Nx = 2*256;
eps_r = n_groove**2*np.ones((2*Nx, 1)); #put in a lot of points in eps_r
border = int(2*Nx*fill_factor);
eps_r[0:border] = n_ridge**2;
fft_fourier_array = grating_fft(eps_r);
x = np.linspace(-lattice_constant,lattice_constant,1000);
period = lattice_constant;
fft_reconstruct = fourier_reconstruction_general(x, period, num_ord, fft_fourier_array);

fourier_array_analytic = grating_fourier_array(Nx, fill_factor, n_ridge, n_groove);
analytic_reconstruct = fourier_reconstruction(x, period, num_ord, n_ridge, n_groove, fill_factor)


plt.figure();
plt.plot(np.real(fft_fourier_array[Nx-20:Nx+20]), linewidth=2)
plt.plot(np.real(fourier_array_analytic[Nx-20:Nx+20]));
plt.legend(('fft', 'analytic'))
plt.show()

plt.figure();
plt.plot(x,fft_reconstruct)
plt.plot(x,analytic_reconstruct);
plt.legend(['fft', 'analytic'])
plt.show()
theta_inc = 0;
spectra = list();
spectra_T = list();

wavelength_scan = np.linspace(0.5,2.3,300)
## construct permittivity harmonic components E
#fill factor = 0 is complete dielectric, 1 is air


##construct convolution matrix
E_conv = np.zeros((2 * num_ord + 1, 2 * num_ord + 1));
E_conv = E_conv.astype('complex')
p0 = Nx;
p_index = np.arange(-num_ord, num_ord + 1);
fourier_array = fft_fourier_array; #_analytic;
for prow in range(2 * num_ord + 1):
    # first term locates z plane, 2nd locates y coumn, prow locates x
    for pcol in range(2 * num_ord + 1):
        pfft = p_index[prow] - p_index[pcol];
        E_conv[prow, pcol] = fourier_array[p0 + pfft];  # fill conv matrix from top left to top right

## FFT of 1/e;
inv_fft_fourier_array = grating_fft(1/eps_r);
##construct convolution matrix
E_conv_inv = np.zeros((2 * num_ord + 1, 2 * num_ord + 1));
E_conv_inv = E_conv_inv.astype('complex')
p0 = Nx;
p_index = np.arange(-num_ord, num_ord + 1);
for prow in range(2 * num_ord + 1):
    # first term locates z plane, 2nd locates y coumn, prow locates x
    for pcol in range(2 * num_ord + 1):
        pfft = p_index[prow] - p_index[pcol];
        E_conv_inv[prow, pcol] = inv_fft_fourier_array[p0 + pfft];  # fill conv matrix from top left to top right

I = np.identity(2*num_ord+1);
# E is now the convolution of fourier amplitudes
for lam0 in wavelength_scan:
    k0 = 2*np.pi/lam0; #free space wavelength in SI units
    print('wavelength: ' + str(lam0));
    ## =====================STRUCTURE======================##

    ## Region I
    n1 = 1;#cmath.sqrt(-1)*1e-12; #apparently small complex perturbations are bad in Region 1, these shouldn't be necessary
    ## Region 2; transmission
    n2 = 1;

    #from the kx_components given the indices and wvln
    #2 * np.pi * np.arange(-N_p, N_p + 1) / (k0 * a_x)
    indices = np.arange(-num_ord, num_ord + 1)

    kx_array = (n1*np.sin(theta_inc) + indices*(lam0 / lattice_constant)); #0 is one of them, k0*lam0 = 2*pi

    ## IMPLEMENT SCALING: these are the fourier orders of the x-direction decomposition.
    KX = np.diag(kx_array); #singular since we have a n=0, m= 0 order and incidence is normal

    ## construct matrix of Gamma^2 ('constant' term in ODE):
    #use fast fourier factorization rules here...
    Q = I- KX @ bslash(E_conv, KX);
    PQ = -bslash(E_conv_inv, Q);

    ## ================================================================================================##
    eigenvals, W = LA.eigh(PQ); #A should be symmetric or hermitian
    #we should be gauranteed that all eigenvals are REAL
    eigenvals = eigenvals.astype('complex');
    lambda_matrix = np.diag((np.sqrt((eigenvals))));

    ## THIS NEGATIVE SIGN IS CRUCIAL, but I'm not sure why
    V = -em.eigen_V(Q, W, lambda_matrix)
    kz_inc = n1;
    ## ================================================================================================##

    ## scattering matrix needed for 'gap medium'
    #if calculations shift with changing selection of gap media, this is BAD; it should not shift with choice of gap
    Wg,Vg, Kzg = hl.homogeneous_1D(KX, 1, m_r = 1)
    ## reflection medium
    Wr,Vr, Kzr = hl.homogeneous_1D(KX, 1, m_r = 1)
    ## transmission medium;
    Wt,Vt, Kzt = hl.homogeneous_1D(KX, 1, m_r = 1)

    ## S matrices for the reflection region
    #Ar, Br = sm.A_B_matrices(Wg, Wr, Vg, Vr);
    Ar, Br = sm.A_B_matrices_half_space(Wr, Wg, Vr, Vg);  # make sure this order is right

    S_ref, Sr_dict = sm.S_R(Ar, Br);  # scatter matrix for the reflection region    ## calculating A and B matrices for scattering matrix
    Sg = Sr_dict;

    ## define S matrix for the GRATING REGION
    A, B = sm.A_B_matrices(W, Wg,  V, Vg);
    S, S_dict = sm.S_layer(A, B, d, k0, lambda_matrix)
    Sg_matrix, Sg = rs.RedhefferStar(Sg, S_dict)

    ## define S matrices for the Transmission region
    At, Bt = sm.A_B_matrices_half_space(Wt, Wg, Vt, Vg);  # make sure this order is right
    St, St_dict = sm.S_T(At, Bt); #scatter matrix for the reflection region
    Sg_matrix, Sg = rs.RedhefferStar(Sg, St_dict)

    #check scattering matrix is unitary
    #print(np.linalg.norm(np.linalg.inv(Sg_matrix)@Sg_matrix - np.matrix(np.eye(2*(2*num_ord+1)))))

    ## ======================== CALCULATE R AND T ===============================##
    K_inc_vector =  n1*np.matrix([np.sin(theta_inc), \
                                         0, np.cos(theta_inc)]);
    #K_inc isn't even used for anyting...

    #cinc is the incidence vector
    cinc = np.zeros((2*num_ord+1, )); #only need one set...
    cinc[num_ord] = 1;
    cinc = cinc.T;
    cinc = np.linalg.inv(Wr) @ cinc;
    ## COMPUTE FIELDS: similar idea but more complex for RCWA since you have individual modes each contributing
    reflected = Wr @ Sg['S11'] @ cinc;
    transmitted = Wt @ Sg['S21'] @ cinc;

    ## reflected is already ry or Ey
    rsq = np.square(np.abs(reflected));
    tsq = np.square(np.abs(transmitted));

    ## compute final reflectivity
    Rdiff = np.real(Kzr)@rsq/np.real(kz_inc); #real because we only want propagating components
    Tdiff = np.real(Kzt)@tsq/np.real(kz_inc)
    R = np.sum(Rdiff);
    T = np.sum(Tdiff);

    print(R);
    spectra.append(R); #spectra_T.append(T);
    spectra_T.append(T)

plt.figure();
plt.plot(wavelength_scan, spectra);
plt.plot(wavelength_scan, spectra_T)
plt.plot(wavelength_scan, np.array(spectra)+np.array(spectra_T))
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


