
## same as the analytic case but with the fft
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import cond
import cmath;
from RCWA_1D_functions.TE_solver import *
from RCWA_1D_functions.TM_solver import *
from RCWA_1D_functions.grating_fft.grating_conv import *
from convolution_matrices.convmat1D import *

# Moharam et. al Formulation for stable and efficient implementation for RCWA
plt.close("all")
'''
1D TE implementation of PLANAR DIFFRACTiON...the easy case
only: sign convention is exp(-ikr) (is the positive propagating wave), so loss is +  not - 
source for fourier decomps is from the paper: Formulation for stable and efficient implementation of
the rigorous coupled-wave analysis of binary gratings by Moharam et. al
'''
np.set_printoptions(precision = 4)

# plt.plot(x, np.real(fourier_reconstruction(x, period, 1000, 1,np.sqrt(12), fill_factor = 0.1)));
# plt.title('check that the analytic fourier series works')
# #'note that the lattice constant tells you the length of the ridge'
# plt.show()

L0 = 1e-6;
e0 = 8.854e-12;
mu0 = 4*np.pi*1e-8;
fill_factor = 0.3; # 50% of the unit cell is the ridge material


num_ord = 20; #INCREASING NUMBER OF ORDERS SEEMS TO CAUSE THIS THING TO FAIL, to many orders induce evanescence...particularly
               # when there is a small fill factor
PQ = 2*num_ord+1;
indices = np.arange(-num_ord, num_ord+1)

n_groove = 12;                # groove (unit-less)
lattice_constant = 0.2;  # SI units
# we need to be careful about what lattice constant means
# in the gaylord paper, lattice constant exactly means (0, L) is one unit cell


d = 0.46;               # thickness, SI units
Nx = 2*256;
eps_r = n_groove*np.ones((2*Nx, 1)); #put in a lot of points in eps_r
border = int(2*Nx*fill_factor);

theta_scan = np.linspace(0,90-0.01,501)*np.pi/180;
theta_spec = list();
theta_spec_T = list();
theta_TM = list();

theta = 0;
wavelength_scan = np.linspace(0.5, 3.2, 201);

## dispersive
I = np.identity(2 * num_ord + 1)
indices = np.arange(-num_ord, num_ord + 1);
spectra_R = [];
spectra_T = [];  # arrays to hold reflectivity and transmissitivity
omega_p = 0.72*np.pi*1e15;
gamma = 5e12;
eps_r= eps_r.astype('complex')
for wvlen in wavelength_scan:
    j = cmath.sqrt(-1);
    lam0 = wvlen;     k0 = 2 * np.pi / lam0; #free space wavelength in SI units
    omega = 2*np.pi*3e8/lam0*1e6;
    epsilon_metal = 1 - omega_p**2/(omega**2 +j*gamma*omega)
    #print(epsilon_metal)

    ## dispersive epsilon
    eps_r[0:border] = epsilon_metal;


    ##construct convolution matrix
    E = convmat1D(eps_r, PQ);
    ## IMPORTANT TO NOTE: the indices for everything beyond this points are indexed from -num_ord to num_ord+1
    ## FFT of 1/e;
    E_conv_inv = convmat1D(1 / eps_r, PQ);

    ## =====================STRUCTURE======================##

    ## Region I: reflected region (half space)
    n1 = 1;#cmath.sqrt(-1)*1e-12; #apparently small complex perturbations are bad in Region 1, these shouldn't be necessary

    ## Region 2; transmitted region
    n2 = 1;

    #from the kx_components given the indices and wvln
    kx_array = k0*(n1*np.sin(theta) + indices*(lam0 / lattice_constant)); #0 is one of them, k0*lam0 = 2*pi
    k_xi = kx_array;
    ## IMPLEMENT SCALING: these are the fourier orders of the x-direction decomposition.
    KX = np.diag((k_xi/k0)); #singular since we have a n=0, m= 0 order and incidence is normal

    ## construct matrix of Gamma^2 ('constant' term in ODE):
    A = np.linalg.inv(E_conv_inv)@(KX@bslash(E, KX) - I); #conditioning of this matrix is not bad, A SHOULD BE SYMMETRIC

    #sum of a symmetric matrix and a diagonal matrix should be symmetric;

    ##
    # when we calculate eigenvals, how do we know the eigenvals correspond to each particular fourier order?
    #eigenvals, W = LA.eigh(A); #A should be symmetric or hermitian, which won't be the case in the TM mode
    eigenvals, W = LA.eig(A);
    #we should be gauranteed that all eigenvals are REAL
    eigenvals = eigenvals.astype('complex');
    Q = np.diag(np.sqrt(eigenvals)); #Q should only be positive square root of eigenvals
    V = E_conv_inv@(W@Q); #H modes

    ## this is the great typo which has killed us all this time
    X = np.diag(np.exp(-k0*np.diag(Q)*d)); #this is poorly conditioned because exponentiation
    ## pointwise exponentiation vs exponentiating a matrix

    ## observation: almost everything beyond this point is worse conditioned
    k_I = k0**2*(n1**2 - (k_xi/k0)**2);                 #k_z in reflected region k_I,zi
    k_II = k0**2*(n2**2 - (k_xi/k0)**2);   #k_z in transmitted region
    k_I = k_I.astype('complex'); k_I = np.sqrt(k_I);
    k_II = k_II.astype('complex'); k_II = np.sqrt(k_II);
    Z_I = np.diag(k_I / (n1**2 * k0 ));
    Z_II = np.diag(k_II /(n2**2 * k0));
    delta_i0 = np.zeros((len(kx_array),1));
    delta_i0[num_ord] = 1;
    n_delta_i0 = delta_i0*j*np.cos(theta)/n1;

    ## design auxiliary variables: SEE derivation in notebooks: RCWA_note.ipynb
    # we want to design the computation to avoid operating with X, particularly with inverses
    # since X is the worst conditioned thing

    O = np.block([
        [W, W],
        [V,-V]
    ]); #this is much better conditioned than S..
    f = I;
    g = j * Z_II; #all matrices
    fg = np.concatenate((f,g),axis = 0)
    ab = np.matmul(np.linalg.inv(O),fg);
    a = ab[0:PQ,:];
    b = ab[PQ:,:];

    term = X @ a @ np.linalg.inv(b) @ X;
    f = W @ (I+term);
    g = V@(-I+term);
    T = np.linalg.inv(np.matmul(j*Z_I, f) + g);
    T = np.dot(T, (np.dot(j*Z_I, delta_i0) + n_delta_i0));
    R = np.dot(f,T)-delta_i0; #shouldn't change
    T = np.dot(np.matmul(np.linalg.inv(b),X),T)

    ## calculate diffraction efficiencies
    #I would expect this number to be real...
    DE_ri = R*np.conj(R)*np.real(np.expand_dims(k_I,1))/(k0*n1*np.cos(theta));
    DE_ti = T*np.conj(T)*np.real(np.expand_dims(k_II,1)/n2**2)/(k0*np.cos(theta)/n1);

    #print(np.sum(DE_ri))
    spectra_R.append(np.sum(DE_ri)); #spectra_T.append(T);
    spectra_T.append(np.sum(DE_ti))

spectra = np.array(spectra_R);
spectra_T = np.array(spectra_T)
plt.figure();
plt.plot(wavelength_scan, spectra);
plt.plot(wavelength_scan, spectra_T)
plt.plot(wavelength_scan, spectra+spectra_T)
# plt.legend(['reflection', 'transmission'])
# plt.axhline(((3.48-1)/(3.48+1))**2,xmin=0, xmax = max(wavelength_scan))
# plt.axhline(((3.48-1)/(3.48+1)),xmin=0, xmax = max(wavelength_scan), color='r')
#
plt.legend(('reflection', 'transmission', 'sum'))
plt.show()



