## same as the analytic case but with the fft
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import cond
import cmath;
from scipy import linalg as LA
from numpy.linalg import solve as bslash
import time
from convolution_matrices.convmat1D import *
from RCWA_1D_functions.grating_fft.grating_conv import *

def nonHermitianEigenSorter(eigenvalues):
    N = len(eigenvalues);
    sorted_indices=[];
    sorted_eigs = [];
    for i in range(N):
        eig = eigenvalues[i];
        if(np.real(eig)>0 and np.imag(eig) == 0):
            sorted_indices.append(i); sorted_eigs.append(eig);
        elif(np.real(eig)==0 and np.imag(eig) > 0):
            sorted_indices.append(i); sorted_eigs.append(eig);
        elif(np.real(eig)>0 and abs(np.imag(eig)) > 0):
            sorted_indices.append(i); sorted_eigs.append(eig);
    return sorted_eigs, sorted_indices;

# Moharam et. al Formulation for stable and efficient implementation for RCWA
plt.close("all")
'''
1D TM implementation of PLANAR DIFFRACTiON
STILL NOT WORKING YET

only: sign convention is exp(-ikr) (is the positive propagating wave), so loss is +  not - 
source for fourier decomps is from the paper: Formulation for stable and efficient implementation of
the rigorous coupled-wave analysis of binary gratings by Moharam et. al
'''

# plt.plot(x, np.real(fourier_reconstruction(x, period, 1000, 1,np.sqrt(12), fill_factor = 0.1)));
# plt.title('check that the analytic fourier series works')
# #'note that the lattice constant tells you the length of the ridge'
# plt.show()

L0 = 1e-6;
e0 = 8.854e-12;
mu0 = 4*np.pi*1e-8;
fill_factor = 0.3; # 50% of the unit cell is the ridge material


num_ord = 3; #INCREASING NUMBER OF ORDERS SEEMS TO CAUSE THIS THING TO FAIL, to many orders induce evanescence...particularly

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

## simulation parameters
theta = (0)*np.pi/180;
spectra = list();
spectra_T = list();

wavelength_scan = np.linspace(0.5, 2, 100)
## construct permittivity harmonic components E
#fill factor = 0 is complete dielectric, 1 is air

##construct convolution matrix
Ezz = np.zeros((2 * num_ord + 1, 2 * num_ord + 1)); Ezz = Ezz.astype('complex')
p0 = Nx; #int(Nx/2);
p_index = np.arange(-num_ord, num_ord + 1);
q_index = np.arange(-num_ord, num_ord + 1);
fourier_array = fft_fourier_array;#fourier_array_analytic;
detected_pffts = np.zeros_like(Ezz);
for prow in range(2 * num_ord + 1):
    # first term locates z plane, 2nd locates y coumn, prow locates x
    row_index = p_index[prow];
    for pcol in range(2 * num_ord + 1):
        pfft = p_index[prow] - p_index[pcol];
        detected_pffts[prow, pcol] = pfft;
        Ezz[prow, pcol] = fourier_array[p0 + pfft];  # fill conv matrix from top left to top right

Exz = np.zeros_like(Ezz);
Ezx = -np.zeros_like(Ezz);
Exz = 0.2*np.eye(PQ)
Ezx = Exz;
print((Exz.shape, Ezx.shape, Ezz.shape))

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

## IMPORTANT TO NOTE: the indices for everything beyond this points are indexed from -num_ord to num_ord+1
## alternate construction of 1D convolution matrix

PQ =2*num_ord+1;
I = np.eye(PQ)
zeros = np.zeros((PQ, PQ))
# E is now the convolution of fourier amplitudes

for wvlen in wavelength_scan:
    j = cmath.sqrt(-1);
    lam0 = wvlen;     k0 = 2 * np.pi / lam0; #free space wavelength in SI units
    print('wavelength: ' + str(wvlen));
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

    # PQ_block = np.block([[zeros, np.linalg.inv(E_conv_inv)],[KX@bslash(E, KX) - I, zeros]])
    # # plt.imshow(np.abs(PQ_block));
    # # plt.show();
    # print('condition of PQ block: '+str(np.linalg.cond(PQ_block)))
    # big_eigenvals, bigW = LA.eig(PQ_block);
    # print((bigW.shape, big_eigenvals.shape))
    # Wp = bigW[0:PQ, PQ:]
    # plt.imshow(abs(bigW))
    # plt.show();
    ## construct matrix of Gamma^2 ('constant' term in ODE):

    ## one thing that isn't obvious is that are we doing element by element division or is it matricial
    B = (KX@bslash(Ezz, KX) - I);
    bE = np.linalg.inv(E_conv_inv) + bslash(Ezz,(Exz@Ezx)); #/Ezz;
    G = j*bslash(Ezz,Ezx) @ KX;
    H = j*KX @bslash(Ezz, Exz);
    #print((G,H))
    print((bE.shape,G.shape, H.shape))
    print((np.linalg.cond(B), np.linalg.cond(bE)))


    M = np.linalg.inv(bE);
    K = -(B + H@np.linalg.inv(bE)@G);
    C = -np.linalg.inv(bE)@G - H@np.linalg.inv(bE);
    Z = np.zeros_like(M);
    I = np.eye(M.shape[0], M.shape[1]);
    OA = np.block([[M, Z],[Z, I]])
    OB = np.block(np.block([[C, K],[-I, Z]]))

    ## these matrices aren't poorly conditioned
    print((np.linalg.cond(OA), np.linalg.cond(OB)))
    ## solve eiegenvalues;
    beigenvals, bigW = LA.eig(OB, OA); #W contains eigenmodes of the form (lambda x, x)

    ## AT THIS POINT, we have still extracted TWO times the number of eigenvalues...
    #try rounding...
    rounded_beigenvals = np.array([round(i,8) for i in beigenvals])
    print(rounded_beigenvals)
    #quadrant_sort = [1 if abs(np.real(i))>=0 and np.imag(i)>=0 else 0 for i in rounded_beigenvals];

    sorted_eigs, sorted_indices = nonHermitianEigenSorter(rounded_beigenvals)
    sorted_indices = np.nonzero(sorted_indices)[0];
    #print(quadrant_sort)
    # sorted_indices = np.nonzero(quadrant_sort)[0]
    print(len(sorted_indices))
    #sorted_indices = np.argsort(np.real(rounded_beigenvals))
    sorted_eigenmodes = bigW[:, sorted_indices];
    #print(sorted_eigenmodes)
    #adding real and imaginary parts seems to work...
    sorted_eigenvals = beigenvals[sorted_indices]
    print(sorted_eigenvals)
    W = sorted_eigenmodes[PQ:,:]
    eigenvals_wp = (sorted_eigenvals[0:PQ]);

    # plt.subplot(121)
    # plt.plot(np.real(beigenvals), np.imag(beigenvals), '.', markersize = 20); plt.title('1st');
    # plt.subplot(122)
    # plt.plot(np.real(beigenvals), np.imag(beigenvals), '.', markersize = 20);
    # plt.plot(np.real(eigenvals_wp), (np.imag(eigenvals_wp)), '.r', markersize = 10)
    # plt.show();
    # ##
    Q = np.diag(eigenvals_wp); #eigenvalue problem is for kz, not kz^2

    V = np.linalg.inv(bE)@(W @ Q + H @ W);

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
    print((W.shape, V.shape))
    
    #this appears to be worse and worse conditioned at higher orders...
    O = np.block([
        [W, W],
        [V,-V]
    ]); #this is much better conditioned than S..
    print('condition of O: '+str(np.linalg.cond(O)))
    print((np.linalg.cond(W), np.linalg.cond(V)))
    # plt.imshow(abs(O))
    # plt.show();
    f = I;
    g = j * Z_II; #all matrices
    fg = np.concatenate((f,g),axis = 0)
    ab = np.matmul(np.linalg.inv(O),fg);
    a = ab[0:PQ,:];
    b = ab[PQ:,:];

    term = X @ a @ np.linalg.inv(b) @ X;
    f = W @ (I + term);
    g = V@(-I+term);
    T = np.linalg.inv(np.matmul(j*Z_I, f) + g);
    T = np.dot(T, (np.dot(j*Z_I, delta_i0) + n_delta_i0));
    R = np.dot(f,T)-delta_i0; #shouldn't change
    T = np.dot(np.matmul(np.linalg.inv(b),X),T)

    ## calculate diffraction efficiencies
    #I would expect this number to be real...
    DE_ri = R*np.conj(R)*np.real(np.expand_dims(k_I,1))/(k0*n1*np.cos(theta));
    DE_ti = T*np.conj(T)*np.real(np.expand_dims(k_II,1)/n2**2)/(k0*np.cos(theta)/n1);

    print('R(lam)='+str(np.sum(DE_ri))+' T(lam) = '+str(np.sum(DE_ti)))
    spectra.append(np.sum(DE_ri)); #spectra_T.append(T);
    spectra_T.append(np.sum(DE_ti))

spectra = np.array(spectra);
spectra_T = np.array(spectra_T)
plt.figure();
plt.plot(wavelength_scan, spectra);
plt.plot(wavelength_scan, spectra_T)
plt.plot(wavelength_scan, spectra+spectra_T)
# plt.legend(['reflection', 'transmission'])
# plt.axhline(((3.48-1)/(3.48+1))**2,xmin=0, xmax = max(wavelength_scan))
# plt.axhline(((3.48-1)/(3.48+1)),xmin=0, xmax = max(wavelength_scan), color='r')
#
plt.show()






