
## same as the analytic case but with the fft
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import cond
import cmath;
from scipy.fftpack import fft, fftfreq, fftshift, rfft
from RCWA_1D_functions.TE_solver import *
from RCWA_1D_functions.TM_solver import *

# Moharam et. al Formulation for stable and efficient implementation for RCWA
plt.close("all")
'''
1D TE implementation of PLANAR DIFFRACTiON...the easy case
only: sign convention is exp(-ikr) (is the positive propagating wave), so loss is +  not - 
source for fourier decomps is from the paper: Formulation for stable and efficient implementation of
the rigorous coupled-wave analysis of binary gratings by Moharam et. al
'''
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


num_ord = 2; #INCREASING NUMBER OF ORDERS SEEMS TO CAUSE THIS THING TO FAIL, to many orders induce evanescence...particularly
               # when there is a small fill factor
PQ = 2*num_ord+1;
indices = np.arange(-num_ord, num_ord+1)

n_ridge = 3.48; #3.48;              # ridge
n_groove = 3.48;                # groove (unit-less)
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


##construct convolution matrix
E = np.zeros((2 * num_ord + 1, 2 * num_ord + 1)); E = E.astype('complex')
p0 = Nx; #int(Nx/2);
p_index = np.arange(-num_ord, num_ord + 1);
q_index = np.arange(-num_ord, num_ord + 1);
fourier_array = fft_fourier_array;#fourier_array_analytic;
detected_pffts = np.zeros_like(E);
for prow in range(2 * num_ord + 1):
    # first term locates z plane, 2nd locates y coumn, prow locates x
    row_index = p_index[prow];
    for pcol in range(2 * num_ord + 1):
        pfft = p_index[prow] - p_index[pcol];
        detected_pffts[prow, pcol] = pfft;
        E[prow, pcol] = fourier_array[p0 + pfft];  # fill conv matrix from top left to top right

## IMPORTANT TO NOTE: the indices for everything beyond this points are indexed from -num_ord to num_ord+1

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

theta_scan = np.linspace(0,90-0.01,501)*np.pi/180;
theta_spec = list();
theta_spec_T = list();
theta_TM = list();

for theta in theta_scan:
    print('theta: '+str(theta*180/np.pi))
    wavelength_scan=[1.1];
    R,T = RCWA_1D_TE(E, lattice_constant, theta, num_ord, wavelength_scan,d)
    R2, T2 = RWCA_1D_TM(E, E_conv_inv, lattice_constant, theta, num_ord, wavelength_scan, d)
    theta_spec.append(R[0]);
    theta_spec_T.append(T[0])
    theta_TM.append([R2[0],T2[0]])

theta_TM = np.array(theta_TM);
plt.figure();
plt.subplot(121)
plt.plot(theta_scan, theta_spec);
plt.plot(theta_scan, theta_spec_T);


plt.xlabel('angle (degrees)')
plt.ylabel('R,T')
plt.title('Angle scan for Gaylord TE Homogeneous')

plt.subplot(122)
plt.plot(theta_scan, theta_TM);
plt.xlabel('angle (degrees)')
plt.ylabel('R,T')
plt.title('Angle scan for Gaylord TM Homogeneous')

plt.show();