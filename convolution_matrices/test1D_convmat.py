import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA


'''
1D convolution matrices are always toeplitz
'''

def grating_fourier_harmonics(order, fill_factor, n_ridge, n_groove):
    if(order == 0):
        return n_ridge**2*fill_factor + n_groove**2*(1-fill_factor);
    else:
        return(n_ridge**2 - n_groove**2)*np.sin(np.pi*order*fill_factor)/(np.pi*order);

def grating_fourier_array(num_ord, fill_factor, n_ridge, n_groove):
    fourier_comps = list();
    for i in range(-num_ord, num_ord+1):
        fourier_comps.append(grating_fourier_harmonics(i, fill_factor, n_ridge, n_groove));

    return fourier_comps;


L0 = 1e-6;
e0 = 8.854e-12;
mu0 = 4*np.pi*1e-8;

Nx = 80; #Nx has to be sufficiently LARGE
num_ord = 20;
indices = np.arange(-num_ord, num_ord+1)

n_ridge = 1;       # ridge
n_groove = 12; # groove

theta_inc = 0;

lam0 = 2.3*L0; #free space wavelength
k0 = 2*np.pi/lam0;

## =====================STRUCTURE======================##
lattice_constant = 1 * L0;
fill_factor = 0.5; #50% of the unit cell is the ridge material
d = 1*L0; #thickness

## Region I
n1 = 1;

## Region 2;
n2 = n_ridge;

kx_array = k0*n1*np.sin(theta_inc)- indices*(lam0 / lattice_constant);

KX = np.diag(kx_array/k0); #singular since we have a n=0, m= 0 order and incidence is normal

## construct permittivity harmonic components E
fourier_array = np.array(grating_fourier_array(Nx, fill_factor, n_ridge, n_groove));
plt.plot(fourier_array);
plt.show()

## check with fft
# Nx = 20; N_groove = int(Nx*fill_factor);
#
# dielectric_dist = n_ridge * np.ones((Nx, 1))
# print(dielectric_dist.shape)
# dielectric_dist[0:N_groove] = n_groove;

##construct convolution matrix
E = np.zeros((2*num_ord+1,2*num_ord+1))

padding = np.zeros(fourier_array.shape[0] - 1, fourier_array.dtype)
first_col = np.r_[fourier_array, padding]
first_row = np.r_[fourier_array[0], padding]
H = LA.toeplitz(first_col, first_row); #this type of toeplitz isn't square;


plt.figure();
p0 = int(len(fourier_array)/2);
p = np.arange(-num_ord, num_ord+1);
q = p
for prow in range(2*num_ord+1):
    # first term locates z plane, 2nd locates y coumn, prow locates x
    for pcol in range(2*num_ord+1):
        pfft = p[prow] - p[pcol];

        E[prow, pcol] = fourier_array[p0+pfft];
    plt.plot(E[prow, :])
plt.show();

## Bo's formalism; which is just to say that every row of the convmat
# is the fourier array shifted by some value
ordMax = num_ord; ordMin = -num_ord
E_check = np.zeros_like(E);
fourier_restriction = fourier_array[p0-2*num_ord:p0+2*num_ord+1]
for i in range(ordMax-ordMin+1):
    ind_start = ordMax-ordMin-i;
    ind_end = ordMax-ordMin-i+1+2*num_ord;
    E_check[i,:] = fourier_restriction[ind_start:ind_end];

#FROM EMLAB, a 1D convolution is always toeplitz
plt.figure();
plt.imshow(E);
plt.show();

print('is symmetric?')
print(np.linalg.norm(E-E.T))
