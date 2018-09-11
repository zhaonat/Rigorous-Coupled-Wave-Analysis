import numpy as np
import matplotlib.pyplot as plt
from convolution_matrices import convmat2D as cm
from RCWA_functions import run_RCWA_simulation as rrs
import cmath
from numpy.linalg import cond
plt.close("all")
'''
RCWA testing with an IMI (ins, metal ins) grating, which should match the spectra we showed in the RCWA_1D code.
as in, if you make a 1D metallic strip and work in the TE polarization, you should notice a band-gap or a reflection
 stop band
'''

#% General Units
degrees = np.pi/180;
L0 = 1e-6; #units of microns;
eps0 = 8.854e-12*L0;
mu0 = 4*np.pi*10**-7*L0;
c0 = 1/(np.sqrt(mu0*eps0))

## lattice and material parameters
a = 0.2;
e_r = 16;

## Specify number of fourier orders to use:
#scalign with number of orders is pretty poor
N = 8; M = 8;

## =============== Simulation Parameters =========================
## set wavelength scanning range
#never want lattice constant and wavelength to match
wavelengths = np.linspace(0.9, 2.9, 300); #500 nm to 1000 nm #be aware of Wood's Anomalies

## drude parameters
## simulating a metal suffers errors ... even with loss added
omega_p = 0.72*np.pi*1e15;
gamma = 5.5e12;
epsilon_tracker = list();
ref = list(); tran = list();
for wvlen in wavelengths:
    print('wvlen: '+str(wvlen));
    omega = 2*np.pi*c0/(wvlen); #must be in SI for eps_drude

    #sign should be positive?
    eps_drude = 1-omega_p**2/(omega**2-cmath.sqrt(-1)*omega*gamma);
    epsilon_tracker.append(eps_drude);
    # ============== build high resolution circle ==================
    Nx = 512;
    Ny = 512;
    A = e_r* np.ones((Nx, Ny)); A = A.astype('complex')
    A2 = e_r* np.ones((Nx, Ny)); A2 = A2.astype('complex')
    x1 = int(Nx/2)-50; x2 = int(Nx/2)+50;
    y1 = int(Ny/2)-50; y2 = int(Ny/2)+50;

    A[:, y1:y2] = eps_drude;  ## A METALLIC HOLE...the fact that we have to recalculate the convolution
    A2[x1:x2,:] = eps_drude;
    # plt.imshow(np.abs(A));
    # plt.show();
    ## =============== Convolution Matrices ==============
    E_r = cm.convmat2D(A, N, M);
    E_r2= cm.convmat2D(A2,N,M);
    NM = (2 * N + 1) * (2 * M + 1);

    ## ================== GEOMETRY OF THE LAYERS AND CONVOLUTIONS ==================##
    thickness_slab = 0.2;  # in units of L0;
    ER = [E_r,e_r*np.identity(NM),E_r2];
    UR = [np.identity(NM), np.identity(NM),np.identity(NM)];
    layer_thicknesses = [thickness_slab, thickness_slab, thickness_slab];  # this retains SI unit convention

    #source parameters
    theta = 0 * degrees; #%elevation angle
    phi = 0 * degrees; #%azimuthal angle

    ## incident wave polarization
    normal_vector = np.array([0, 0, -1]) #positive z points down;
    ate_vector = np.array([0, 1, 0]); #vector for the out of plane E-field
    #ampltidue of the te vs tm modes (which are decoupled)
    pte = 1/np.sqrt(2);     #TE which is Hz, Ex, Ey
    ptm = cmath.sqrt(-1)/np.sqrt(2); #TM mode, which is Ez, Hx, Hy

    lattice_constants = [a, a];
    e_half = [1,1];
    R,T = rrs.run_RCWA_2D(wvlen, theta, phi, ER, UR, layer_thicknesses, lattice_constants, pte, ptm, N,M, e_half)
    ref.append(R);
    tran.append(T)
    print(R);


ref = np.array(ref);
tran = np.array(tran);
absorption = 1-(ref+tran);

plt.figure();
plt.plot(wavelengths, np.real(epsilon_tracker));
plt.plot(wavelengths, np.imag(epsilon_tracker))
plt.title('drude metal epsilon')

plt.figure();
plt.subplot(121);
plt.imshow(np.abs(A))
plt.subplot(122);
plt.imshow(np.abs(E_r));

plt.figure();
plt.plot(wavelengths, ref);
plt.plot(wavelengths, tran);
plt.plot(wavelengths, 1-(ref+tran))
plt.plot(wavelengths, ref+tran+absorption);
plt.legend(('ref', 'tran', 'abs'))

plt.show()

