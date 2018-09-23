import numpy as np
import matplotlib.pyplot as plt
from convolution_matrices import convmat2D as cm
from PWEM_functions import K_matrix as km
from PWEM_functions import PWEM_eigen_problem as eg
import cmath
'''
solve PWEM for the infintie waveguide array
and generate band structure; results should be consistent with FDFD hybro grating project
'''

eps0 = 8.854e-12;
mu0 = 4*np.pi*1e-7;
c0 = 3e8;
L0 = 1e-6; #SI unit scaling
## lattice and material parameters
ax = 0.2; #units of microns
ay = 0.5;

#generate irreducible BZ sample
T1 = 2*np.pi/ax;
T2 = 2*np.pi/ay;

# determine number of orders to use
P = 1; #don't play with odd orders
Q = 1;
PQ = (2*P+1)*(2*Q+1)
# ============== build high resolution waveguide array, non-dispersive ==================
Nx = 512;
Ny = 512;


## ======================== run band structure calc ==========================##
kx_scan = np.linspace(1e-1, np.pi, 200)/ax;
eps_tracker = [];
omega_eig_store = [];
omega_p = 0.72*np.pi*1e15;
gamma = 5.5e12;
for beta_x in kx_scan:
    beta_y = 0;
    Kx, Ky = km.K_matrix_cubic_2D(beta_x, beta_y, ax, ay, P, Q);

    #from here, we can actually dtermine omega from our specification of kx and ky
    # omega = c0*np.sqrt(beta_x**2+beta_y**2)/L0;
    #
    # ## we can do a determination of a dispersive medium from here
    # #drude for example
    #eps_drude = 1-omega_p**2/(omega**2-cmath.sqrt(-1)*gamma*omega);
    #print(eps_drude); eps_tracker.append(eps_drude);

    epsilon = np.ones((Nx, Ny));
    halfy = int(Ny / 2);
    epsilon[halfy - 100:halfy + 100, :] = -12;

    ## =============== Convolution Matrices ==============
    E_r = cm.convmat2D(epsilon, P, Q);

    ## ===========================================================

    eigenvalues, eigenvectors, A_matrix = eg.PWEM2D_TE(Kx, Ky, E_r); #what are the eigenvalue units
    omega_eig_store.append(np.sqrt(abs(np.real(eigenvalues))))
    plt.plot(beta_x*np.ones((PQ,)), np.sort(np.sqrt(abs(np.real(eigenvalues)))), '.r')
    plt.plot(beta_x*np.ones((PQ,)), np.sort(np.sqrt(abs(np.imag(eigenvalues)))), '.g')

## plot the light line

plt.plot(kx_scan, abs(kx_scan));

plt.show()
    # question: which eigenvalues are relevant for plotting the band structure?

