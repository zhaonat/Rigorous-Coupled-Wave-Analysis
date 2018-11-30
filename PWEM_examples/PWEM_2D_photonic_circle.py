import numpy as np
import matplotlib.pyplot as plt
from convolution_matrices import convmat2D as cm
from PWEM_functions import K_matrix as km
from PWEM_functions import PWEM_eigen_problem as eg
'''
solve PWEM for a simple circular structure in a square unit cell
and generate band structure
compare with CEM EMLab; also Johannopoulos book on Photonics
'''

## lattice and material parameters
a = 1;
radius = 0.2*a;
e_r = 8.9;
c0 = 3e8;
#generate irreducible BZ sample
T1 = 2*np.pi/a;
T2 = 2*np.pi/a;

# determine number of orders to use
P = 5;
Q = 5;
PQ = (2*P+1)*(2*Q+1)
# ============== build high resolution circle ==================
Nx = 512; Ny = 512;
A = np.ones((Nx,Ny));
ci = int(Nx/2); cj= int(Ny/2);
cr = (radius/a)*Nx;
I,J=np.meshgrid(np.arange(A.shape[0]),np.arange(A.shape[1]));

dist = np.sqrt((I-ci)**2 + (J-cj)**2);
A[np.where(dist<cr)] = e_r;

#visualize structure
plt.imshow(A);
plt.show()

## =============== Convolution Matrices ==============
E_r = cm.convmat2D(A, P,Q)
print(E_r.shape)
print(type(E_r))
plt.figure();
plt.imshow(abs(E_r), cmap = 'jet');
plt.colorbar()
plt.show()

## =============== K Matrices =========================
beta_x = beta_y = 0;
plt.figure();

## check K-matrices for normal icnidence
Kx, Ky = km.K_matrix_cubic_2D(0,0, a, a, P, Q);
np.set_printoptions(precision = 3)

print(Kx.todense())
print(Ky.todense())

band_cutoff = PQ; #number of bands to plot
## ======================== run band structure calc ==========================##
kx_scan = np.linspace(-np.pi, np.pi, 500)/a;
kx_mat = np.repeat(np.expand_dims(kx_scan, axis = 1), PQ,axis = 1)

eig_store = []
for beta_x in kx_scan:
    beta_y = beta_x;
    Kx, Ky = km.K_matrix_cubic_2D(beta_x, beta_y, a, a, P, Q);
    eigenvalues, eigenvectors, A_matrix = eg.PWEM2D_TE(Kx, Ky, E_r);
    #eigenvalues...match with the benchmark...but don't match with
    eig_store.append(np.sqrt(np.real(eigenvalues)));
    #plt.plot(beta_x*np.ones((PQ,)), np.sort(np.sqrt(eigenvalues)), '.')
eig_store = np.array(eig_store);

plt.plot(kx_mat[:,0:band_cutoff], eig_store[:,0:band_cutoff]/(2*np.pi),'.g');

    # question: which eigenvalues are relevant for plotting the band structure?

## solve the TM mode (TE in PC literature), should have smaller gap
eig_store = []
for beta_x in kx_scan:
    beta_y = beta_x;
    Kx, Ky = km.K_matrix_cubic_2D(beta_x, beta_y, a, a, P, Q);
    eigenvalues, eigenvectors, A_matrix = eg.PWEM2D_TM(Kx, Ky, E_r);
    #eigenvalues...match with the benchmark...but don't match with
    eig_store.append(np.sqrt(np.real(eigenvalues)));
    #plt.plot(beta_x*np.ones((PQ,)), np.sort(np.sqrt(eigenvalues)), '.')

eig_store = np.array(eig_store);

plt.plot(kx_mat, eig_store/(2*np.pi), '.b');
plt.plot(kx_scan, abs(kx_scan)/(2*np.pi));
plt.ylim([0,0.8])
plt.savefig('PWEM_photonic_crystal.png')
plt.show()

