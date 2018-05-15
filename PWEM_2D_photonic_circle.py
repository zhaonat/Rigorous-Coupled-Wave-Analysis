import numpy as np
import matplotlib.pyplot as plt
from convolution_matrices import convmat2D as cm
from PWEM_functions import K_matrix as km
from PWEM_functions import PWEM_eigen_problem as eg
'''
solve PWEM for a simple circular structure in a square unit cell
and generate band structure
compare with CEM EMLab
'''

## lattice and material parameters
a = 1;
radius = 0.35;
e_r = 9;

#generate irreducible BZ sample
T1 = 2*np.pi/a;
T2 = 2*np.pi/a;


# determine number of orders to use
P = 5; #don't play with odd orders
Q = 5;

# ============== build high resolution circle ==================
Nx = 512; Ny = 512;
A = e_r*np.ones((Nx,Ny));
ci = int(Nx/2); cj= int(Ny/2);
cr = (radius/a)*Nx;
I,J=np.meshgrid(np.arange(A.shape[0]),np.arange(A.shape[1]));

dist = np.sqrt((I-ci)**2 + (J-cj)**2);
A[np.where(dist<cr)] = 1;

#visualize structure
plt.imshow(A);
plt.show()

## =============== Convolution Matrices ==============
E_r = cm.convmat2D(A, P,Q)
print(type(E_r))
plt.figure();
plt.imshow(abs(E_r), cmap = 'jet');
plt.colorbar()
plt.show()


## =============== K Matrices =========================
beta_x = beta_y = 0;
plt.figure();
for beta_x in np.linspace(0, np.pi, 20):
    beta_y = 0;
    Kx, Ky = km.K_matrix_cubic_2D(beta_x, beta_y, a, a, P, Q);

    eigenvalues, eigenvectors, A_matrix = eg.PWEM2D_TE(Kx, Ky, E_r);
    #eigenvalues...match with the benchmark...but don't match with

    plt.plot(beta_x*np.ones((25,)), np.sort(eigenvalues), '.')
plt.show()
    # question: which eigenvalues are relevant for plotting the band structure?