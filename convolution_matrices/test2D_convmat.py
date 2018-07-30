import numpy as np
import matplotlib.pyplot as plt
from convolution_matrices.convmat2D import *
#generate a picture array with a circle
Nx = 2*256
Ny = 2*256;

A = 9*np.ones((Nx,Ny));
ci = Nx/2-1; cj= Ny/2-1;
cr = np.round(0.35*Nx);
I,J=np.meshgrid(np.arange(A.shape[0]),np.arange(A.shape[1]));

dist = np.sqrt((I-ci)**2 + (J-cj)**2);
A[np.where(dist<cr)] = 1;

plt.imshow(A);
plt.show()

##fft
P = 1; Q = 1;
C1 = convmat2D(A, P, Q);
C2 = convmat2D_o(A, 3,3);
print(np.linalg.norm(C1-C2)) #make sure the two ways of writing convmat are the same...
print(C1.shape)
np.set_printoptions(precision=2)
print(C1)
plt.imshow(np.abs(C1));
plt.show()


## test 2d convmat on a homogeneous medium, should return a scaled identity matrix
eps_grid = np.ones((Nx,Ny));
C_uniform = convmat2D(eps_grid,3,3)
print(C_uniform)