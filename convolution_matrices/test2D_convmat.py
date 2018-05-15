import numpy as np
import matplotlib.pyplot as plt
from convolution_matrices.convmat2D import convmat2D
#generate a picture array with a circle
Nx = 100
Ny = 100;

A = np.ones((Nx,Ny));
ci = 50; cj= 50;
cr = 40;
I,J=np.meshgrid(np.arange(A.shape[0]),np.arange(A.shape[1]));

dist = np.sqrt((I-ci)**2 + (J-cj)**2);
A[np.where(dist<cr)] = 12;

plt.imshow(A);
plt.show()

##fft
P = 10; Q = 10;
C = convmat2D(A, P, Q);
plt.imshow(np.abs(C));
plt.show()