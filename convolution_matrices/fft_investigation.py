## investigating how ffts work in numpy
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt

eps_grid = np.ones((20,20));
eps_grid[5:15, 5:15] = 12;

test1 = fft.fftshift(fft.fft2(eps_grid)); #this fortunately agrees with matlab...
plt.figure();
plt.imshow(abs(test1))
plt.colorbar()
plt.show()

Nx = 512; Ny = 512;
e_r = 6;
a =1; radius = 0.35;
A = e_r*np.ones((Nx,Ny));
ci = int(Nx/2); cj= int(Ny/2);
cr = (radius/a)*Nx;
I,J=np.meshgrid(np.arange(A.shape[0]),np.arange(A.shape[1]));

dist = np.sqrt((I-ci)**2 + (J-cj)**2);
A[np.where(dist<cr)] = 1;

Afc = np.fft.fftshift(np.fft.fft2(A));
plt.figure();
plt.imshow(np.log(abs(Afc)))
plt.show()


