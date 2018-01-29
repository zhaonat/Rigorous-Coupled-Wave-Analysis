import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

Nx = 100; Ny = 100;
eps_r = np.ones((Nx, Ny));

eps_r[40:60, 20:80] = 12; #only a 1D grating has a true toeplitz structure
#eps_r[45:55, 45:55] = 10;
print(eps_r.shape)
plt.imshow(eps_r);
plt.show()

## execute convolution using fourier convolve
fft = np.fft.fft2(eps_r)
plt.plot(np.fft.fftshift(np.abs(fft)));
plt.show()

test= (signal.convolve2d(fft, fft));
test2 = signal.fftconvolve(eps_r, eps_r)
plt.subplot(121);
plt.imshow(np.abs(test));
plt.subplot(122);
plt.imshow(np.abs(test2));
plt.show()

## 1D trial
eps_r = np.ones((100,1));
eps_r[20:80] = 12;
test3 = signal.fftconvolve(eps_r, eps_r);
plt.plot(np.abs(test3));
plt.show()

