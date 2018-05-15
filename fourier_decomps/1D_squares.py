import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy
import cmath
# do a fourier transform of a square function, which is analytically known
'''
since the fourier decompositions of some functions are analytic, particulalry in 1D,
we don't even need the FFT algorithm for the convolution matrix of epsilon in certain cases...
'''

n_order = 600;

#determine the b_n coefs

b_n_coefs = list();
for i in range(1,n_order):
    if(abs(i)%2 == 1):
        b_n_coefs.append(1*4/(i*np.pi));
    else:
        b_n_coefs.append(0);

#construct fourier series
x = np.linspace(-np.pi, np.pi, 1000);
L = np.pi;
f_eval = list();
for i in range(len(x)):
    f_x = 0;
    for j in range(1,n_order):
        f_x +=b_n_coefs[j-1]*np.sin(j*np.pi*x[i]/L);
    print(f_x)
    f_eval.append(f_x);

plt.plot(x, f_eval);
plt.title('fourier reconstruction of square box with orders = '+str(n_order))
#plt.plot(x, np.sin(x));
plt.show()


#using a DFT a discrete fourier transform; I think this is the same thing as a fourier series #
# discrete sample of the function of interest

#create a heaviside function.
heaviside_discrete = list();
for i in x:
    if(i > L/2 or i < -L/2):
        heaviside_discrete.append(0);
    else:
        heaviside_discrete.append(1);

dft_heaviside= np.fft.fft(heaviside_discrete); #these should be all the fourier components of the step
#to check, let's reconstruct the fourier series
max_N = int(len(x)/2);

fourier_rec = 0.5/(2*L); #0th order is just the average of the step;
for i in range(-max_N, max_N):
    fourier_rec += (1/(2*L))*dft_heaviside[i]*np.exp(-cmath.sqrt(-1)*2*np.pi*i*x/(2*L));
plt.figure()
plt.plot(np.fft.fftshift(dft_heaviside));
plt.show()
plt.figure()
plt.plot(x, heaviside_discrete);
plt.plot(x, fourier_rec); #it's close, but scaling seems off
plt.show()

