import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy
# do a fourier transform of a square function, which is analytically known


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
plt.plot(np.fft.fftshift(b_n_coefs))
plt.show()

#using a DFT;
# discrete sample of the function of interest

heaviside_discrete = list();
for i in x:
    if(i > L or i < -L):
        heaviside_discrete.append(0);
    else:
        heaviside_discrete.append(1);

dft_heaviside= np.fft.fft(heaviside_discrete);
plt.plot(np.fft.fftshift(dft_heaviside));
plt.show()
plt.plot(x, heaviside_discrete);
plt.show()

# we need the toeplitz matrix
dft_mat = scipy.linalg.dft(1000);
a = dft_mat*heaviside_discrete;
