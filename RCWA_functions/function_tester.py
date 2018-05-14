import numpy as np

# incident wave
lam0 = 2;
theta = 0; phi = 0;
c0 = 3e8;
#extract kx_inc and ky_inc

#lattice constants of the layer (um)
a_x = 1;
a_y = 1;

#reflection and transmission regions (uniform)
e = [1,1];
m = [1,1];

#initial specifications required
M = 100; #M and N specifies the grid, we need a discrete grid because
N = 100;
m = list(range(-M, M+1));
n = list(range(-N, N+1));
P = M; Q = N; # do the orders relate to m and n? how?

e_r = np.ones((M,N));
u_r = 1;

# specify structure

#now define Kx, Ky, matrices (continuous in all layers)
for i in range(len(m)):
    Kx[i] = kx_inc - 2*np.pi*i/a_x;

for i in range(len(n)):
    Ky[i] = ky_inc - 2*np.pi*i/a_y;

#now calculate kz_r and kz_trans using the dispersion relation

## initialize device scattering matrices
S11 = np.zeros((2*P*Q, 2*P*Q));

## now we iterate through the layers like TMM using redheffer star