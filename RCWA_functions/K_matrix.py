'''
functions which generate the K matrices along each direction
this is identical between PWEM and RCWA
'''
import numpy as np
from scipy import sparse
from numpy.linalg import cond

def K_matrix_cubic_2D(beta_x, beta_y, k0, a_x, a_y, N_p, N_q):
    #    K_i = beta_i - pT1i - q T2i - r*T3i
    # but here we apply it only for cubic and tegragonal geometries in 2D
    '''
    :param beta_x: input k_x,inc/k0
    :param beta_y: k_y,inc/k0; #already normalized...k0 is needed to normalize the 2*pi*lambda/a
            however such normalization can cause singular matrices in the homogeneous module (specifically with eigenvalues)
    :param T1:reciprocal lattice vector 1
    :param T2:
    :param T3:
    :return:
    '''
    #(indexing follows (1,1), (1,2), ..., (1,N), (2,1),(2,2),(2,3)...(M,N) ROW MAJOR
    # but in the cubic case, k_x only depends on p and k_y only depends on q
    k_x = beta_x - 2*np.pi*np.arange(-N_p, N_p+1)/(k0*a_x);
    k_y = beta_y - 2*np.pi*np.arange(-N_q, N_q+1)/(k0*a_y);

    kx, ky = np.meshgrid(k_x, k_y); #this is the N_p x N_q grid
    # final matrix should be sparse...since it is diagonal at most
    #order in flatten actually doesn't matter in the end
    Kx = np.diag(kx.flatten(order = 'C')); #default is C or column major
    Ky = np.diag(ky.flatten(order = 'C'))

    return Kx, Ky

