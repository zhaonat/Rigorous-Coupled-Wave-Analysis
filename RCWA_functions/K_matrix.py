'''
functions which generate the K matrices along each direction
this is identical between PWEM and RCWA
'''
import numpy as np
from scipy import sparse

def K_matrix_cubic_2D(beta_x, beta_y, k0, a_x, a_y, N_p, N_q):
    #    K_i = beta_i - pT1i - q T2i - r*T3i
    # but here we apply it only for cubic and tegragonal geometries in 2D
    '''
    :param beta_i:
    :param T1:reciprocal lattice vector 1
    :param T2:
    :param T3:
    :return:
    '''
    #(indexing follows (1,1), (1,2), ..., (1,N), (2,1),
    # but in the cubic case, k_x only depends on p and k_y only depends on q
    k_x = beta_x - 2*np.pi*np.arange(-N_p, N_p+1)/(k0*a_x);
    k_y = beta_y - 2*np.pi*np.arange(-N_q, N_q+1)/(k0*a_y);

    kx, ky = np.meshgrid(k_x, k_y)
    # final matrix should be sparse...since it is diagonal at most
    Kx = sparse.diags(np.ndarray.flatten(kx))
    Ky = sparse.diags(np.ndarray.flatten(ky))

    return Kx, Ky

