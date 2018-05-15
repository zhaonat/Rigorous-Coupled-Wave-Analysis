'''
functions which generate the K matrices along each direction
this is identical between PWEM and RCWA
'''
import numpy as np
from scipy import sparse

def K_matrix_cubic_2D(beta_x, beta_y, a_x, a_y, N_p, N_q):
    #    K_i = beta_i - pT1i - q T2i - r*T3i
    # but here we apply it only for cubic and tegragonal geometries in 2D
    '''
    :param beta_i:
    :param T1:reciprocal lattice vector 1
    :param T2:
    :param T3:
    :return:
    '''
    k_x = beta_x - 2*np.pi*np.arange(-int(N_p/2), int(N_p/2)+1)/a_x;
    k_y = beta_y - 2*np.pi*np.arange(-int(N_q/2), int(N_q/2)+1)/a_y;

    kx, ky = np.meshgrid(k_x, k_y)
    # final matrix should be sparse...since it is diagonal at most
    Kx = sparse.diags(np.ndarray.flatten(kx))
    Ky = sparse.diags(np.ndarray.flatten(ky))

    return Kx, Ky
