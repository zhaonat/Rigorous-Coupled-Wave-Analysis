import numpy as np
from scipy.linalg import block_diag
import cmath;

def homogeneous_module(Kx, Ky, e_r, m_r = 1):
    '''
    homogeneous layer is much simpler to do, so we will create an isolated module to deal with it
    :return:
    '''
    i = cmath.sqrt(-1);
    N = len(Kx);
    I = np.matrix(np.identity(N));
    P = e_r**-1*np.block([[Kx*Ky, e_r*m_r*I-Kx**2], [Ky**2-m_r*e_r*I, -Ky*Kx]])
    Q = (e_r/m_r)*P;
    W = np.matrix(np.identity(2*N))
    Kz_mag = np.sqrt(abs(m_r*e_r*I-Kx**2-Ky**2));
    eigenvalues = block_diag(i*Kz_mag, i*Kz_mag)
    V = Q*np.linalg.inv(eigenvalues)

    return W,V,Kz_mag