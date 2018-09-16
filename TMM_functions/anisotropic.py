''''
anisotropic TMM formulation
PQ formulation breaks down since the
differential equation
dz([ex, ey, hx, hy]) = Gamma ([ex, ey hx, hy])) has a Gamma matrix which is completely dense

'''

import numpy as np
import cmath;
from scipy.linalg import eig

def Gamma(kx, ky, e_tensor, m_r):
    '''
    :param kx:
    :param ky:
    :param e_tensor: is a 3x3 matrix containing dielectric components
    #exx exy exz  | e[0,0] e[1,0] e[2,0]
    #eyx eyy eyz  | e[1,0] e[1,1] e[1,2]
    #ezx ezy ezz  | e[2,0] e[2,1]
    :param m_r:
    :return:
    '''
    e = e_tensor; #for compactness of notation
    j = cmath.sqrt(-1);
    Gamma = np.matrix([[-j*(kx*(e[2,0]/e[2,2]) + ky/m_r)     ,  j*kx*(1/m_r-e[2,1]/e[2,2])           , kx*ky/e[2,2]        , -kx**2/e[2,2]+m_r    ],
                       [j*ky*(-e[2,0]/e[2,2])                , -j*(ky*(e[2,1]/e[2,2]))               , ky**2/e[2,2] - (m_r), -kx*ky/e[2,2]        ],
                       [kx*ky/m_r+e[1,0]-e[1,2]*e[2,0]/e[2,2], -kx**2/m_r+e[1,1]-e[1,2]*e[2,1]/e[2,2],-j*(ky*e[1,2]/e[2,2]),  j*kx*(e[1,2]/e[2,2])],
                       [ky**2/m_r-e[0,0]+e[0,2]*e[2,0]/e[2,2], -kx*ky/m_r-e[0,1]+e[0,2]*e[2,1]/e[2,2], j*ky*(e[0,2]/e[2,2]), -j*(kx*e[0,2]/e[2,2])]]);

    #if we deal with this matrix, then the traditional TMM approach is unstable...
    return Gamma;

def eigen_Gamma(Gamma):
    [U,V] = eig(Gamma);

    #execute a sorting algorithm on the eigenvalues
    return U,V