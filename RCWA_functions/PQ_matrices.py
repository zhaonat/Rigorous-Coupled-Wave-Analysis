import numpy as np

## description
# kx, ky are normalized wavevector MATRICES NOW (by k0)
# matrices because we have expanded in some number of spatial harmonics
# e_r and m_r do not have e0 or mu0 in them
# presently we assume m_r is homogeneous 1
##====================================##

def P(Kx, Ky, e_conv):
        return np.matrix([[Kx * Ky,  e_conv - Kx * Kx],
                                         [Ky ** 2 - e_conv, -Kx * Ky]]);


def Q(Kx, Ky, e_conv):
        Q = np.matrix([[Kx * e_conv.I * Ky, (1) - Kx * e_conv.I * Kx],
                               [Ky *e_conv.I *Ky - (1), -Kx * e_conv.I * Ky]]);
        return Q;

## simple test case;

