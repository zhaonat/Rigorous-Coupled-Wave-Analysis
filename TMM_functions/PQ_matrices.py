import numpy as np

## description
# kx, ky are normalized wavevectors (by k0)
# e_r and m_r do not have e0 or mu0 in them
##====================================##

def P(kx, ky, e_r, m_r):
        return (1/e_r)*np.matrix([[kx * ky, m_r * e_r - kx ** 2],
                                  [ky ** 2 - m_r * e_r, -kx * ky]]);


def Q(kx, ky, e_r, m_r):
        Q = (1/m_r)*np.matrix([[kx * ky, (m_r * e_r) - kx ** 2],
                               [ky ** 2 - (m_r * e_r), -kx * ky]]);
        return Q;

## simple test case;

