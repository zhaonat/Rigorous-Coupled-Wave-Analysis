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

def P_Q_kz(kx, ky, e_r, mu_r):
    '''
    r is for relative so do not put epsilon_0 or mu_0 here
    :param kx:
    :param ky:
    :param e_r:
    :param mu_r:
    :return:
    '''
    kz = np.sqrt(mu_r * e_r - kx ** 2 - ky ** 2);
    q = Q(kx, ky, e_r, mu_r)
    p = P(kx, ky, e_r, mu_r)

    return p, q, kz;
## simple test case;

def mode_module(kx, ky, e_r, m_r):
    '''
    module which outputs all the mode matrices needed in TMM
    :param kx:
    :param ky:
    :param e_r:
    :param m_r:
    :return:
    '''
    return None;
