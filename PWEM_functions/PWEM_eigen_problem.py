
import numpy as np
from scipy import sparse
from scipy import linalg
'''
function to construct the matrix operators for the eigenproblem
'''

def PWEM2D_TE(Kx, Ky, E_r):
    '''
    currently, we assume that mu_r is always homogeneous
    :param Kx:
    :param Ky:
    :param E_r:
    :return:
    '''

    A = Kx.todense()**2 + Ky.todense()**2; #can do this because Kx and Ky are diagonal
    #A = A.todense(); #this can be bad, but for now...
    B = E_r;
    eigenvalues, eigenvectors = linalg.eig(A,B);
    #get eigenvalues of this
    return eigenvalues, eigenvectors, A;


def PWEM2D_TM(Kx, Ky, E_r):
    '''
    currently, we assume that mu_r is always homogeneous
    :param Kx:
    :param Ky:
    :param E_r: fourier decomp conv matrix of eps_r
    :return:
    '''

    #A = Kx.todense() ** 2 + Ky.todense() ** 2
    Er_inv = np.linalg.inv(E_r);
    A = Kx@Er_inv@Kx +Ky@Er_inv@Ky;
    eigenvalues, eigenvectors = np.linalg.eig(A);
    #get eigenvalues of this
    return eigenvalues, eigenvectors,A;
