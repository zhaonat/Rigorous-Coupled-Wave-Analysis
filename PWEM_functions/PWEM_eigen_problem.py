
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
    :param E_r:
    :return:
    '''

    A = Kx**2 + Ky**2
    B = E_r;
    V = sparse.csr_matrix.dot(np.linalg.inv(B),A);
    eigenvalues, eigenvectors = np.linalg.eig(V);
    #get eigenvalues of this
    return eigenvalues, eigenvectors;
