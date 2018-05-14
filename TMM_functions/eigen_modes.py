'''
functions which analyzes the eigenmodes of a medium
'''

import numpy as np

def eigen_W(Gamma):
    '''
    for the E_field
    use: you would only really want to use this if the media is anisotropic in any way
    :param Gamma: 2x2 matrix for the scattering formalism
    :return:
    '''
    Lambda, W = np.linalg.eig(Gamma);  # LAMBDa is effectively refractive index
    lambda_matrix = np.diag(Lambda);

    return W, lambda_matrix

def eigen_V(Q, W, lambda_matrix):
    #V = Q*W*(lambda)^-1
    '''
    eigenmodes for the i*eta*H field
    :param Q: Q matrix
    :param W: modes from eigen W
    :param lambda_matrix: eigen values from W
    :return:
    '''
    return Q*W*np.linalg.inv(lambda_matrix);