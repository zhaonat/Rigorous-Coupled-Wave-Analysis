import numpy as np
from scipy import linalg as LA


def A(W1, W2, V1, V2):
    '''
    :param W1: gap E-field modes
    :param W2: layer E-modes
    :param V1: gap H-field modes
    :param V2: gap H-modes
    # the numbering is just 1 and 2 because the order differs if we're in the structure
    # or outsid eof it
    :return:
    '''
    A = np.linalg.inv(W1) * W2 + np.linalg.inv(V1) * V2;
    return A;

def B(W1, W2, V1, V2):
    B = np.linalg.inv(W1)*W2 - np.linalg.inv(V1)*V2;
    return B;


def S_layer(A,B, Li, k0, modes):
    '''
    function to create scatter matrix in the ith layer of the uniform layer structure
    :param A: function A =
    :param B: function B
    :param k0 #free -space wavevector
    :param Li #length of ith layer
    :return:
    '''
    X_i = LA.expm(modes * Li * k0);  # k and L are in Si Units
    term1 = (A - X_i * B * A.I * X_i * B).I
    S11 = term1 * (X_i * B * A.I * X_i * A - B);
    S12 = term1 * (X_i) * (A - B * A.I * B);
    S22 = S11;
    S21 = S12;
    S_dict = {'S11': S11, 'S22': S22,  'S12': S12,  'S21': S21};
    S = np.block([[S11, S12], [S21, S22]]);
    return S, S_dict;


def S_R(A,B):
    '''
    function to create scattering matrices in the reflection regions
    different from S_layer because these regions only have one boundary condition to satisfy
    :param A:
    :param B:
    :return:
    '''
    S11 = -np.linalg.inv(A)*B;
    S12 = 2*np.linalg.inv(A);
    S21 = 0.5*(A - B*np.linalg.inv(A)*B);
    S22 = B*np.linalg.inv(A)
    S_dict = {'S11': S11, 'S22': S22,  'S12': S12,  'S21': S21};
    S = np.block([[S11, S12], [S21, S22]]);
    return S, S_dict;