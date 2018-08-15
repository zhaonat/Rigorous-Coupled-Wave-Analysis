import numpy as np
from scipy import linalg as LA


## NOTE: * operator does NOT PERFORM MATRIX MULTIPLICATION IN PYTHON UNLESS the matrices are np.matrix objects

def A(W_layer, Wg, V_layer, Vg): # PLUS SIGN
    '''
    OFFICIAL EMLAB prescription
    inv(W_layer)*W_gap
    :param W_layer: layer E-modes
    :param Wg: gap E-field modes
    :param V_layer: layer H_modes
    :param Vg: gap H-field modes
    # the numbering is just 1 and 2 because the order differs if we're in the structure
    # or outsid eof it
    :return:
    '''
    assert type(W_layer) == np.matrixlib.defmatrix.matrix, 'not np.matrix'
    assert type(Wg) == np.matrixlib.defmatrix.matrix, 'not np.matrix'
    assert type(V_layer) == np.matrixlib.defmatrix.matrix, 'not np.matrix'
    assert type(Vg) == np.matrixlib.defmatrix.matrix, 'not np.matrix'

    A = np.linalg.inv(W_layer) * Wg + np.linalg.inv(V_layer) * Vg;
    return A;

def B(W_layer, Wg, V_layer, Vg): #MINUS SIGN
    '''
    :param W_layer: layer E-modes
    :param Wg: gap E-field modes
    :param V_layer: layer H_modes
    :param Vg: gap H-field modes
    # the numbering is just 1 and 2 because the order differs if we're in the structure
    # or outsid eof it
    :return:
    '''
    assert type(W_layer) == np.matrixlib.defmatrix.matrix, 'not np.matrix'
    assert type(Wg) == np.matrixlib.defmatrix.matrix, 'not np.matrix'
    assert type(V_layer) == np.matrixlib.defmatrix.matrix, 'not np.matrix'
    assert type(Vg) == np.matrixlib.defmatrix.matrix, 'not np.matrix'

    B = np.linalg.inv(W_layer) * Wg - np.linalg.inv(V_layer) * Vg;
    return B;


def A_B_matrices_half_space(W_layer, Wg, V_layer, Vg):
    ## this function is needed because for the half-spaces (reflection and transmission
    # spaces, the convention for calculating A and B is DIFFERENT than inside the main layers
    a = A(Wg, W_layer, Vg, V_layer);
    b = B(Wg, W_layer, Vg, V_layer);
    return a, b;


def A_B_matrices(W_layer, Wg, V_layer, Vg):
    '''
    single function to output the a and b matrices needed for the scatter matrices
    :param W_layer: gap
    :param Wg:
    :param V_layer: gap
    :param Vg:
    :return:
    '''
    a = A(W_layer, Wg, V_layer, Vg);
    b = B(W_layer, Wg, V_layer, Vg);
    return a, b;

def S_layer(A,B, Li, k0, modes):
    '''
    function to create scatter matrix in the ith layer of the uniform layer structure
    we assume that gap layers are used so we need only one A and one B
    :param A: function A =
    :param B: function B
    :param k0 #free -space wavevector magnitude (normalization constant) in Si Units
    :param Li #length of ith layer (in Si units)
    :return: S (4x4 scatter matrix) and Sdict, which contains the 2x2 block matrix as a dictionary
    '''
    assert type(A) == np.matrixlib.defmatrix.matrix, 'not np.matrix'
    assert type(B) == np.matrixlib.defmatrix.matrix, 'not np.matrix'

    #sign convention (EMLAB is exp(-1i*k\dot r))
    X_i = LA.expm(-modes * Li * k0);  # k and L are in Si Units
    #X_i could be a problem in RCWA

    term1 = (A - X_i * B * A.I * X_i * B).I
    S11 = term1 * (X_i * B * A.I * X_i * A - B);
    S12 = term1 * (X_i) * (A - B * A.I * B);
    S22 = S11;
    S21 = S12;
    S_dict = {'S11': S11, 'S22': S22,  'S12': S12,  'S21': S21};
    S = np.block([[S11, S12], [S21, S22]]);
    return S, S_dict;


def S_R(Ar, Br):
    '''
    function to create scattering matrices in the reflection regions
    different from S_layer because these regions only have one boundary condition to satisfy
    :param Ar:
    :param Br:
    :return:
    '''
    assert type(Ar) == np.matrixlib.defmatrix.matrix, 'not np.matrix'
    assert type(Br) == np.matrixlib.defmatrix.matrix, 'not np.matrix'


    S11 = -np.linalg.inv(Ar) * Br;
    S12 = 2*np.linalg.inv(Ar);
    S21 = 0.5*(Ar - Br * np.linalg.inv(Ar) * Br);
    S22 = Br * np.linalg.inv(Ar)
    S_dict = {'S11': S11, 'S22': S22,  'S12': S12,  'S21': S21};
    S = np.block([[S11, S12], [S21, S22]]);
    return S, S_dict;

def S_T(At, Bt):
    '''
    function to create scattering matrices in the transmission regions
    different from S_layer because these regions only have one boundary condition to satisfy
    :param At:
    :param Bt:
    :return:
    '''
    assert type(At) == np.matrixlib.defmatrix.matrix, 'not np.matrix'
    assert type(Bt) == np.matrixlib.defmatrix.matrix, 'not np.matrix'

    S11 = (Bt) * np.linalg.inv(At);
    S21 = 2*np.linalg.inv(At);
    S12 = 0.5*(At - Bt * np.linalg.inv(At) * Bt);
    S22 = - np.linalg.inv(At)*Bt
    S_dict = {'S11': S11, 'S22': S22,  'S12': S12,  'S21': S21};
    S = np.block([[S11, S12], [S21, S22]]);
    return S, S_dict;

