import numpy as np
'''
not sure if this is strictly correct... yet
however, we know that the conv matrix in 1D must be of a Toeplitz SHAPE!
'''


def convmat1D(A, P):
    '''
    :param A: # 1D array of fourier components; right now, we have the input as the real space rep...
    :param P: # of order in X
    :return:
    '''
    N = A.shape;

    NH = P;
    p = list(range(-int(np.floor(P / 2)), int(np.floor(P / 2)) + 1));
    print(p)
    Af = (1 / np.prod(N)) * np.fft.fftshift(np.fft.fftn(A));
    # central indices;
    p0 = int(np.floor(N[0] / 2));
    q0 = int(np.floor(N[1] / 2));

    C = np.zeros((NH, NH))
    C = C.astype(complex);
    for prow in range(P):
        # first term locates z plane, 2nd locates y column, prow locates x
        for pcol in range(P):
            pfft = Af[prow] - Af[pcol];
            C[prow, pcol] = pfft;

    return C;

##testing
