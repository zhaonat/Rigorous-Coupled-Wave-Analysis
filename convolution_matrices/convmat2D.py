import numpy as np

def convmat2D(A, P, Q):
    '''
    :param A: input is currently whatever the real space representation of the structure is
    :param P: Pspecifies max order in x (so the sum is from -P to P
    :param Q: specifies max order in y (so the sum is from -Q to Q
    :return:
    '''
    N = A.shape;

    NH = (2*P+1) * (2*Q+1) ;
    p = list(range(-P, P + 1)); #array of size 2Q+1
    q = list(range(-Q, Q + 1));

    ## do fft
    Af = (1 / np.prod(N)) * np.fft.fftshift(np.fft.fft2(A));
    # natural question is to ask what does Af consist of..., what is the normalization for?

    ## NOTE: indexing error; N[0] actually corresponds to y and N[1] corresponds to x.

    # central indices marking the (0,0) order
    p0 = int((N[1] / 2)); #Af grid is Nx, Ny
    q0 = int((N[0] / 2)); #no +1 offset or anything needed because the array is orders from -P to P

    C = np.zeros((NH, NH))
    C = C.astype(complex);
    for qrow in range(2*Q+1): #remember indices in the arrary are only POSITIVE
        for prow in range(2*P+1): #outer sum
            # first term locates z plane, 2nd locates y column, prow locates x
            row = (qrow) * (2*P+1) + prow; #natural indexing
            for qcol in range(2*Q+1): #inner sum
                for pcol in range(2*P+1):
                    col = (qcol) * (2*P+1) + pcol; #natural indexing
                    pfft = p[prow] - p[pcol]; #get index in Af; #index may be negative.
                    qfft = q[qrow] - q[qcol];
                    C[row, col] = Af[q0 + pfft, p0 + qfft]; #index may be negative.

    return C;


def convmat2D_o(A, P, Q):
    '''
    :param A: input is currently whatever the real space representation of the structure is
    :param P: Pspecifies total number of orders
    :param Q:
    :return:
    '''
    N = A.shape;

    NH = P*Q ;
    p = list(range(-int(P/2), int(P/2) + 1));
    q = list(range(-int(Q/2), int(Q/2) + 1));

    ## do fft
    Af = (1 / np.prod(N)) * np.fft.fftshift(np.fft.fftn(A));
    # natural question is to ask what does Af consist of..., what is the normalization for?

    # central indices marking the (0,0) order
    p0 = int(np.floor(N[0] / 2)); #Af grid is Nx, Ny
    q0 = int(np.floor(N[1] / 2)); #we have to do minus 1 because indices are from 0 to N-1 for N element arrays

    C = np.zeros((NH, NH))
    C = C.astype(complex);
    for qrow in range(Q): #remember indices in the arrary are only POSITIVE
        for prow in range(P): #outer sum
            # first term locates z plane, 2nd locates y column, prow locates x
            row = (qrow) * (P) + prow; #natural indexing
            for qcol in range(Q): #inner sum
                for pcol in range(P):
                    col = (qcol) * (P) + pcol; #natural indexing
                    pfft = p[prow] - p[pcol];
                    qfft = q[qrow] - q[qcol];
                    C[row, col] = Af[p0 + pfft, q0 + qfft];

    return C;