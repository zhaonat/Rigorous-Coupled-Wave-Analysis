import numpy as np

def convmat2D(A, P, Q):
    N = A.shape;

    NH = P * Q ;
    p = list(range(-int(np.floor(P / 2)), int(np.floor(P / 2)) + 1));
    print(p)
    q = list(range(-int(np.floor(Q / 2)), int(np.floor(Q / 2)) + 1));

    Af = (1 / np.prod(N)) * np.fft.fftshift(np.fft.fftn(A));

    # central indices;
    p0 = int(np.floor(N[0] / 2));
    q0 = int(np.floor(N[1] / 2));

    C = np.zeros((NH, NH))
    C = C.astype(complex);
    for qrow in range(Q):
        for prow in range(P):
            # first term locates z plane, 2nd locates y column, prow locates x
            row = (qrow) * P + prow;
            for qcol in range(Q):
                for pcol in range(P):
                    col = (qcol) * P + pcol;
                    pfft = p[prow] - p[pcol];
                    qfft = q[qrow] - q[qcol];
                    C[row, col] = Af[p0 + pfft, q0 + qfft];

    return C;