import numpy as np
import matplotlib.pyplot as plt
## preliminary tests
#inputs: A, P, Q, R
# A is the discrete representation of epsilon

#number of spatial harmonics (or orders)
P = 6;
Q = 6;
R = 6;
Nx = 20; Ny = 20; Nz = 1; #this is fundamentally 3D...not sure how to make general for 2D

N = np.array([Nx, Ny, Nz]);

## generalize two 2D geometries;

A = np.ones(N+1)
A[2:18, 2:18, 0] = 12;
plt.imshow(A[:,:,0]);
plt.show()
# deal with different dimensionalities
if(len(N) == 1):
    Q = 1; R = 1;
elif(len(N) == 2):
    R = 1;

NH = P*Q*R;
p = list(range(-int(np.floor(P/2)), int(np.floor(P/2))+1));
print(p)
q = list(range(-int(np.floor(Q/2)), int(np.floor(Q/2))+1));
r = list(range(-int(np.floor(R/2)), int(np.floor(R/2))+1));

Af = (1/np.prod(N))*np.fft.fftshift(np.fft.fftn(A));

#central indices;
p0 = int(np.floor(Nx/2));
q0 = int(np.floor(Ny/2));
r0 = int(np.floor(Nz/2));

C = np.zeros((NH, NH))
C = C.astype(complex);
for rrow in range(R):
    for qrow in range(Q):
        for prow in range(P):
            #first term locates z plane, 2nd locates y column, prow locates x
            row = (rrow)*Q*P+(qrow)*P + prow;
            for rcol in range(R):
                for qcol in range(Q):
                    for pcol in range(P):
                        col = (rcol)*Q*P + (qcol)*P + pcol;
                        pfft = p[prow] - p[pcol];
                        qfft = q[qrow] - q[qcol];
                        rfft = r[rrow] - r[rrow]
                        C[row, col] = Af[p0+pfft, q0+qfft, r0+rfft];


plt.imshow(np.abs(Af[:, :, 0]));
plt.show()
plt.imshow(np.abs(C));
plt.show()
plt.plot(np.diag(abs(C)))
plt.show()