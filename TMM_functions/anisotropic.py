''''
anisotropic TMM formulation
PQ formulation breaks down since the
differential equation
dz([ex, ey, hx, hy]) = Gamma ([ex, ey hx, hy])) has a Gamma matrix which is completely dense

'''

import numpy as np
import cmath;
from scipy.linalg import eig

def nonHermitianEigenSorter(eigenvalues):
    N = len(eigenvalues);
    sorted_indices=[];
    sorted_eigs = [];
    for i in range(N):
        eig = eigenvalues[i];
        if(np.real(eig)>0 and np.imag(eig) == 0):
            sorted_indices.append(i); sorted_eigs.append(eig);
        elif(np.real(eig)==0 and np.imag(eig) > 0):
            sorted_indices.append(i); sorted_eigs.append(eig);
        elif(np.real(eig)>0 and abs(np.imag(eig)) > 0):
            sorted_indices.append(i); sorted_eigs.append(eig);
    return sorted_eigs, sorted_indices;


def Gamma(kx, ky, e_tensor, m_r):
    '''
    :param kx:
    :param ky:
    :param e_tensor: is a 3x3 matrix containing dielectric components
    #exx exy exz  | e[0,0] e[1,0] e[2,0]
    #eyx eyy eyz  | e[1,0] e[1,1] e[1,2]
    #ezx ezy ezz  | e[2,0] e[2,1]
    :param m_r:
    :return:
    '''
    e = e_tensor; #for compactness of notation
    j = cmath.sqrt(-1);
    a11 = -1j*(ky*mu_tensor[1,2]/mu_tensor[2,2] + kx*(epsilon_tensor[2,0]/epsilon_tensor[2,2]))
    a12 = 1j*kx*(mu_tensor[1,2]/mu_tensor[2,2] - epsilon_tensor[2,1]/epsilon_tensor[2,2]);
    a13 = kx*ky/epsilon_tensor[2,2] + mu_tensor[1,0] - mu_tensor[1,2]*mu_tensor[2,0]/mu_tensor[2,2];
    a14 = -kx**2/epsilon_tensor[2,2] + mu_tensor[1,1]-  mu_tensor[1,2]*mu_tensor[2,1]/mu_tensor[2,2];

    a21 = 1j* ky *(mu_tensor[0,2]/mu_tensor[2,2] - epsilon_tensor[2,0]/epsilon_tensor[2,2]);
    a22 = -1j * kx*(mu_tensor[0,2]/mu_tensor[2,2]) +ky *(epsilon_tensor[2,1]/epsilon_tensor[2,2]);
    a23 = ky**2/epsilon_tensor[2,2] - mu_tensor[0,0] +  mu_tensor[0,2]*mu_tensor[2,0]/mu_tensor[2,2];
    a24 =  -kx*ky/epsilon_tensor[2,2] - mu_tensor[0,1] + mu_tensor[0,2]*mu_tensor[2,1]/mu_tensor[2,2];

    a31 = (kx*ky/mu_tensor[2,2] + epsilon_tensor[1,0] - epsilon_tensor[1,2]*epsilon_tensor[2,0]/epsilon_tensor[2,2])
    a32 = (-kx**2/mu_tensor[2,2] +epsilon_tensor[1,1] - epsilon_tensor[1,2]*epsilon_tensor[2,1]/epsilon_tensor[2,2]);
    a33 = -1j*(ky*(epsilon_tensor[1,2]/epsilon_tensor[2,2])+kx*(mu_tensor[2,0]/mu_tensor[2,2]));
    a34 = 1j*kx*(epsilon_tensor[1,2]/epsilon_tensor[2,2]-mu_tensor[2,1]/mu_tensor[2,2] )

    a41 = ky**2/mu_tensor[2,2] - epsilon_tensor[0,0] +  epsilon_tensor[0,2]*epsilon_tensor[2,0]/epsilon_tensor[2,2];
    a42 = -kx*ky/mu_tensor[2,2] - epsilon_tensor[0,1] + epsilon_tensor[0,2]*epsilon_tensor[2,1]/epsilon_tensor[2,2];
    a43 = 1j*ky*(epsilon_tensor[0,2]/epsilon_tensor[2,2]-mu_tensor[2,0]/mu_tensor[2,2] );
    a44 = -1j*(kx*(epsilon_tensor[0,2]/epsilon_tensor[2,2])+ky*(mu_tensor[2,1]/mu_tensor[2,2]));
    A = np.matrix([[a11, a12, a13, a14],
              [a21, a22, a23, a24],
              [a31, a32, a33, a34],
              [a41, a42, a43, a44]]);

    return A;

def eigen_Gamma(Gamma):
    ''' must execute some kind of sorting'''
    [U,V] = eig(Gamma);

    #execute a sorting algorithm on the eigenvalues
    return U,V