## same as the analytic case but with the fft
import numpy as np
import cmath;
from scipy import linalg as LA


def RCWA_1D_TE(E, lattice_constant, theta, num_ord, wavelength_scan,d):
    '''

    :param E:
    :param lattice_constant:
    :param theta:
    :param num_ord:
    :param wavelength_scan:
    :param d:  thickness
    :return:
    '''

    I = np.identity(2 * num_ord + 1)
    indices = np.arange(-num_ord, num_ord + 1);
    spectra_R = []; spectra_T = []; #arrays to hold reflectivity and transmissitivity

    # E is now the convolution of fourier amplitudes
    for wvlen in wavelength_scan:
        j = cmath.sqrt(-1);
        lam0 = wvlen;     k0 = 2 * np.pi / lam0; #free space wavelength in SI units

        ## =====================STRUCTURE======================##
        ## Region I: reflected region (half space)
        n1 = 1;#cmath.sqrt(-1)*1e-12; #apparently small complex perturbations are bad in Region 1, these shouldn't be necessary

        ## Region 2; transmitted region
        n2 = 1;

        #from the kx_components given the indices and wvln
        kx_array = k0*(n1*np.sin(theta) + indices*(lam0 / lattice_constant)); #0 is one of them, k0*lam0 = 2*pi
        k_xi = kx_array;
        ## IMPLEMENT SCALING: these are the fourier orders of the x-direction decomposition.
        KX = np.diag(kx_array/k0);
        KX2 = np.diag(np.power((k_xi/k0),2)); #singular since we have a n=0, m= 0 order and incidence is normal

        ## construct matrix of Gamma^2 ('constant' term in ODE):
        A = KX2 - E; #conditioning of this matrix is not bad, A SHOULD BE SYMMETRIC
        #sum of a symmetric matrix and a diagonal matrix should be symmetric;

        ##
        # when we calculate eigenvals, how do we know the eigenvals correspond to each particular fourier order?
        eigenvals, W = LA.eigh(A); #A should be symmetric or hermitian
        #we should be gauranteed that all eigenvals are REAL
        eigenvals = eigenvals.astype('complex');
        Q = np.diag(np.sqrt(eigenvals)); #Q should only be positive square root of eigenvals
        V = W@Q; #H modes
        #print((np.linalg.cond(W), np.linalg.cond(V))) #well conditioned
        ## this is the great typo which has killed us all this time
        X = np.diag(np.exp(-k0*np.diag(Q)*d)); #this is poorly conditioned because exponentiation
        ## pointwise exponentiation vs exponentiating a matrix

        ## observation: almost everything beyond this point is worse conditioned
        k_I = k0**2*(n1**2 - (k_xi/k0)**2);                 #k_z in reflected region k_I,zi
        k_II = k0**2*(n2**2 - (k_xi/k0)**2);   #k_z in transmitted region
        k_I = k_I.astype('complex'); k_I = np.sqrt(k_I);
        k_II = k_II.astype('complex'); k_II = np.sqrt(k_II);
        Y_I = np.diag(k_I/k0);
        Y_II = np.diag(k_II/k0);


        delta_i0 = np.zeros((len(kx_array),1));
        delta_i0[num_ord] = 1;
        n_delta_i0 = delta_i0*j*n1*np.cos(theta); #this is a VECTOR

        ## design auxiliary variables: SEE derivation in notebooks: RCWA_note.ipynb
        # we want to design the computation to avoid operating with X, particularly with inverses
        # since X is the worst conditioned thing

        #print((np.linalg.cond(W), np.linalg.cond(V)))
        # #Bo's solution
        # O = np.block([
        #     [W, W],
        #     [V,-V]
        # ]); #this is much better conditioned than S..
        #print(np.linalg.cond(O))

        Wi = np.linalg.inv(W);
        Vi = np.linalg.inv(V);

        #Oi = 0.5*np.block([[Wi, Vi],[Wi, -Vi]])
        # f = I;
        # g = j*Y_II; #all matrices
        # fg = np.concatenate((f,g),axis = 0)
        # #ab = np.matmul(np.linalg.inv(O),fg);
        # # ab = np.matmul(Oi, fg);
        # # a = ab[0:PQ,:];
        # # b = ab[PQ:,:];

        a = 0.5*(Wi+j*Vi@Y_II);
        b = 0.5*(Wi-j*Vi@Y_II);
        fbiX = np.matmul(np.linalg.inv(b),X)

        #altTerm = (a@X@X@b); #not well conditioned and I-altTermis is also poorly conditioned.
        #print(np.linalg.cond(I-np.linalg.inv(altTerm)))
        #print(np.linalg.cond(X@b)); #not well conditioned.

        term = X@a@fbiX; # THIS IS SHITTILY CONDITIONED
        # print((np.linalg.cond(X), np.linalg.cond(term)))
        # print(np.linalg.cond(I+term)); #but this is EXTREMELY WELL CONDITIONED.
        f = np.matmul(W, I+term);
        g = np.matmul(V,-I+term);
        T = np.linalg.inv(j*np.matmul(Y_I,f)+g);
        T = np.matmul(T,(np.matmul(j*Y_I,delta_i0)+n_delta_i0));
        R = np.matmul(f,T)-delta_i0;
        T = np.matmul(fbiX, T)

        ## calculate diffraction efficiencies
        #I would expect this number to be real...
        DE_ri = R*np.conj(R)*np.real(np.expand_dims(k_I,1))/(k0*n1*np.cos(theta));
        DE_ti = T*np.conj(T)*np.real(np.expand_dims(k_II,1))/(k0*n1*np.cos(theta));

        #print(np.sum(DE_ri))
        spectra_R.append(np.sum(DE_ri)); #spectra_T.append(T);
        spectra_T.append(np.sum(DE_ti))

    return spectra_R, spectra_T;





