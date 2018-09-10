import numpy as np
import matplotlib.pyplot as plt
from RCWA_functions import K_matrix as km
from RCWA_functions import PQ_matrices as pq
from TMM_functions import eigen_modes as em
from TMM_functions import scatter_matrices as sm
from RCWA_functions import redheffer_star as rs
from RCWA_functions import rcwa_initial_conditions as ic
from RCWA_functions import homogeneous_layer as hl
import cmath


def run_RCWA_2D(lam0, theta, phi, ER, UR, layer_thicknesses, lattice_constants, pte, ptm, N,M, e_half):
    '''
    :param lam0:
    :param theta:
    :param phi:
    :param ER:
    :param UR:
    :param layer_thicknesses:
    :param lattice_constants:
    :param pte:
    :param ptm:
    :param N:
    :param M:
    :param e_half: [e_r e_t], dielectric constants of the reflection and transmission spaces
    :return:
    '''
    ## convention specifications
    normal_vector = np.array([0, 0, -1])  # positive z points down;
    ate_vector = np.array([0, 1, 0]);  # vector for the out of plane E-field
    ## ===========================

    Lx = lattice_constants[0];
    Ly = lattice_constants[1];
    NM = (2 * N + 1) * (2 * M + 1);

    # define vacuum wavevector k0
    k0 = 2*np.pi/lam0;
    ## ============== values to keep track of =======================##
    S_matrices = list();
    kz_storage = list();
    X_storage = list();
    ## ==============================================================##

    m_r = 1; e_r = e_half[0];
    ## incident wave properties, at this point, everything is in units of k_0
    n_i = np.sqrt(e_r * m_r);

    # actually, in the definitions here, kx = k0*sin(theta)*cos(phi), so kx, ky here are normalized
    kx_inc = n_i * np.sin(theta) * np.cos(phi);
    ky_inc = n_i * np.sin(theta) * np.sin(phi);  # constant in ALL LAYERS; ky = 0 for normal incidence
    kz_inc = cmath.sqrt(e_r * 1 - kx_inc ** 2 - ky_inc ** 2);

    # remember, these Kx and Ky come out already normalized
    Kx, Ky = km.K_matrix_cubic_2D(kx_inc, ky_inc, k0, Lx,Ly, N, M);  # Kx and Ky are diagonal but have a 0 on it

    ## =============== K Matrices for gap medium =========================
    ## specify gap media (this is an LHI so no eigenvalue problem should be solved
    e_h = 1;
    Wg, Vg, Kzg = hl.homogeneous_module(Kx, Ky, e_h)

    ### ================= Working on the Reflection Side =========== ##
    Wr, Vr, kzr = hl.homogeneous_module(Kx, Ky, e_r);
    kz_storage.append(kzr)

    ## calculating A and B matrices for scattering matrix
    # since gap medium and reflection media are the same, this doesn't affect anything
    Ar, Br = sm.A_B_matrices(Wg, Wr, Vg, Vr);

    ## s_ref is a matrix, Sr_dict is a dictionary
    S_ref, Sr_dict = sm.S_R(Ar, Br);  # scatter matrix for the reflection region
    S_matrices.append(S_ref);
    Sg = Sr_dict;

    ## go through the layers
    for i in range(len(ER)):
        # ith layer material parameters
        e_conv = ER[i];
        mu_conv = UR[i];

        # longitudinal k_vector
        P, Q, kzl = pq.P_Q_kz(Kx, Ky, e_conv, mu_conv)
        kz_storage.append(kzl)
        Gamma_squared = P @ Q;

        ## E-field modes that can propagate in the medium, these are well-conditioned
        W_i, lambda_matrix = em.eigen_W(Gamma_squared);
        V_i = em.eigen_V(Q, W_i, lambda_matrix);

        # now defIne A and B, slightly worse conditoined than W and V
        A, B = sm.A_B_matrices(W_i, Wg, V_i, Vg);  # ORDER HERE MATTERS A LOT because W_i is not diagonal

        # calculate scattering matrix
        Li = layer_thicknesses[i];
        S_layer, Sl_dict = sm.S_layer(A, B, Li, k0, lambda_matrix)
        S_matrices.append(S_layer);

        ## update global scattering matrix using redheffer star
        Sg_matrix, Sg = rs.RedhefferStar(Sg, Sl_dict);

    ##========= Working on the Transmission Side==============##
    m_t = 1;
    e_t = e_half[1];
    Wt, Vt, kz_trans = hl.homogeneous_module(Kx, Ky, e_t)

    # get At, Bt
    # since transmission is the same as gap, order does not matter
    At, Bt = sm.A_B_matrices(Wg, Wt, Vg, Vt)

    ST, ST_dict = sm.S_T(At, Bt)
    S_matrices.append(ST);
    # update global scattering matrix
    Sg_matrix, Sg = rs.RedhefferStar(Sg, ST_dict);

    ## finally CONVERT THE GLOBAL SCATTERING MATRIX BACK TO A MATRIX

    K_inc_vector = n_i * np.array([np.sin(theta) * np.cos(phi), \
                                    np.sin(theta) * np.sin(phi), np.cos(theta)]);

    E_inc, cinc, Polarization = ic.initial_conditions(K_inc_vector, theta, normal_vector, pte, ptm, N, M)
    # print(cinc.shape)
    # print(cinc)

    cinc = np.linalg.inv(Wr) @ cinc;
    ## COMPUTE FIELDS: similar idea but more complex for RCWA since you have individual modes each contributing
    reflected = Wr @ Sg['S11'] @ cinc;
    transmitted = Wt @ Sg['S21'] @ cinc;

    rx = reflected[0:NM, :];  # rx is the Ex component.
    ry = reflected[NM:, :];  #
    tx = transmitted[0:NM, :];
    ty = transmitted[NM:, :];

    # longitudinal components; should be 0
    rz = np.linalg.inv(kzr) @ (Kx @ rx + Ky @ ry);
    tz = np.linalg.inv(kz_trans) @ (Kx @ tx + Ky @ ty)

    ## we need to do some reshaping at some point

    ## apparently we're not done...now we need to compute 'diffraction efficiency'
    r_sq = np.square(np.abs(rx)) + np.square(np.abs(ry)) + np.square(np.abs(rz));
    t_sq = np.square(np.abs(tx)) + np.square(np.abs(ty)) + np.square(np.abs(tz));
    R = np.real(kzr) * r_sq / np.real(kz_inc);
    T = np.real(kz_trans) * t_sq / (np.real(kz_inc));

    return np.sum(R), np.sum(T);


## need a simulation which can return the field profiles inside the structure