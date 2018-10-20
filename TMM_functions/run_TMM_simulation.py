import numpy as np
import cmath
from TMM_functions import PQ_matrices as pq
from TMM_functions import scatter_matrices as sm
from TMM_functions import redheffer_star as rs
from TMM_functions import generate_initial_conditions as ic
from scipy import linalg as LA

def run_TMM_simulation(wavelengths, polarization_amplitudes, theta, phi, ER, UR, layer_thicknesses,\
                       transmission_medium, incident_medium):
    """
    :param wavelengths:
    :param theta:
    :param phi:
    :param ER: relative dielectric constants of each layer
    :param UR: relative permeability of each layer
    :param layer_thicknesses:
    :param transmission_medium: [et, mt]
    :param incident_medium:
    :return:
    """

    assert len(layer_thicknesses) == len(ER) == len(UR); "number of layer parameters not the same"
    ##
    #remove the kx and ky constraints on theta and phi
    ##

    ref = [];
    trans = [];
    I = np.matrix(np.eye(2, 2));  # unit 2x2 matrix

    [e_r, m_r] = incident_medium;
    [e_t, m_t] = transmission_medium;
    n_i = np.sqrt(e_r*m_r);
    kx = n_i * np.sin(theta) * np.cos(phi);  # constant in ALL LAYERS; kx = 0 for normal incidence
    ky = n_i * np.sin(theta) * np.sin(phi);  # constant in ALL LAYERS; ky = 0 for normal incidence
    normal_vector = np.array([0, 0, -1])  # positive z points down;
    ate_vector = np.matrix([0, 1, 0]);  # vector for the out of plane E-field

    ## =================  specify gap media ========================##
    e_h = 1;
    m_h = 1;
    Pg, Qg, kzg = pq.P_Q_kz(kx, ky, e_h, m_h)
    Wg = I;  # Wg should be the eigenmodes of the E field, which paparently is the identity, yes for a homogeneous medium
    sqrt_lambda = cmath.sqrt(-1) * Wg;
    # remember Vg is really Qg*(Omg)^-1; Vg is the eigenmodes of the H fields
    Vg = Qg * Wg * (sqrt_lambda) ** -1;

    ## ========================================== ##

    [pte, ptm] = polarization_amplitudes;

    for i in range(len(wavelengths)):  # in SI units
        ## initialize global scattering matrix: should be a 4x4 identity so when we start the redheffer star, we get I*SR

        Sg11 = np.matrix(np.zeros((2, 2)));
        Sg12 = np.matrix(np.eye(2, 2));
        Sg21 = np.matrix(np.eye(2, 2));
        Sg22 = np.matrix(np.zeros((2, 2)));  # matrices
        Sg = np.block(
            [[Sg11, Sg12], [Sg21, Sg22]]);  # initialization is equivelant as that for S_reflection side matrix

        ### ================= Working on the Reflection Side =========== ##
        Pr, Qr, kzr = pq.P_Q_kz(kx, ky, e_r, m_r)

        ## ============== values to keep track of =======================##
        S_matrices = list();
        kz_storage = [kzr];
        X_storage = list();
        ## ==============================================================##

        # define vacuum wavevector k0
        lam0 = wavelengths[i];  # k0 and lam0 are related by 2*pi/lam0 = k0
        k0 = 2*np.pi/lam0;
        ## modes of the layer
        Om_r = np.matrix(cmath.sqrt(-1) * kzr * I);
        X_storage.append(Om_r);
        W_ref = I;
        V_ref = Qr * Om_r.I;  # can't play games with V like with W because matrices for V are complex

        ## calculating A and B matrices for scattering matrix
        Ar, Br = sm.A_B_matrices(Wg, W_ref, Vg, V_ref);

        S_ref, Sr_dict = sm.S_R(Ar, Br);  # scatter matrix for the reflection region
        S_matrices.append(S_ref);
        Sg, D_r, F_r = rs.RedhefferStar(Sg, S_ref);

        ## go through the layers
        for i in range(len(ER)):
            # ith layer material parameters
            e = ER[i];
            m = UR[i];

            # longitudinal k_vector
            P, Q, kzl = pq.P_Q_kz(kx, ky, e, m)
            kz_storage.append(kzl)

            ## E-field modes that can propagate in the medium
            W_i = I;
            ## corresponding H-field modes.
            Om = cmath.sqrt(-1) * kzl * I;
            X_storage.append(Om)
            V_i = Q * np.linalg.inv(Om);

            # now defIne A and B
            A, B = sm.A_B_matrices(Wg, W_i, Vg, V_i);

            # calculate scattering matrix
            S_layer, Sl_dict = sm.S_layer(A, B, layer_thicknesses[i], k0, Om)
            S_matrices.append(S_layer);

            ## update global scattering matrix using redheffer star
            Sg, D_i, F_i = rs.RedhefferStar(Sg, S_layer);

        ##========= Working on the Transmission Side==============##
        Pt, Qt, kz_trans = pq.P_Q_kz(kx, ky, e_t, m_t);
        kz_storage.append(kz_trans);

        Om = cmath.sqrt(-1) * kz_trans * I;
        Vt = Qt * np.linalg.inv(Om);

        # get At, Bt
        At, Bt = sm.A_B_matrices(Wg, I, Vg, Vt)

        ST, ST_dict = sm.S_T(At, Bt)
        S_matrices.append(ST);
        # update global scattering matrix
        Sg, D_t, F_t = rs.RedhefferStar(Sg, ST);

        K_inc_vector = n_i * k0 * np.matrix([np.sin(theta) * np.cos(phi), \
                                             np.sin(theta) * np.sin(phi), np.cos(theta)]);

        # cinc is the c1+
        E_inc, cinc, Polarization = ic.initial_conditions(K_inc_vector, theta, normal_vector, pte, ptm)

        ## COMPUTE FIELDS
        Er = Sg[0:2, 0:2] * cinc;  # S11; #(cinc = initial mode amplitudes), cout = Sg*cinc; #2d because Ex, Ey...
        Et = Sg[2:, 0:2] * cinc;  # S21

        Er = np.squeeze(np.asarray(Er));
        Et = np.squeeze(np.asarray(Et));

        Erx = Er[0];
        Ery = Er[1];
        Etx = Et[0];
        Ety = Et[1];

        # apply the grad(E) = 0 equation to get z components
        Erz = -(kx * Erx + ky * Ery) / kzr;
        Etz = -(kx * Etx + ky * Ety) / kz_trans;  ## using divergence of E equation here

        # add in the Erz component to vectors
        Er = np.matrix([Erx, Ery, Erz]);  # a vector
        Et = np.matrix([Etx, Ety, Etz]);

        R = np.linalg.norm(Er) ** 2;
        T = np.linalg.norm(Et) ** 2;
        ref.append(R);
        trans.append(T);

    return ref, trans

def run_TMM_dispersive(wavelengths, polarization_amplitudes, theta, phi, ER, UR, layer_thicknesses,\
                       transmission_medium, incident_medium):
    """
    ER and UR are matrices
    :param wavelengths:
    :param theta:
    :param phi:
    :param ER: relative dielectric constants of each layer
    :param UR: relative permeability of each layer
    :param layer_thicknesses:
    :param transmission_medium:
    :param incident_medium:
    :return:
    """
    assert ER.shape == UR.shape

    num_layers = len(layer_thicknesses);
    ref = [];
    trans = [];
    I = np.matrix(np.eye(2, 2));  # unit 2x2 matrix

    [e_r, m_r] = incident_medium;
    [e_t, m_t] = transmission_medium;
    n_i = np.sqrt(e_r*m_r);
    kx = n_i * np.sin(theta) * np.cos(phi);  # constant in ALL LAYERS; kx = 0 for normal incidence
    ky = n_i * np.sin(theta) * np.sin(phi);  # constant in ALL LAYERS; ky = 0 for normal incidence
    normal_vector = np.array([0, 0, -1])  # positive z points down;
    ate_vector = np.matrix([0, 1, 0]);  # vector for the out of plane E-field

    ## =================  specify gap media ========================##
    e_h = 1;
    m_h = 1;
    Pg, Qg, kzg = pq.P_Q_kz(kx, ky, e_h, m_h)
    Wg = I;  # Wg should be the eigenmodes of the E field, which paparently is the identity, yes for a homogeneous medium
    sqrt_lambda = cmath.sqrt(-1) * Wg;
    # remember Vg is really Qg*(Omg)^-1; Vg is the eigenmodes of the H fields
    Vg = Qg * Wg * (sqrt_lambda) ** -1;

    ## ========================================== ##

    [pte, ptm] = polarization_amplitudes;

    for i in range(len(wavelengths)):  # in SI units
        ## initialize global scattering matrix: should be a 4x4 identity so when we start the redheffer star, we get I*SR

        Sg11 = np.matrix(np.zeros((2, 2)));
        Sg12 = np.matrix(np.eye(2, 2));
        Sg21 = np.matrix(np.eye(2, 2));
        Sg22 = np.matrix(np.zeros((2, 2)));  # matrices
        Sg = np.block(
            [[Sg11, Sg12], [Sg21, Sg22]]);  # initialization is equivelant as that for S_reflection side matrix

        ### ================= Working on the Reflection Side =========== ##
        Pr, Qr, kzr = pq.P_Q_kz(kx, ky, e_r, m_r)

        ## ============== values to keep track of =======================##
        S_matrices = list();
        kz_storage = [kzr];
        X_storage = list();
        ## ==============================================================##

        # define vacuum wavevector k0
        lam0 = wavelengths[i];  # k0 and lam0 are related by 2*pi/lam0 = k0
        k0 = 2*np.pi/lam0;
        ## modes of the layer
        Om_r = np.matrix(cmath.sqrt(-1) * kzr * I);
        X_storage.append(Om_r);
        W_ref = I;
        V_ref = Qr * Om_r.I;  # can't play games with V like with W because matrices for V are complex

        ## calculating A and B matrices for scattering matrix
        Ar, Br = sm.A_B_matrices(Wg, W_ref, Vg, V_ref);

        S_ref, Sr_dict = sm.S_R(Ar, Br);  # scatter matrix for the reflection region
        S_matrices.append(S_ref);
        Sg, D_r, F_r = rs.RedhefferStar(Sg, S_ref);

        ## go through the layers
        for j in range(num_layers):
            # ith layer material parameters
            e = ER[j,i];
            m = UR[j,i];

            # longitudinal k_vector
            P, Q, kzl = pq.P_Q_kz(kx, ky, e, m)
            kz_storage.append(kzl)

            ## E-field modes that can propagate in the medium
            W_i = I;
            ## corresponding H-field modes.
            Om = cmath.sqrt(-1) * kzl * I;
            X_storage.append(Om)
            V_i = Q * np.linalg.inv(Om);

            # now defIne A and B
            A, B = sm.A_B_matrices(Wg, W_i, Vg, V_i);

            # calculate scattering matrix
            S_layer, Sl_dict = sm.S_layer(A, B, layer_thicknesses[j], k0, Om)
            S_matrices.append(S_layer);

            ## update global scattering matrix using redheffer star
            Sg, D_i, F_i = rs.RedhefferStar(Sg, S_layer);

        ##========= Working on the Transmission Side==============##
        Pt, Qt, kz_trans = pq.P_Q_kz(kx, ky, e_t, m_t);
        kz_storage.append(kz_trans);

        Om = cmath.sqrt(-1) * kz_trans * I;
        Vt = Qt * np.linalg.inv(Om);

        # get At, Bt
        At, Bt = sm.A_B_matrices(Wg, I, Vg, Vt)

        ST, ST_dict = sm.S_T(At, Bt)
        S_matrices.append(ST);
        # update global scattering matrix
        Sg, D_t, F_t = rs.RedhefferStar(Sg, ST);

        K_inc_vector = n_i * k0 * np.matrix([np.sin(theta) * np.cos(phi), \
                                             np.sin(theta) * np.sin(phi), np.cos(theta)]);

        # cinc is the c1+
        E_inc, cinc, Polarization = ic.initial_conditions(K_inc_vector, theta, normal_vector, pte, ptm)

        ## COMPUTE FIELDS
        Er = Sg[0:2, 0:2] * cinc;  # S11; #(cinc = initial mode amplitudes), cout = Sg*cinc; #2d because Ex, Ey...
        Et = Sg[2:, 0:2] * cinc;  # S21

        Er = np.squeeze(np.asarray(Er));
        Et = np.squeeze(np.asarray(Et));

        Erx = Er[0];
        Ery = Er[1];
        Etx = Et[0];
        Ety = Et[1];

        # apply the grad(E) = 0 equation to get z components
        Erz = -(kx * Erx + ky * Ery) / kzr;
        Etz = -(kx * Etx + ky * Ety) / kz_trans;  ## using divergence of E equation here

        # add in the Erz component to vectors
        Er = np.matrix([Erx, Ery, Erz]);  # a vector
        Et = np.matrix([Etx, Ety, Etz]);

        R = np.linalg.norm(Er) ** 2;
        T = np.linalg.norm(Et) ** 2;
        ref.append(R);
        trans.append(T);

    return ref, trans

