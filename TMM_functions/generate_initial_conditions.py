import numpy as np

def initial_conditions(K_inc_vector, theta, ate_vector, normal_vector, pte, ptm):
    if (theta != 0):
        ate_vector = np.cross(K_inc_vector, normal_vector);
        ate_vector = ate_vector / (np.linalg.norm(ate_vector));
    atm_vector = np.cross(ate_vector, K_inc_vector);
    atm_vector = atm_vector / (np.linalg.norm(atm_vector))

    Polarization = pte * ate_vector + ptm * atm_vector;
    E_inc = Polarization;
    # go from mode coefficients to FIELDS
    Polarization = np.squeeze(np.array(Polarization));
    cinc = np.array([Polarization[0], Polarization[1]]);
    cinc = np.matrix(cinc).T;

    return E_inc, cinc, Polarization
