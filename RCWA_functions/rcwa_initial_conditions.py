import numpy as np

def delta_vector(P, Q):
    '''
        create a vector with a 1 corresponding to the 0th order
    '''
    vector = np.zeros((P*Q,));

    #the index of the (0,0) element requires a conversion using sub2ind
    index = int(P/2)*P+int(Q/2);
    vector[index] = 1
    return vector;


def initial_conditions(K_inc_vector, theta, normal_vector, pte, ptm, P, Q):
    '''
    :param K_inc_vector:
    :param theta: angle of incience
    :param ate_vector:
    :param normal_vector: pointing into z direction
    :param pte: te polarization amplitude
    :param ptm: tm polarization amplitude
    :return:
    calculates the incident E field, cinc, and the polarization fro the initial condition vectors
    '''
    if (theta != 0):
        ate_vector = np.cross(K_inc_vector, normal_vector);
        ate_vector = ate_vector / (np.linalg.norm(ate_vector));
    else:
        ate_vector = np.array([0,1,0]);

    atm_vector = np.cross(ate_vector, K_inc_vector);
    atm_vector = atm_vector / (np.linalg.norm(atm_vector))

    Polarization = pte * ate_vector + ptm * atm_vector; #total E_field incident which is a 3 component vector (ex, ey, ez)
    E_inc = Polarization;
    # go from mode coefficients to FIELDS
    Polarization = np.squeeze(np.array(Polarization));
    delta = delta_vector(P,Q);

    #cinc
    cinc = np.hstack((Polarization[0]*delta, Polarization[1]*delta));
    cinc = np.matrix(cinc).T; #mode amplitudes of Ex, and Ey

    return E_inc, cinc, Polarization
