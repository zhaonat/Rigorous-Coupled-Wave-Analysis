import numpy as np
import matplotlib.pyplot as plt;
import cmath;
from TMM_functions import PQ_matrices as pq
from TMM_functions import scatter_matrices as sm
from TMM_functions import redheffer_star as rs
from TMM_functions import generate_initial_conditions as ic
from scipy import linalg as LA

#%% DEFINE SIMULATION PARAMETERS
# TMM CAN FAIL IF KZ = 0 IN ANY MEDIA!!!!
#% General Units
degrees = np.pi/180;
L0 = 1e-6; #units of microns;
eps0 = 8.854e-12;
mu0 = 4*np.pi-7;
c0 = 1/(np.sqrt(mu0*eps0))
I = np.matrix(np.eye(2,2)); #unit 2x2 matrix

## REFLECTION AND TRANSMSSION SPACE PARAMETERS
m_r = 1; e_r = 1;
m_t = 1; e_t = 1;

## set wavelength scanning range
wavelengths = L0*np.linspace(0.1, 0.2,300)
k0_layers = 2*np.pi/wavelengths;
#source parameters
theta = 0 * degrees; #%elevation angle
phi = 0 * degrees; #%azimuthal angle

## incident wave properties, at this point, everything is in units of k_0
n_i = np.sqrt(e_r*m_r);
kx = n_i*np.sin(theta)*np.cos(phi); #constant in ALL LAYERS
ky = n_i*np.sin(theta)*np.sin(phi); #constant in ALL LAYERS
kz_inc = cmath.sqrt(e_r * m_r - kx ** 2 - ky ** 2);
normal_vector = np.array([0, 0, -1]) #positive z points down;
ate_vector = np.matrix([0, 1, 0]); #vector for the out of plane E-field
K_inc_vector = n_i * np.array([np.sin(theta) * np.cos(phi), \
                               np.sin(theta) * np.sin(phi), np.cos(theta)]);

#ampltidue of the te vs tm modes (which are decoupled)
pte = 1/np.sqrt(2);
ptm = cmath.sqrt(-1)/np.sqrt(2);

E_inc, cinc, Polarization = ic.initial_conditions(K_inc_vector, theta, ate_vector, normal_vector, pte, ptm)

print('--------incident wave paramters----------------')
print(str(cinc)+', '+str(np.linalg.norm(cinc)));
print('incident n_i: '+str(n_i))
print('kx_inc: '+str(kx)+' ky_inc: '+str(ky))
print('kz_inc: ' + str(kz_inc));
print('-----------------------------------------------')

## initialize global scattering matrix
Sg11 = np.matrix(np.zeros((2,2)));
Sg12 = np.matrix(np.eye(2,2));
Sg21 = np.matrix(np.eye(2,2));
Sg22 = np.matrix(np.zeros((2,2)));
Sg = np.block([[Sg11, Sg12],[Sg21, Sg22]]);

## Homogeneous gap layer parameters (which have thickness 0);
#gap media, which can be anything!, so we take the CEM usage
e_h = 1+kx**2+ky**2; m_h = 1;
# CEM lab chooses e_h = 1+kx^2 + ky^2, which is not one for off-normal
kzg = m_h*e_h - kx**2 - ky**2; #k vector in gap medium
Qg = (1/m_h)*np.matrix([[kx * ky,  (e_h*m_h - kx ** 2)], [ky ** 2 - e_h*m_h, -kx * ky]])
Wg = I;
Omg = cmath.sqrt(-1)*kzg*I;
# remember Vg is really Qg*(Omg)^-1;
Vg = Qg*Omg.I;

#thickness 0 means L = 0, which only pops up in the xponential part of the expression
ER = [12];
UR = [1];
L =  L0*0.1*np.array([1]); #this retains SI unit convention

ref = list(); trans = list();
for i in range(len(wavelengths)): #in SI units
    # define vacuum wavevector k0
    k0 = k0_layers[i];
    lam0 = wavelengths[i];
    ### ================= Working on the Reflection Side =========== ##
    kzr = np.sqrt(m_r * e_r - kx ** 2 - ky ** 2);
    Pr = pq.P(kx, ky, m_r, e_r);
    Qr = pq.Q(kx, ky, e_r, m_r);
    Om_r = np.matrix(cmath.sqrt(-1) * kzr * I);
    W_ref = I;
    V_ref = Qr * Om_r.I;

    ## SOMEHOW WE GOT THIS WRONG? usually Ar and Br is interchanged
    Br = I - Vg.I * V_ref;
    Ar = I + Vg.I * V_ref;

    S_ref, Sr_dict = sm.S_R(Ar, Br); #special matrices for the trans and ref regions

    Sg, D_r, F_r = rs.RedhefferStar(Sg, S_ref);
    Q_storage = list(); P_storage=  list();
    kz_storage = list();
    X_storage = list();

    ## go through the layers
    for i in range(len(ER)):
        #ith layer material parameters
        e = ER[i]; m = UR[i];
        #longitudinal k_vector
        kzl= np.sqrt(e*m - kx**2 - ky**2);
        kz_storage.append(kzl)
        ## Q and P matrices are from rearranging maxwell's equations
        Q = pq.Q(kx, ky, e, m); Q_storage.append(Q);
        P = pq.P(kx, ky, e, m); P_storage.append(P);

        ## E-field modes that can propagate in the medium
        W_i = I;
        Om = cmath.sqrt(-1) * kzl * I;
        X_i = LA.expm(Om * L[i] * k0); #k0 and L are in Si Units
        ## corresponding H-field modes.
        V_i = Q * np.linalg.inv(Om);
        #now defIne A and B
        B = I - np.linalg.inv(V_i) * Vg; #simplification for LHI layers
        A = I + np.linalg.inv(V_i) * Vg;
        #calculate scattering matrix

        S_layer, Sl_dict = sm.S_layer(A, B, L[i], k0, Om)
        ## update global scattering matrix
        Sg, D_i, F_i = rs.RedhefferStar(Sg, S_layer);

    ##========= Working on the Transmission Side==============##
    kz_trans = np.sqrt(e_t * m_t - kx ** 2 - ky ** 2);

    Pt = pq.P(kx, ky, e_t, m_t);
    Qt = pq.Q(kx, ky, e_t, m_t);
    Om = cmath.sqrt(-1) * kz_trans * I;
    Vt = Qt*np.linalg.inv(Om);
    Bt = I - np.linalg.inv(Vg) * Vt;
    At = I + np.linalg.inv(Vg) * Vt;

    St11 = Bt*np.linalg.inv(At);
    St12 = 0.5*I*(At - Bt*np.linalg.inv(At)*Bt);
    St21 = 2*np.linalg.inv(At);
    St22 = -np.linalg.inv(At)*Bt
    ST = np.block([[St11, St12],[St21, St22]]);

    Sg, D_t, F_t = rs.RedhefferStar(Sg, ST);

    ## COMPUTE FIELDS
    Er = (W_ref*Sg[0:2,0:2]*cinc); #S11
    Et = (I*Sg[2:,0:2]*cinc);      #S21
    Er = np.squeeze(np.asarray(Er));
    Et = np.squeeze(np.asarray(Et));

    Erx = Er[0]; Ery = Er[1];
    Etx = Et[0]; Ety = Et[1];

    Erz = -(kx*Erx+ky*Ery)/kzr;
    Etz = -(kx*Etx+ky*Ety)/kz_trans; ## using divergence of E equation here
    Er = np.matrix([Erx, Ery, Erz]);
    Et = np.matrix([Etx, Ety, Etz]);

    R = np.linalg.norm(Er)**2;
    T = np.linalg.norm(Et)**2;
    ref.append(R);
    trans.append(T);


## BASIC PLOTTING
ref = np.array(ref);
trans = np.array(trans)
plt.figure();
plt.plot(wavelengths, ref);
plt.plot(wavelengths, trans);
plt.plot(wavelengths, ref+trans);
plt.plot(wavelengths, 1-ref-trans);

plt.show()
