import numpy as np
from TMM_functions.anisotropic import *

e_tensor = np.array([[1,1,0],[1,1,0],[0,0,1]]);
m_r = 1;
kx = 0;
ky = 0;

gamma = Gamma(kx, ky, e_tensor, m_r)

U,V = eigen_Gamma(gamma);

