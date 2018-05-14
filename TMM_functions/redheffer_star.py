import numpy as np
import cmath

def RedhefferStar(SA,SB): #SA and SB are both 2x2 block matrices;

    assert type(SA) == np.matrixlib.defmatrix.matrix, 'not np.matrix'
    assert type(SB) == np.matrixlib.defmatrix.matrix, 'not np.matrix'


    I = np.matrix(np.eye(2,2));
    # once we break every thing like this, we should still have matrices
    SA_11 = SA[0:2,0:2]; SA_12 = SA[0:2,2:]; SA_21 = SA[2:, 0:2]; SA_22 = SA[2:,2:];
    SB_11 = SB[0:2,0:2]; SB_12 = SB[0:2,2:]; SB_21 = SB[2:, 0:2]; SB_22 = SB[2:,2:];

    assert type(SA_11) == np.matrixlib.defmatrix.matrix, 'not np.matrix'


    D = np.linalg.inv(I-SB_11*SA_22);
    F = np.linalg.inv(I-SA_22*SB_11);

    SAB_11 = SA_11 + SA_12*D*SB_11*SA_21;
    SAB_12 = SA_12*D*SB_12;
    SAB_21 = SB_21*F*SA_21;
    SAB_22 = SB_22 + SB_21*F*SA_22*SB_12;

    SAB = np.block([[SAB_11, SAB_12],[SAB_21, SAB_22]])
    return SAB, D, F;

#test case;
# homogeneous, identical layers
# Sg11 = np.matrix(np.zeros((2,2)));
# Sg12 = np.matrix(np.eye(2,2));
# Sg21 = np.matrix(np.eye(2,2));
# Sg22 = np.matrix(np.zeros((2,2))); #matrices
# SA = np.block([[Sg11, Sg12],[Sg21, Sg22]]); #initialization is equivelant as that for S_reflection side matrix
# SB = np.block([[Sg11, Sg12],[Sg21, Sg22]])
#
# I = np.matrix(np.eye(2, 2));
# # once we break every thing like this, we should still have matrices
# SA_11 = SA[0:2, 0:2];
# SA_12 = SA[0:2, 2:];
# SA_21 = SA[2:, 0:2];
# SA_22 = SA[2:, 2:];
# SB_11 = SB[0:2, 0:2];
# SB_12 = SB[0:2, 2:];
# SB_21 = SB[2:, 0:2];
# SB_22 = SB[2:, 2:];
#
#
# D = SA_12 * np.linalg.inv(I - SB_11 * SA_22);
# F = SB_21 * np.linalg.inv(I - SA_22 * SB_11);
#
# SAB_11 = SA_11 + D * SB_11 * SA_21;
# SAB_12 = D * SB_12;
# SAB_21 = F * SA_21;
# SAB_22 = SB_22 + F * SA_22 * SB_12;
#
# SAB = np.block([[SAB_11, SAB_12], [SAB_21, SAB_22]])
#
# print(np.linalg.norm(SAB-SA))