import cmath
import numpy as np
nums = [2+1j, 2-1j,-2+1j, -2-1j];

nums = [ 2.16932629+0.08864465j,  2.16932629-0.08864465j, -2.16932629+0.08864465j,
 -2.16932629-0.08864465j -0. ,       -1.95349574j , 0.        +1.95349574j,
 -0.        -0.74708443j , 0.        +0.74708443j,  0.09615035-0.j,
 -0.09615035+0.j        ];

nums = [-3.62219264+0.08541411j, -3.62219264-0.08541411j,  3.62219264+0.08541411j,
  3.62219264-0.08541411j, -0.        -1.37526785j,  0.        +1.37526785j,
 -1.57799386+0.j    ,     -1.43919972+0.j   ,       1.57799386-0.j,
  1.43919972-0.j        ]

def ccPair(nums):
    N = len(nums);
    d1 = [];
    d2 = [];
    mask = [0 for i in range(N)];
    for i in range(N):
        for j in range(i+1,N):
            if(nums[i] == np.conj(nums[j])):
                if(np.imag(nums[i])>0):
                    d1.append(nums[i]);
                    d2.append(nums[j]);
                    mask[i] =1;
                else:
                    d1.append(nums[j]);
                    d2.append(nums[i]);
                    mask[j] = 1;
            elif((np.imag(nums[i]) == 0 and nums[i] == -nums[j])):
                if(np.real(nums[i]) >0):
                    d1.append(nums[i]);
                    d2.append(nums[j])
                    mask[i] = 1;
                else:
                    d2.append(nums[i])
                    d1.append(nums[j])
                    mask[j] = 1;
    return d1, d2, mask;


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

print(nonHermitianEigenSorter(nums))