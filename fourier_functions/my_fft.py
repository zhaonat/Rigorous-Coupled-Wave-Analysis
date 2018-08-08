## my implementation of fast fourier transform algorithm

import numpy as np
import cmath

def dft(data):
    '''
    data is a 1D discrete set of numbers (can be complex)
    essential idea is a dft maps one nx1 vector to another nx1 vector, both in the complex plane
    basic algorithm as is is O(n^2), which is sub-optimal
    :param data:
    :return:

    '''
    i = cmath.sqrt(-1);
    num_ord = len(data);
    xk = [];
    for n in range(num_ord): #iterate through all values of data
        sum = 0;
        for k in range(num_ord): #for each data point, iterate through all fourier orders
            sum+=data[n]*np.exp(-2*np.pi*i*n*k/num_ord)
        xk.append(sum);
    return xk;


def fft():
    '''
    
    :return:
    '''
    return None;