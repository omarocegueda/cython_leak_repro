#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
import numpy as np
cimport cython
cimport numpy as cnp

def leak(int[:] s):
    cdef:
        int i,j,k,l
        double[:,:,:,:] block = np.zeros(shape=(s[0], s[1], s[2], s[3]), dtype=np.float64)
    with nogil:
        for i in range(s[0]):
            for j in range(s[1]):
                for k in range(s[2]):
                    for l in range(s[3]):
                        block[i,j,k,l] = i+j+k+l
    return block

def noleak(int[:] s, double[:,:,:,:] out):
    cdef:
        int i,j,k,l
    with nogil:
        for i in range(s[0]):
            for j in range(s[1]):
                for k in range(s[2]):
                    for l in range(s[3]):
                        out[i,j,k,l] = i+j+k+l
