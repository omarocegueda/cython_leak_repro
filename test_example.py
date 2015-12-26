import numpy as np
import psutil
from example import leak, noleak

def test_leak():
    shape = np.array([200,200,200,3], dtype=np.int32)
    niter = 30
    for i in range(niter):
        b = leak(shape)
        bcopy = b[...,1].copy()
        del bcopy
        del b
        print('Free memory: %d %%'%(psutil.virtual_memory()[2],))


def test_noleak():
    shape = np.array([200,200,200,3], dtype=np.int32)
    niter = 30
    for i in range(niter):
        b = np.empty(shape=tuple(shape), dtype=np.float64)
        noleak(shape, b)
        bcopy = b[...,1].copy()
        del bcopy
        del b
        print("Free memory: %d %%"%(psutil.virtual_memory()[2],))