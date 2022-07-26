# -*- coding: utf-8 -*-
import numpy as np

@jit(nopython=True, fastmath=True)
def _coeff_mat(x, deg):
    mat_ = np.zeros(shape=(x.shape[0],deg + 1))
    const = np.ones_like(x)
    mat_[:,0] = const
    mat_[:, 1] = x
    if deg > 1:
        for n in range(2, deg + 1):
            mat_[:, n] = x**n
    return mat_
    
@njit
def _fit_x(a, b):
    # linalg solves ax = b
    det_ = np.linalg.lstsq(a, b, rcond=-1)[0]
    #det_ = np.linalg.lstsq(a, b, rcond=None)[0]
    return det_ 

@njit
def fit_poly(x, y, deg):
    a = _coeff_mat(x, deg)
    p = _fit_x(a, y)
    # Reverse order so p[0] is coefficient of highest order
    return p[::-1]

@jit(nopython=True, fastmath=True)
def eval_polynomial(P, x, deg):
    '''
    Compute polynomial P(x) where P is a vector of coefficients, highest
    order coefficient at P[0].  Uses Horner's Method.
    '''
    result = np.zeros(len(x))
    deg += 1
    for i in range(deg):
        result = (x * result) + P[i]
    return result