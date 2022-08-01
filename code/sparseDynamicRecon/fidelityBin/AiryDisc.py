'''
Created on 25 Oct 2021

@author: Rob Tovey

We approximate the Airy Disc function with a sum of cubic splines as 
described in:
    http://bigwww.epfl.ch/preprints/meijering0301p.pdf
    
The Taylor expansion of j1 is
j1(x) = x/2 - x^3/16 + x^5/384- x^7/18432 + x^9/1474560 + ... 

'''

import numpy as np
from .utils import jit, prange, __params, __pparams
from .Gaussian import GaussianFidelity
from scipy.special import j1
from numba import float64, int32


@jit(float64(float64, int32), inline='always', **__params)
def _cubic_eval(x, which):
    '''
    Cubic order kernel:
        which <= x < which+1
    Returns the spline evaluated at x:            
    '''
    if which == -2:
        return 2 + x * (4 + x * (2.5 + x * .5))
    elif which == -1:
        return 1 - x ** 2 * (2.5 + x * 1.5)
    elif which == 0:
        return 1 - x ** 2 * (2.5 - x * 1.5)
    elif which == 1:
        return 2 - x * (4 - x * (2.5 - x * .5))
    return 0


@jit([(float64, int32)], inline='always', **__params)
def _cubic_grad(x, which):
    '''
    Cubic order kernel:
        which <= x < which+1
    Returns the spline and gradient evaluated at x:
    '''
    if which == -2:
        return np.array((2. + x * (4 + x * (2.5 + x * .5)),
                         4. + x * (5 + x * 1.5)))
    elif which == -1:
        return np.array((1. - x ** 2 * (2.5 + x * 1.5),
                         x * (-5 - x * 4.5)))
    elif which == 0:
        return np.array((1. - x ** 2 * (2.5 - x * 1.5),
                         x * (-5 + x * 4.5)))
    elif which == 1:
        return np.array((2. - x * (4 - x * (2.5 - x * .5)),
                         -4. + x * (5 - x * 1.5)))
    return np.array((0., 0.))


@jit(float64(float64, int32), inline='always', **__params)
def _quartic_eval(x, which):
    '''
    Cubic order kernel:
        which <= x < which+1
    Returns the spline evaluated at x:
    '''
    if which == -3:
        return -1.5 - x * (7 / 4 + x * (2 / 3 + x * (1 / 12)))
    elif which == -2:
        return 2.5 + x * (59 / 12 + x * (3 + x * (7 / 12)))
    elif which == -1:
        return 1. - x ** 2 * (7 / 3 + x * (4 / 3))
    elif which == 0:
        return 1. + x ** 2 * (-7 / 3 + x * (4 / 3))
    elif which == 1:
        return 2.5 + x * (-59 / 12 + x * (3 - x * (7 / 12)))
    elif which == 2:
        return -1.5 + x * (7 / 4 + x * (-2 / 3 + x * (1 / 12)))
    else:
        return 0.


@jit((float64, int32), inline='always', **__params)
def _quartic_grad(x, which):
    '''
    Cubic order kernel:
        which <= x < which+1
    Returns the spline and gradient evaluated at x:
    '''
    if which == -3:
        return np.array((-1.5 - x * (7 / 4 + x * (2 / 3 + x * (1 / 12))),
                         -7 / 4 - x * (4 / 3 + x * .25)))
    elif which == -2:
        return np.array((2.5 + x * (59 / 12 + x * (3 + x * (7 / 12))),
                         59 / 12 + x * (6 + x * 1.75)))
    elif which == -1:
        return np.array((1. - x ** 2 * (7 / 3 + x * (4 / 3)),
                         x * (-14 / 3 - x * 4)))
    elif which == 0:
        return np.array((1. + x ** 2 * (-7 / 3 + x * (4 / 3)),
                         x * (-14 / 3 + x * 4)))
    elif which == 1:
        return np.array((2.5 + x * (-59 / 12 + x * (3 - x * (7 / 12))),
                         -59 / 12 + x * (6 - x * 1.75)))
    elif which == 2:
        return np.array((-1.5 + x * (7 / 4 + x * (-2 / 3 + x * (1 / 12))),
                         7 / 4 + x * (-4 / 3 + x * .25)))
    return np.array((0., 0.))


_p = 1, -1 / 8, 1 / 192, -1 / 9216, .9695 / 737280  # the last coefficient is dampened to have a better average error
r, R = 2, 10
_dx3 = .05, 1 / .05  # error is ~ 5e-7 = 0.004 * dx^3
_C3 = np.array([(2 * j1(x) / x) ** 2 if x >= .01
                else (_p[0] + _p[1] * x ** 2 + _p[2] * x ** 4 + _p[3] * x ** 6 + _p[4] ** x ** 8) ** 2
                for x in np.arange(r - 3 * _dx3[0], R + 4 * _dx3[0], _dx3[0])], dtype='float64')
_dx4 = .15, 1 / .15  # error is ~ 3e-7 = 0.0006 * dx^4
_C4 = np.array([(2 * j1(x) / x) ** 2 if x >= .01
                else (_p[0] + _p[1] * x ** 2 + _p[2] * x ** 4 + _p[3] * x ** 6 + _p[4] ** x ** 8) ** 2
                for x in np.arange(r - 3 * _dx4[0], R + 4 * _dx4[0], _dx4[0])], dtype='float64')
_C3[-2:] = _C4[-3:] = 0  # enforces continuity at cutoff


@jit(float64(float64), **__params)
def AiryDisc(x):
    if x > R ** 2:
        return 0
    elif x <= r ** 2:
        return (_p[0] + x * (_p[1] + x * (_p[2] + x * (_p[3] + x * _p[4])))) ** 2
    else:
        X = (x ** .5 - r) * _dx3[1] + 3  # conversion to coordinates of _C3
        n = int(X)  # rounded down
        X -= n
        v = 0
        for i in range(-2, 2):
            v += _C3[n - i] * _cubic_eval(X + i, i)
        return v


@jit(float64[:](float64), **__params)
def D_AiryDisc(x):
    if x > R ** 2:
        return np.array((0., 0.))
    elif x <= r ** 2:
        v = _p[0] + x * (_p[1] + x * (_p[2] + x * (_p[3] + x * _p[4])))
        dv = _p[1] + x * (2 * _p[2] + x * (3 * _p[3] + x * (4 * _p[4])))
        return np.array((v ** 2, 2 * v * dv))
    else:
        x = x ** .5
        X = (x - r) * _dx3[1] + 3  # conversion to coordinates of _C3
        n = int(X)  # rounded down
        X -= n
        v = np.zeros(2)
        for i in range(-2, 2):
            v += _C3[n - i] * _cubic_grad(X + i, i)
        v[1] *= _dx3[1] / (2 * x)
        return v


@jit(float64(float64), **__params)
def AiryDisc2(x):
    if x > R ** 2:
        return 0
    elif x <= r ** 2:
        return (_p[0] + x * (_p[1] + x * (_p[2] + x * (_p[3] + x * _p[4])))) ** 2
    else:
        X = (x ** .5 - r) * _dx4[1] + 3  # conversion to coordinates of _C3
        n = int(X)  # rounded down
        X -= n
        v = 0
        for i in range(-3, 3):
            v += _C4[n - i] * _quartic_eval(X + i, i)
    return v


@jit(float64[:](float64), **__params)
def D_AiryDisc2(x):
    if x > R ** 2:
        return np.array((0., 0.))
    elif x <= r ** 2:
        v = _p[0] + x * (_p[1] + x * (_p[2] + x * (_p[3] + x * _p[4])))
        dv = _p[1] + x * (2 * _p[2] + x * (3 * _p[3] + x * (4 * _p[4])))
        return np.array((v ** 2, 2 * v * dv))
    else:
        x = x ** .5
        X = (x - r) * _dx4[1] + 3  # conversion to coordinates of _C4
        n = int(X)  # rounded down
        X -= n
        v = np.zeros(2)
        for i in range(-3, 3):
            v += _C4[n - i] * _quartic_grad(X + i, i)
        v[1] *= _dx4[1] / (2 * x)
        return v


@jit(**__pparams)
def _Airy_fwrd(rho, loc, data, a, b, thresh, balanced, out):
    '''
    rho.shape = [I,3*J] or [I,1+2*J]
    loc.shape = [J,K,2]
    data.shape = [J,K]
    a,b,thresh = float
    out.shape = [J,K]

    out[j,k] = a*sum_i rho[i,j].a * AiryDisc(b*|rho[i,j].x - loc[j,k]|^2) - data[j,k]

    if |x|^2 > thresh, then we assume AiryDisc(b*|x|^2) = 0
    '''

    for j in prange(data.shape[0]):  # time
        J0 = 3 * j; J1 = 3 * j
        if balanced:
            J0 = 0; J1 = 2 * j
        for k in range(data.shape[1]):  # coordinate of data
            x = loc[j, k, 0]
            y = loc[j, k, 1]
            v = 0
            for i in range(rho.shape[0]):  # curve number
                r = (rho[i, 1 + J1] - x) ** 2 + (rho[i, 2 + J1] - y) ** 2
                if r < thresh:
                    v += rho[i, J0] * AiryDisc(b * r)

            out[j, k] = a * v - data[j, k]


@jit(**__pparams)
def _Airy_bwrd(X, res, loc, a, b, thresh, out):
    '''
    X.shape = [J,I,2]
    res.shape = [J,K]
    loc.shape = [J,K,2]
    a,b,thresh = float
    out.shape = [J,I]

    out[j,i] = a*sum_k res[j,k]*AiryDisc(b*|X[j,i] - loc[k]|^2)

    if |x|^2 > thresh, then we assume AiryDisc(b*|x|^2) = 0
    '''
    for j in prange(out.shape[0]):
        x = X[0] if X.shape[0] == 1 else X[j]
        for i in range(out.shape[1]):
            v = 0
            for k in range(res.shape[1]):  # coordinate of data
                r = (x[i, 0] - loc[j, k, 0]) ** 2 + (x[i, 1] - loc[j, k, 1]) ** 2
                if r < thresh:
                    v += res[j, k] * AiryDisc(b * r)

            out[j, i] = a * v


@jit(inline='always', **__params)
def __Airy_bwrd_diff(m, x, y, res, loc, a, b, thresh, out, J0, J1):
    v = 0; dvx = 0; dvy = 0
    for k in range(res.shape[0]):  # coordinate of data
        r = (x - loc[k, 0]) ** 2 + (y - loc[k, 1]) ** 2
        if r < thresh:
            g = res[k] * D_AiryDisc(b * r)
            v += g[0]
            g[1] *= 2 * m * b
            dvx += (x - loc[k, 0]) * g[1]
            dvy += (y - loc[k, 1]) * g[1]

    out[J0] += a * v
    out[1 + J1] = a * dvx
    out[2 + J1] = a * dvy


@jit(**__pparams)
def _Airy_bwrd_diff(rho, res, loc, a, b, thresh, balanced, out):
    '''
    rho.shape = [I,3*J] or [I,1+2*J]
    res.shape = [J,K]
    loc.shape = [J,K,2]
    a,b,thresh = float
    out.shape = rho.shape

    Let
        F(rho) = a*sum_{i,j} rho.a[i,j] * sum_k res[j,k] * AiryDisc(b*|rho.x[i,j] - loc[j,k]|^2)),
    then we return F and out[i,j] = dF/drho[i,j].
    '''

    F = 0
    if balanced:
        for i in prange(rho.shape[0]):
            m = rho[i, 0]
            for j in range(res.shape[0]):  # time
                J = 2 * j
                __Airy_bwrd_diff(m, rho[i, J + 1], rho[i, J + 2], res[j], loc[j], a, b, thresh, out[i], 0, J)

            F += m * out[i, 0]  # total energy
    else:
        for i in prange(rho.shape[0]):
            for j in range(res.shape[0]):  # time
                J = 3 * j
                m = rho[i, J]
                __Airy_bwrd_diff(m, rho[i, J + 1], rho[i, J + 2], res[j], loc[j], a, b, thresh, out[i], J, J)
                F += m * out[i, J]  # total energy

    return F


class AiryDiscFidelity(GaussianFidelity):
    '''
    This object represents the function
        F(rho) = .5*scale * sum_{i,j} (g\star rho[i](loc[j]) - data[i,j])^2
    where i is a time coordinate, j is space, and g is an Airy Disc
        g(x) = radius^-norm * AiryDisc(x^2/radius^2)


    Parameters
    ----------
    locations
        Array of shape [T, N, 2] or [N, 2] where N is the number of samples
        at each time-point. Assumed constant in time in the latter case.
    radius, norm, data
        See above
    scale=np.prod(locations.max()-locations.min())/data.size
        Scales the full function


    Attributes
    ----------
    data, locations, scale
        Can be updated to change the behaviour of the methods of this instance.
    radius, norm
        See above. Note that changing these attributes will not change the behaviour
        of this instance.


    See more in help(QuadraticFidelity)
    '''

    def __init__(self, locations, radius, norm, data, scale=None):
        def fwrd(rho, data, balanced, out):
            return _Airy_fwrd(rho, self.locations, data, *self._params, balanced, out)

        def bwrd(x, res, out):
            return _Airy_bwrd(x, res, self.locations, *self._params, out=out)

        def bwrd_diff(rho, res, balanced, out):
            return _Airy_bwrd_diff(rho, res, self.locations, *self._params, balanced, out)

        super().__init__(locations, radius, norm, data, scale=scale)

        self._op = [fwrd, bwrd, bwrd_diff]
        self._params = self._params[0], -self._params[1], radius ** 2 * R


if __name__ == '__main__':
    # Testing of gradient consistency
    for AD, D_AD in ((AiryDisc, D_AiryDisc), (AiryDisc2, D_AiryDisc2)):
        x = np.arange(0, R, 2 ** .5 * 1e-2)
        X = np.array([1] + [(2 * j1(xx) / xx) ** 2 for xx in x[1:]])
        Y = np.array([AD(xx ** 2) for xx in x])
        Z = np.array([D_AD(xx ** 2)[0] for xx in x])
        print('\n', ('%.2e    ' * 3) % (abs(X - Y).max(), abs(X - Y).mean(), abs(Y - Z).max()))

        for x, dx in ((5 ** .5, 1), (5 ** .5, -1), (2 ** .5, 1), (0., 1)):
            A = D_AD(x)
            print('\n', A[1])
            for i in range(10):
                print(('%+.3e    ' * 3) % (
                    D_AD(x + .1 ** i * dx)[0] - A[0], 10 ** i * (D_AD(x + .1 ** i * dx)[0] - A[0]),
                    100 ** i * (D_AD(x + .1 ** i * dx)[0] - A[0] - dx * .1 ** i * A[1])))
