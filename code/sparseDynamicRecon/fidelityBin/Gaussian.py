'''
Created on 6 Dec 2021

@author: Rob
'''
import numpy as np
from .utils import QuadraticFidelity, jit, prange, __params, __pparams, FLOAT, ZERO


@jit(**__pparams)
def _gaussian_fwrd(rho, loc, data, a, b, thresh, balanced, out):
    '''
    rho.shape = [I,3*J] or [I,1+2*J] 
    loc.shape = [J,K,2]
    data.shape = [J,K]
    a,b,thresh = float
    out.shape = [J,K]

    out[j,k] = a*(sum_i rho[i,j].a * np.exp(b*|rho[i,j].x - loc[j,k]|^2)) - data[j,k]

    if |x|^2 > thresh, then we assume np.exp(b*|x|^2) = 0   
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
                    v += rho[i, J0] * np.exp(b * r)

            out[j, k] = a * v - data[j, k]


@jit(**__pparams)
def _gaussian_bwrd(X, res, loc, a, b, thresh, out):
    '''
    X.shape = [J,I,2]
    res.shape = [J,K]
    loc.shape = [J,K,2]
    a,b,thresh = float
    out.shape = [J,I]

    out[j,i] = a*(sum_k res[j,k]*np.exp(b*|X[j,i] - loc[k]|^2))

    if |x|^2 > thresh, then we assume np.exp(b*|x|^2) = 0   
    '''
    for j in prange(out.shape[0]):
        x = X[0] if X.shape[0] == 1 else X[j]
        for i in range(out.shape[1]):
            v = 0
            for k in range(res.shape[1]):  # coordinate of data
                r = (x[i, 0] - loc[j, k, 0]) ** 2 + (x[i, 1] - loc[j, k, 1]) ** 2
                if r < thresh:
                    v += res[j, k] * np.exp(b * r)

            out[j, i] = a * v


@jit(inline='always', **__params)
def __gaussian_bwrd_diff(m, x, y, res, loc, a, b, thresh, out, J0, J1):
    v = 0; dvx = 0; dvy = 0
    for k in range(res.shape[0]):  # coordinate of data
        r = (x - loc[k, 0]) ** 2 + (y - loc[k, 1]) ** 2
        if r < thresh:
            g = res[k] * np.exp(b * r)
            v += g
            g *= 2 * m * b
            dvx += (x - loc[k, 0]) * g
            dvy += (y - loc[k, 1]) * g

    out[J0] += a * v
    out[1 + J1] = a * dvx
    out[2 + J1] = a * dvy


@jit(**__pparams)
def _gaussian_bwrd_diff(rho, res, loc, a, b, thresh, balanced, out):
    '''
    rho.shape = [I,3*J] or [I,1+2*J] 
    res.shape = [J,K]
    loc.shape = [J,K,2]
    a,b,thresh = float
    out.shape = rho.shape

    Let
        F(rho) = a * sum_{i,j} rho.a[i,j] * sum_k res[j,k] * np.exp(b*|rho.x[i,j] - loc[j,k]|^2)),
    then we return F and out[i,j] = dF/drho[i,j].
    '''

    F = 0
    if balanced:
        for i in prange(rho.shape[0]):
            m = rho[i, 0]
            for j in range(res.shape[0]):  # time
                J = 2 * j
                __gaussian_bwrd_diff(m, rho[i, J + 1], rho[i, J + 2], res[j], loc[j], a, b, thresh, out[i], 0, J)

            F += m * out[i, 0]  # total energy
    else:
        for i in prange(rho.shape[0]):
            for j in range(res.shape[0]):  # time
                J = 3 * j
                m = rho[i, J]
                __gaussian_bwrd_diff(m, rho[i, J + 1], rho[i, J + 2], res[j], loc[j], a, b, thresh, out[i], J, J)
                F += m * out[i, J]  # total energy

    return F


class GaussianFidelity(QuadraticFidelity):
    '''
    This object represents the function
        F(rho) = sum_{i,j} .5*(g\star rho[i](loc[j]) - data[i,j])^2
    where i is a time coordinate, j is space, and g is a gaussian
        g(x) = radius^-scale * exp(-x^2/radius^2)


    Parameters
    ----------
    locations
        Array of shape [T, N, 2] or [N, 2] where N is the number of samples
        at each time-point. Assumed constant in time in the latter case.
    radius, norm, data
        See above
    scale=np.prod(locations.max()-locations.min())/data.size
        Scales the full function
    ZERO=utils.ZERO (= 1e-10)
        Thresholds the value of the Gaussian at this level


    Attributes
    ----------
    data, locations, scale
        Can be updated to change the behaviour of the methods of this instance.
    radius, norm, ZERO
        See above. Note that changing these attributes will not change the behaviour
        of this instance.


    See more in help(QuadraticFidelity)
    '''

    def __init__(self, locations, radius, norm, data, ZERO=ZERO):
        def fwrd(rho, data, balanced, out):
            return _gaussian_fwrd(rho, self.locations, data, *self._params, balanced, out)

        def bwrd(x, res, out):
            return _gaussian_bwrd(x, res, self.locations, *self._params, out=out)

        def bwrd_diff(rho, res, balanced, out):
            return _gaussian_bwrd_diff(rho, res, self.locations, *self._params, balanced, out)

        if scale is None:
            scale = locations[..., 0].ptp() * locations[..., 1].ptp() / data.size

        super().__init__(fwrd, bwrd, bwrd_diff, FLOAT, data, scale=scale)

        locations = locations.reshape(-1, self.data.shape[1], 2)
        assert locations.shape[0] in (1, self.T)
        self.locations = np.tile(locations, (self.T, 1, 1)) if locations.shape[0] == 1 else locations

        self.radius, self.norm = radius, norm
        self.ZERO = ZERO
        self._params = radius ** (-norm), -1 / radius ** 2, -radius ** 2 * np.log(ZERO)
