'''
Created on 6 Dec 2021

@author: Rob
'''
import numpy as np
from .utils import QuadraticFidelity, jit, prange, __params, __pparams, COMPLEX


@jit(**__pparams)
def _fourier_fwrd(rho, loc, data, balanced, out):
    '''
    rho.shape = [I,3*J] or [I,1+2*J]
    loc.shape = [K,2]
    data.shape = [J,K]
    out.shape = [J,K]

    out[j,k] = sum_i rho[i,j].a * np.exp(-i * rho[i,j].x*loc[k]) - data[j,k]
    '''
    for j in prange(data.shape[0]):  # time
        J0 = 3 * j; J1 = 3 * j;
        if balanced:
            J0 = 0; J1 = 2 * j
        for k in range(data.shape[1]):  # coordinate of data
            x = loc[j, k, 0]
            y = loc[j, k, 1]
            v = 0
            for i in range(rho.shape[0]):  # curve number
                X = rho[i, 1 + J1]; Y = rho[i, 2 + J1]
                a = rho[i, J0]
                v += a * np.exp(-1j * (X * x + Y * y))

            out[j, k] = v - data[j, k]


@jit(**__pparams)
def _fourier_bwrd(X, res, loc, out):
    '''
    X.shape = [J,I,2]
    res.shape = [J,K]
    loc.shape = [J,K,2]
    out.shape = [J,I]

    out[j,i] = sum_k res[j,k] * np.exp(-i * X[j,i]*loc[k] )
    '''
    for j in prange(out.shape[0]):
        x = X[0] if X.shape[0] == 1 else X[j]
        for i in range(out.shape[1]):

            v = 0
            for k in range(res.shape[1]):  # coordinate of data
                g = np.exp(-1j * (x[i, 0] * loc[j, k, 0] + x[i, 1] * loc[j, k, 1]))
                v += complex(res[j, k].real * g.real, res[j, k].imag * g.imag)

            out[j, i] = v


@jit(inline='always', **__params)
def __fourier_bwrd_diff(m, x, y, res, loc, out, J0, J1):

    v = 0; dvx = 0; dvy = 0
    for k in range(res.shape[0]):  # coordinate of data
        g = np.exp(-1j * (x * loc[k, 0] + y * loc[k, 1]))
        # v = res.r*cos(loc*x) - 1j * res.i*sin(loc*x)
        v += complex(res[k].real * g.real, res[k].imag * g.imag)
        # dv = -res.r*loc.x*sin(loc*x) - 1j * res.i*loc.i*cos(loc*x)
        res_g = complex(res[k].real * g.imag, -res[k].imag * g.real)
        dvx += loc[k, 0] * res_g
        dvy += loc[k, 1] * res_g

    out[J0] += v
    out[1 + J1] = m * dvx
    out[2 + J1] = m * dvy


@jit(**__pparams)
def _fourier_bwrd_diff(rho, res, loc, balanced, out):
    '''
    rho.shape = [I,3*J] or [I,1+2*J]
    res.shape = [J,K]
    loc.shape = [J,K,2]
    out.shape = rho.shape

    Let
        F(rho) = a * sum_{i,j} rho.a[i,j] * sum_k res[j,k] * np.exp(-i * rho.x[i,j]*loc[k] ),
    then we return F and out[i,j] = dF/drho[i,j].
    '''

    F = 0
    if balanced:
        for i in prange(rho.shape[0]):
            m = rho[i, 0]
            for j in range(res.shape[0]):  # time
                J = 2 * j
                __fourier_bwrd_diff(m, rho[i, J + 1], rho[i, J + 2], res[j], loc[j], out[i], 0, J)
            F += m * out[i, 0]  # total energy
    else:
        for i in prange(rho.shape[0]):
            for j in range(res.shape[0]):  # time
                J = 3 * j
                m = rho[i, J]
                __fourier_bwrd_diff(m, rho[i, J + 1], rho[i, J + 2], res[j], loc[j], out[i], J, J)
                F += m * out[i, J]  # total energy

    return F


class FourierFidelity(QuadraticFidelity):
    '''
    This object represents the function
        F(rho) = .5*scale * sum_{i,j} (\int g_j(x) drho[i](x) - data[i,j])^2
    where i is a time coordinate, g_j is the Fourier kernel
        g_j(x) = exp(-i x\cdot loc[j])

    Parameters
    ----------
    locations
        Array of shape [T, N, 2] or [N, 2] where N is the number of samples
        at each time-point. Assumed constant in time in the latter case.
    data, scale=1/data.size
        See above


    Attributes
    ----------
    locations, scale
        Can be updated to change the behaviour of the methods of this instance.


    See more in help(QuadraticFidelity)
    '''

    def __init__(self, locations, data, scale=None):
        def fwrd(rho, data, balanced, out):
            return _fourier_fwrd(rho, self.locations, data, balanced, out)

        def bwrd(x, res, out):
            return _fourier_bwrd(x, res, self.locations, out)

        def bwrd_diff(rho, res, balanced, out):
            return _fourier_bwrd_diff(rho, res, self.locations, balanced, out)

        if scale is None:
            scale = 1 / data.size

        super().__init__(fwrd, bwrd, bwrd_diff, COMPLEX, data, scale=scale)

        locations = locations.reshape(-1, self.data.shape[1], 2)
        assert locations.shape[0] in (1, self.T)
        self.locations = np.tile(locations, (self.T, 1, 1)) if locations.shape[0] == 1 else locations
