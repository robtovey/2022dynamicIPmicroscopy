'''
Created on 22 Feb 2022

@author: Rob Tovey
'''
import numpy as np
from .utils import QuadraticFidelity, jit, prange, __params, __pparams, FLOAT, ZERO


@jit(**__pparams)
def _gaussian_fwrd(rho, grid, data, a, b, rad, balanced, out):
    '''
    rho.shape = [I,3*J] or [I,1+2*J] 
    grid = (dx, dy, 1/dx, 1/dy)
    data.shape = [J,Kx, Ky]
    a,b = float
    rad = (rx, ry)
    out.shape = [J,Kx, Ky]

    out[j,k] = a*(sum_i rho[i,j].a * np.exp(b*|rho[i,j].x - loc[j,k]|^2)) - data[j,k]

    if |x|_infinity > rad, then we assume np.exp(b*|x|^2) = 0   
    '''

    for j in prange(data.shape[0]):  # time
        J0 = 3 * j; J1 = 3 * j
        if balanced:
            J0 = 0; J1 = 2 * j
        expX = np.empty(2 * rad[0])  # = exp(b*x^2) for each possible x
        expY = np.empty(2 * rad[1])  # = exp(b*y^2) for each possible y

        for i in range(rho.shape[0]):  # curve number
            m = rho[i, J0]; x = rho[i, J1 + 1]; y = rho[i, J1 + 2]
            # switch to pixel coordinates, +1 accounts for rounding and non-inclusive range(a,b)
            px = int(x * grid[2]) + 1; py = int(y * grid[3]) + 1
            px0 = max(0, px - rad[0]); px1 = min(data.shape[1], px + rad[0])
            py0 = max(0, py - rad[1]); py1 = min(data.shape[2], py + rad[1])

            # pre-compute all exponentials
            for px in range(px0, px1):
                expX[px - px0] = np.exp(b * (x - px * grid[0]) ** 2)
            for py in range(py0, py1):
                expY[py - py0] = np.exp(b * (y - py * grid[1]) ** 2)

            for px in range(px0, px1):
                for py in range(py0, py1):
                    # out[j, px, py] += m * np.exp(b * ((x - px * grid[0])**2 + (y - py * grid[1])**2))
                    out[j, px, py] += m * expX[px - px0] * expY[py - py0]

        for px in range(data.shape[1]):
            for py in range(data.shape[2]):
                out[j, px, py] = a * out[j, px, py] - data[j, px, py]


@jit(**__pparams)
def _gaussian_bwrd(X, res, grid, a, b, rad, out):
    '''
    X.shape = [J,I,2]
    res.shape = [J,Kx,Ky]
    grid = (dx, dy, 1/dx, 1/dy)
    a,b = float
    rad = (rx, ry)
    out.shape = [J,I]

    out[j,i] = a*(sum_k res[j,k]*np.exp(b*|X[j,i] - loc[k]|^2))

    if |x|_infinity > rad, then we assume np.exp(b*|x|^2) = 0   
    '''
    for j in prange(out.shape[0]):  # time slice
        expX = np.empty(2 * rad[0])  # = exp(b*x^2) for each possible x
        expY = np.empty(2 * rad[1])  # = exp(b*y^2) for each possible y

        XX = X[0] if X.shape[0] == 1 else X[j]
        for i in range(out.shape[1]):  # coordinate of evaluation
            x = XX[i, 0]; y = XX[i, 1]
            # switch to pixel coordinates, +1 accounts for rounding and non-inclusive range(a,b)
            px = int(x * grid[2]) + 1; py = int(y * grid[3]) + 1
            px0 = max(0, px - rad[0]); px1 = min(res.shape[1], px + rad[0])
            py0 = max(0, py - rad[1]); py1 = min(res.shape[2], py + rad[1])

            # pre-compute all exponentials
            for px in range(px0, px1):
                expX[px - px0] = np.exp(b * (x - px * grid[0]) ** 2)
            for py in range(py0, py1):
                expY[py - py0] = np.exp(b * (y - py * grid[1]) ** 2)

            v = 0
            for px in range(px0, px1):
                for py in range(py0, py1):
                    # v += res[j, px, py] * np.exp(b * ((x - px * grid[0])**2 + (y - py * grid[1])**2))
                    v += res[j, px, py] * expX[px - px0] * expY[py - py0]

            out[j, i] = a * v


@jit(inline='always', **__params)
def __gaussian_bwrd_diff(m, x, y, res, grid, a, b, rad, out, J0, J1, expX, expY):
    # switch to pixel coordinates, +1 accounts for rounding and non-inclusive range(a,b)
    px = int(x * grid[2]) + 1; py = int(y * grid[3]) + 1
    px0 = max(0, px - rad[0]); px1 = min(res.shape[0], px + rad[0])
    py0 = max(0, py - rad[1]); py1 = min(res.shape[1], py + rad[1])

    # pre-compute all exponentials
    for px in range(px0, px1):
        expX[px - px0] = np.exp(b * (x - px * grid[0]) ** 2)
    for py in range(py0, py1):
        expY[py - py0] = np.exp(b * (y - py * grid[1]) ** 2)

    v = 0; dvx = 0; dvy = 0
    for px in range(px0, px1):
        dx = x - px * grid[0]
        for py in range(py0, py1):
            dy = y - py * grid[1]
            # g = res[px, py] * np.exp(b * (dx**2 + dy**2))
            g = res[px, py] * expX[px - px0] * expY[py - py0]
            v += g
            g *= 2 * m * b
            dvx += dx * g
            dvy += dy * g

    out[J0] += a * v
    out[J1 + 1] = a * dvx
    out[J1 + 2] = a * dvy


@jit(**__pparams)
def _gaussian_bwrd_diff(rho, res, grid, a, b, rad, balanced, out):
    '''
    rho.shape = [I,3*J] or [I,1+2*J] 
    res.shape = [J,Kx,Ky]
    grid = (dx, dy, 1/dx, 1/dy)
    a,b = float
    rad = (rx, ry)
    out.shape = rho.shape

    Let
        F(rho) = a * sum_{i,j} rho.a[i,j] * sum_k res[j,k] * np.exp(b*|rho.x[i,j] - loc[j,k]|^2)),
    then we return F and out[i,j] = dF/drho[i,j].
    '''

    F = 0
    if balanced:
        for i in prange(rho.shape[0]):
            expX = np.empty(2 * rad[0])  # = exp(b*x^2) for each possible x
            expY = np.empty(2 * rad[1])  # = exp(b*y^2) for each possible y
            m = rho[i, 0]
            for j in range(res.shape[0]):  # time
                J = 2 * j
                __gaussian_bwrd_diff(
                    m, rho[i, J + 1], rho[i, J + 2], res[j], grid, a, b, rad, out[i], 0, J, expX, expY)

            F += m * out[i, 0]  # total energy
    else:
        for i in prange(rho.shape[0]):
            expX = np.empty(2 * rad[0])  # = exp(b*x^2) for each possible x
            expY = np.empty(2 * rad[1])  # = exp(b*y^2) for each possible y
            for j in range(res.shape[0]):  # time
                J = 3 * j
                m = rho[i, J]
                __gaussian_bwrd_diff(
                    m, rho[i, J + 1], rho[i, J + 2], res[j], grid, a, b, rad, out[i], J, J, expX, expY)
                F += m * out[i, J]  # total energy

    return F


class GaussianGridFidelity(QuadraticFidelity):
    '''
    This object represents the function
        F(rho) = .5*scale * sum_{i,j} (g\star rho[i](loc[j]) - data[i,j])^2
    where i is a time coordinate, j is space, g is a gaussian
        g(x) = radius^-norm * exp(-x^2/radius^2)
    and loc is given by
        loc[px,py] = px*dx + py*dy
    for some floats (dx,dy).


    Parameters
    ----------
    grid
        Pair (dx,dy) so that loc[px,py] = px*dx + py+dy. Assumed constant in time.
        # TODO: implement time-varying grid?
    radius, norm, data
        See above
    scale=dx*dy/data.shape[0]
        Scales the full function
    ZERO=utils.ZERO (= 1e-10)
        Thresholds the value of the Gaussian at this level


    Attributes
    ----------
    data, scale
        Can be updated to change the behaviour of the methods of this instance.
    grid, radius, norm, ZERO
        See above. Note that changing these attributes will not change the behaviour
        of this instance.


    See more in help(QuadraticFidelity)
    '''

    def __init__(self, grid, radius, norm, data, scale=None, ZERO=ZERO):
        def fwrd(rho, data, balanced, out):
            return _gaussian_fwrd(rho, self.grid, data, *self._params, balanced, out)

        def bwrd(x, res, out):
            return _gaussian_bwrd(x, res, self.grid, *self._params, out)

        def bwrd_diff(rho, res, balanced, out):
            return _gaussian_bwrd_diff(rho, res, self.grid, *self._params, balanced, out)

        if scale is None:
            scale = grid[0] * grid[1]

        super().__init__(fwrd, bwrd, bwrd_diff, FLOAT, data, scale=scale)

        grid = np.array((grid[0], grid[1], 1 / grid[0], 1 / grid[1]), dtype='float64')

        self.grid, self.radius, self.norm, self.ZERO = grid, radius, norm, ZERO
        self._params = (radius ** (-norm), -1 / radius ** 2,
                        np.ceil((-np.log(ZERO)) ** .5 * radius * grid[2:]).astype(int))

    def gif(self, filename, rho=None, residual=False, scale=None, trace=False, trace_thresh=(.01, .5)):
        '''
        TODO: write this
        '''
        data = self.data if ((rho is None) or trace) else self.fwrd(rho)
        if residual:
            data = data - self.data
        if type(filename) is not str:
            from os import path
            filename = path.join(*filename)

        if data.ndim == 2:  # not yet a video
            raise NotImplementedError

        from PIL import Image
        scale = (255 / data.max()) if scale is None else scale
        data = np.maximum(0, np.minimum(255, data * scale)).astype('uint8')
        if trace and rho.shape[0] > 0:
            trace = int(trace)
            assert rho is not None
            data = np.ascontiguousarray(np.tile(data[..., None], (1, 1, 1, 3)))
            from matplotlib import pyplot
            color = pyplot.get_cmap('Set1')
            color = [color(i / 9)[:3] for i in range(8)]  # 8 nicely contrasting RGB colors
            color = (255 * np.array([color[i % 8] for i in range(rho.shape[0])])).astype('uint8')
            trace_thresh = np.array(trace_thresh, dtype='float64') * rho.a.max()
            trace_scale = 1 / (trace_thresh[1] - trace_thresh[0])
            # alpha value will be 0 for rho.a< trace_thresh[0] and 1 for rho.a > trace_thresh[1]

            @jit(nopython=True)
            def insert_color(data, A, X):
                for s in range(data.shape[0]):
                    for t in range(max(0, s + 1 - trace), s + 1):
                        for i in range(X.shape[0]):
                            a = (A[i, 0 if A.shape[1] == 1 else t] - trace_thresh[0]) * trace_scale
                            if a <= 0:
                                continue
                            # # First option: round to nearest pixel
                            x = X[i, t]  # pixel of curve i at time t
                            if 0 <= x[0] < data.shape[1] and 0 <= x[1] < data.shape[2]:
                                if a >= 1:
                                    data[s, x[0], x[1]] = color[i]  # set color
                                else:
                                    for j in range(3):  # weighted blend of colors
                                        data[s, x[0], x[1], j] += int(a * (color[i, j] - data[s, x[0], x[1], j]))
                            # # Second option: balance weight between all neighbouring pixels (looks too blurred to me)
                            # x = X[i, t]  # pixel of curve i at time t
                            # if 0 < x[0] < data.shape[1] - 1 and 0 < x[1] < data.shape[2] - 1:
                            #     y0, y1 = int(x[0]), int(x[1])
                            #     for j0 in range(y0, y0 + 2):
                            #         for j1 in range(y1, y1 + 2):
                            #             w = min(1, a) * 0.25 * (2 - abs(j0 - x[0]) - abs(j1 - x[1]))
                            #             for j in range(3):  # weighted blend of colors
                            #                 data[s, j0, j1, j] += int(w * (color[i, j] - data[s, j0, j1, j]))
            insert_color(data, np.ascontiguousarray(rho.a),
                        (rho.x * self.grid[-2:].reshape(1, 1, 2)).round().astype('int32'))
                         # rho.x * self.grid[-2:].reshape(1, 1, 2))

        data = [Image.fromarray(a).transpose(2).quantize() for a in data]
        data[0].save(filename, save_all=True, append_images=data[1:],
                     duration=1e4 // len(data), loop=0)
