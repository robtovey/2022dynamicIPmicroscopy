'''
Created on 6 Dec 2021

@author: Rob
'''
import numpy as np
# Import from ../utils.py:
import os; cwd = os.getcwd(); os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from sparseDynamicRecon.utils import jit, prange, __params, __pparams, FLOAT, COMPLEX, ZERO
os.chdir(cwd)  # re-set original working directory
__guparams = {'nopython': True, 'cache': True, 'fastmath': True, 'target': 'parallel'}


def isbalanced(rho, T): return (rho.shape[1] % T != 0)


def iscomplexdtype(dtype):
    try:
        np.array(1j, dtype=dtype)
        return True
    except TypeError:
        return False


class Fidelity:
    '''
    This object represents a smooth function mapping measures to real numbers.
    Such functions must be callable, implement a (discrete) gradient, and
    implement a continuous linearisation for Frank-Wolfe.

    The measures rho must be a dynamic measure with either:
        rho_i = [mass, x_0, y_0, x_1, y_1, ...]     or
        rho_i = [m_0, x_0, y_0, m_1, x_1, y_1, ...]
    '''

    def __call__(self, rho):
        '''
        Function mapping dynamic measures to real numbers.
        '''
        raise NotImplementedError

    def grad(self, rho, discrete=True):
        '''
        If discrete, then this returns an array
            F = F(rho) and dF = dF/d(m,x,y)(rho)
        with dF.shape == rho.shape. Otherwise, it returns
            F = F(rho) and dF = dF/drho
        where dF is a function from (x,y) to the real numbers.
        '''
        if discrete:
            return self._grad(rho)
        else:
            return self.linearise(rho, energy=True)

    def linearise(self, rho, energy=False):
        '''
        If energy, then this returns
            F = F(rho) and dF = dF/drho(rho)
        where dF is a differentiable function from (x,y) to the real numbers.
        If energy is False, then this just returns dF.
        '''
        return self._linearise(rho, energy)


class LinFidelity:
    '''
    This object represents a linear function mapping measures to real numbers.
    Such functions must be callable and implement its gradient.

    We treat with ambiguity the idea that L is both a continuous function in time,
    i.e. L = L_t(x), and L is a function of measures, L(rho) = \int L(t,x) drho(t,x).

    The measures rho must be a dynamic measure with either:
        rho_i = [mass, x_0, y_0, x_1, y_1, ...]     or
        rho_i = [m_0, x_0, y_0, m_1, x_1, y_1, ...]
    '''

    def __call__(self, rho=None, x=None):
        '''
        If rho is given, this returns
            \int L(x) drho(x)
        Otherwise, we return
            L_{i,j} = L_i(x_{i,j})
        where either x has shape
            x.shape = (time, N, 2)
        or x.shape = (N,2).
        '''
        assert any(t is None for t in (rho, x))
        assert not all(t is None for t in (rho, x))
        raise NotImplementedError

    def grad(self, rho=None, x=None):
        '''
        If rho is given, this returns
            dL = dL(rho)/drho,
        otherwise, we return
            dL = dL_t(x)/dx
        where either x has shape
            x.shape = (time, N, 2)
        or x.shape = (N,2).

        In both cases we return the discrete gradient.
        '''
        assert any(t is None for t in (rho, x))
        assert not all(t is None for t in (rho, x))
        if rho is None:
            return self._grad_x(x)
        else:
            rho = rho.arr if hasattr(rho, 'arr') else rho
            return self._grad_rho(rho)


class QuadraticFidelity(Fidelity):
    '''
    This fidelity represents a function of the form
        F(rho) = .5*scale*|A rho - data|^2
    where
        |x|^2 = sum_{i=1}^{x.size} x[i]^2.

    Parameters
    ----------
    fwrd(rho, data, balanced, out)
        Computes out[:] = A rho - data
        where rho is an array with
            rho[i] = [m_i, x_i(0), y_i(0), ..., x_i(1), y_i(1)]
        if balanced, otherwise
            rho[i] = [m_i(0), x_i(0), y_i(0), ..., m_i(1), x_i(1), y_i(1)]
        and m_i(t) is the mass of curve i at time t.
    bwrd(x, res, out)
        Computes out[i] = (A^* res)(x[i])
    bwrd_diff(rho, res, balanced, out)
        Let F(rho) = \int (A^* res) drho.
        This function returns the value of F(rho) and computes its discrete
        derivative inplace in out (out.shape = rho.shape)
    dtype
        Casts most arrays to this dtype before processing with one of its methods
    data
        The data array for the fidelity
    scale=1/data.size


    Attributes
    ----------
    dtype
        The input dtype
    data
        The input data
    scale
        Normalisation of the full function
    T
        Number of time-points, i.e. data.shape[0]
    iscomplex
        Flag indicating whether the dtype is complex or not


    Methods
    -------
    res(rho)
        Returns A rho - data
    fwrd(rho)
        Returns A rho
    __call__(rho)
        Returns .5|A rho - data|^2
    grad(rho, discrete=True)
        If discrete, then this returns the scalar
            F = F(rho)
        and the array
            dF = dF(rho)/drho
        where rho is treated as a standard array.
        Otherwise it returns the scalar
            F = F(rho)
        and the linear map
            dF = A^*(A rho - data).
    linearise(rho, energy=False)
        The linearised energy is L = A^*(A rho - data)
        If energy==True, then also .5|A rho - data|^2 is returned.
    gif(filename, rho=None, residual=False, scale=None)
        Saves a gif of an object in data-space.
        If rho is provided, we take D=fwrd(rho), otherwise D=self.data.
        The scale parameter should rescale between [0, 255], the default is 1/D.max().
        If residual==False (default), then scale*D is saved, otherwise it is scale*abs(D-data).
    '''

    def __init__(self, fwrd, bwrd, bwrd_diff, dtype, data, scale=None):
        self._op = [fwrd, bwrd, bwrd_diff]
        self.dtype = dtype
        self.data = data.astype(dtype)
        self.T = self.data.shape[0]
        self.iscomplex = iscomplexdtype(dtype)

        self._scale = 1 / data.size if scale is None else scale

    def _norm(self, vec): return .5 * np.linalg.norm(vec.ravel()) ** 2

    def res(self, rho):
        '''
        return A rho - data
        '''
        rho = rho.arr if hasattr(rho, 'arr') else rho
        vec = np.zeros(self.data.shape, dtype=self.dtype)
        self._op[0](rho, self.data, isbalanced(rho, self.T), vec)
        return vec  # = A*rho - data

    def fwrd(self, rho):
        '''
        return A rho
        '''
        rho = rho.arr if hasattr(rho, 'arr') else rho
        vec = np.zeros(self.data.shape, dtype=self.dtype)
        self._op[0](rho, 0 * self.data, isbalanced(rho, self.T), vec)
        return vec  # = A*rho - 0*data

    def __call__(self, rho): return self._scale * self._norm(self.res(rho))

    def _grad(self, rho):
        rho = rho.arr if hasattr(rho, 'arr') else rho
        dF = np.zeros(rho.shape, dtype=self.dtype)
        res = self.res(rho)
        F = self._norm(res)
        self._op[2](rho, res, isbalanced(rho, self.T), dF)

        if self.iscomplex:
            F = np.real(F) + np.imag(F)  # should be real anyway
            dF = dF.real + dF.imag
        return self._scale * F, self._scale * dF

    def _linearise(self, rho, energy):
        res = self.res(rho)
        L = QuadraticLinFidelity(self._scale * res, self._op, self.dtype)
        if energy:
            return self._scale * self._norm(res), L
        else:
            return L


class QuadraticLinFidelity(LinFidelity):
    def __init__(self, res, op, dtype):
        self.res, self._op, self.dtype = res, op, dtype
        self.T = res.shape[0]
        self.iscomplex = iscomplexdtype(dtype)

    def __call__(self, rho=None, x=None):
        assert any(t is None for t in (rho, x))
        assert not all(t is None for t in (rho, x))
        if x is None:
            if hasattr(rho, 'a'):
                a, x = rho.a, rho.x
            elif isbalanced(rho, self.T):
                a, x = rho[:, :1], rho[:, 1:].reshape(-1, self.T, 2)
            else:
                rho = rho.reshape(-1, self.T, 3)
                a, x = rho[:, :, 0], rho[:, :, 1:]

            x = np.transpose(x, (1, 0, 2))  # order = (time, curve, x/y)
            a = np.transpose(a, (1, 0))  # order = (time, curve)
            return (a * self._grad_x(x)).sum()
        elif x.ndim == 2 or x.shape[0] == 1:
            x = x.reshape(1, -1, 2)
            return self._grad_x(np.broadcast_to(x, (self.T, x.shape[1], 2)))
        else:
            return self._grad_x(x)

    def _grad_x(self, x):
        # slight misnomer, the gradient returned when x is given is
        # L(x), i.e. dL/drho
        out = np.zeros(x.shape[:2], dtype=self.dtype)
        self._op[1](x, self.res, out)
        if self.iscomplex:
            out = out.real + out.imag
        return out

    def _grad_rho(self, rho):
        out = np.zeros(rho.shape, dtype=self.dtype)
        F = self._op[2](rho, self.res, isbalanced(rho, self.res.shape[0]), out)
        if self.iscomplex:
            F = F.real + F.imag
            out = out.real + out.imag
        return F, out
