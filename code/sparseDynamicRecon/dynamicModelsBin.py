'''
Created on 9 Feb 2021

@author: Rob Tovey
'''

from os import path, makedirs
import numpy as np
from .utils import jit, __params, plt, FLOAT, COMPLEX
from matplotlib.collections import LineCollection
from .FunctionSpaces import FunctionSpace, Function, IndexedSpace, IndexedArray
from .OTPathUtils import UBpathOT, pathOT


@jit(**__params)
def _distance(indx1, arr1, indx2, arr2):
    d, n = 0.0, arr1.shape1
    i, j, I, J = 0, 0, indx1[0], indx2[0]

    while True:
        if I < J:
            d += arr1[i, 0] ** 2
            i += 1
            if i == indx1.size:
                break
            I = indx1[i]
        elif I > J:
            d += arr2[j, 0] ** 2
            j += 1
            if j == indx2.size:
                break
            J = indx2[j]
        else:
            for k in range(n):
                d += (arr1[i, k] - arr2[j, k]) ** 2
            i += 1; j += 1
            if i == indx1.size or j == indx2.size:
                break
            I, J = indx1[i], indx2[j]

    while i < indx1.size:
        d += arr1[i, 0] ** 2
        i += 1
    while j < indx2.size:
        d += arr2[j, 0] ** 2
        j += 1

    return np.sqrt(d)


@jit(cache=True)
def sumarr(old, new, index):
    for i in range(index.size):
        new[index[i]] += old[i]


class CurveSpace(FunctionSpace):
    def __init__(self, dim, T, balanced=True):
        if balanced:
            size = 1 + dim * len(T)  # array is (m, x0, y0, x1, y1, ...)
        else:
            size = (1 + dim) * len(T)  # array is (m0, x0, y0, m1, x1, y1, ...)
        super().__init__((dim, T, balanced, size))
        self._element = CurveMeasure
        self._dt = (T[-1] - T[0]) / len(T)
        self._slice = slice(1) if self.balanced else slice(None, None, 1 + self.dim)

    @property
    def shape(self): return self.this[-1],
    @property
    def size(self): return self.this[-1]
    @property
    def dim(self): return self.this[0]
    @property
    def T(self): return self.this[1]
    @property
    def balanced(self): return self.this[2]

    def distance(self, f, g, this=None):
        f = f.asarray() if hasattr(f, 'asarray') else f
        g = g.asarray() if hasattr(g, 'asarray') else g
        # return vector L1 norm over masses/locations
        return abs(f - g).sum() * self._dt

    def norm(self, f, p=1, this=None):
        f = f.asarray() if hasattr(f, 'asarray') else f
        if self.balanced:
            return abs(f[0]) * (self.this[1][-1] - self.this[1][0])
        elif p == 1:
            return abs(f[::1 + self.dim]).sum() * self._dt
        else:
            return (abs(f[::1 + self.dim]) ** p).sum() ** (1 / p) * self._dt


class CurveMeasure(Function):
    def __init__(self, arr, FS, this=None):
        super().__init__(FS, this)
        self.arr = arr
        self.dim, self.T, self.balanced, self.size = FS.this
        self._slice = FS._slice

    @property
    def a(self): return self.arr[self._slice]

    @property
    def x(self): return self.arr[1:].reshape(-1,
                                             self.dim) if self.balanced else self.arr.reshape(-1, 1 + self.dim)[:, 1:]

    def asarray(self): return self.arr

    def copy(self, like=None):
        if like is None:
            arr = self.arr.copy()  # deep copy
        elif np.isscalar(like):
            return self.FS.one(like, self.this)  # create a scalar vector
        elif hasattr(like, 'asarray'):
            arr = like.asarray()
        else:
            arr = like
        return self._copy(arr, self.FS, self.this)

    def _copy(self, arr, FS, this): return self.FS._element(arr, FS, this)

    def plot(self, FS, ax=None, update=None, cmap=None, **kwargs):
        ax = plt.gca() if ax is None else ax
        if self.dim == 1:  # location on x and mass on y
            raise NotImplementedError
        elif self.dim == 2:  # location on (x,y) and mass by shading
            cmap = plt.cm.get_cmap(cmap)
            a, x = self.a, self.x
            vmax = a.max()
            c = [cmap(c / vmax) for c in a]
            if update is None:
                return ax.scatter(x[:, 0], x[:, 1], c=c,
                                  vmin=0, vmax=vmax, cmap='gray', **kwargs)
            else:
                update.set_offsets(x[:, 1:])
                update.set_facecolors(c)
                return update
        else:
            raise NotImplementedError

    def __add__(self, other):
        if hasattr(other, 'FS'):
            assert other.FS == self.FS
            other = other.arr
        out = self.arr.copy()
        out[self._slice] += other[self._slice]
        return self._copy(out, self.FS, self.this)

    def __sub__(self, other):
        if hasattr(other, 'FS'):
            assert other.FS == self.FS
            other = other.arr
        out = self.arr.copy()
        out[self._slice] -= other[self._slice]
        return self._copy(out, self.FS, self.this)

    def __mul__(self, other):
        assert np.isscalar(other)
        out = self.arr.copy()
        out[self._slice] *= other
        return self._copy(out, self.FS, self.this)

    def __truediv__(self, other):
        assert np.isscalar(other)
        out = self.arr.copy()
        out[self._slice] /= other
        return self._copy(out, self.FS, self.this)

    def __imul__(self, other):
        assert np.isscalar(other)
        self.arr[self._slice] *= other

    def __itruediv__(self, other):
        assert np.isscalar(other)
        self.arr[self._slice] /= other

    def __neg__(self):
        out = self.arr.copy()
        out[self._slice] = -out[self._slice]
        return self._copy(out, self.FS, self.this)

    __radd__ = __add__
    __rmul__ = __mul__

    def norm(self, p=1): return self.FS.norm(self, p, this=self.this)


class DynamicMeasureSpace(IndexedSpace):
    def __init__(self, CS, n=0):
        super().__init__((n, CS.size))
        self.T, self.balanced, self.FS = CS.T, CS.balanced, CS
        self._element = DynamicMeasure

    @property
    def this(self): return (self._shape, self._ind, self.FS)

    @this.setter
    def this(self, value):
        # This function never changes the child FS
        if hasattr(value, 'ind'):
            self._shape, self._ind = value.shape, value.ind
        else:
            self._shape, self._ind = value[0], value[1]
        assert self._shape[0] == self._ind.shape[0]

    def __len__(self): return self._shape[0]

    def distance(self, f, g, this=None):
        f = f.asarray() if hasattr(f, 'asarray') else f
        g = g.asarray() if hasattr(g, 'asarray') else g
        # return vector L1 norm over masses/locations
        N = min(f.shape[0], g.shape[0])
        return (abs(f[:N] - g[:N]).sum() * self.FS._dt
                + self.norm(f[N:]) + self.norm(g[N:]))

    def norm(self, f, p=1, this=None):
        f = f.asarray() if hasattr(f, 'asarray') else f
        if f.size == 0:
            return 0
        # array[:, j] = atom j
        if self.balanced:
            return abs(f[:, 0]).sum() * (self.T[-1] - self.T[0])
        else:
            return np.linalg.norm(f[:, self.FS._slice], ord=p, axis=1).sum() * self.FS._dt

    def __iter__(self):
        self._index = -1
        return self

    def __next__(self):
        self._index += 1
        if self._index == self.size:
            raise StopIteration
        else:
            return self.FS  # TODO: this is a weird thing to return...

    def __getitem__(self, indx): return self.FS  # TODO: this is a weird thing to return...


class DynamicMeasure(IndexedArray):
    def __init__(self, arr, FS=None, this=None):
        if isinstance(FS, CurveSpace):
            FS = DynamicMeasureSpace(FS)
        if hasattr(arr, 'shape'):
            arr = arr.reshape(-1, FS.FS.size)  # numpy array
        elif isinstance(arr, DynamicMeasure):
            arr = arr.arr
        elif isinstance(arr[0], CurveMeasure):  # list of CurveMeasures
            arr = np.concatenate([a.asarray()[None] for a in arr], axis=0)
        else:  # list of arrays
            arr = np.concatenate([a[None] for a in arr], axis=0)
        super().__init__(arr, FS, this)
        self.dim = self.FS.FS.dim

    def __next__(self):
        self._index += 1
        if self._index == self.shape[0]:
            raise StopIteration
        else:
            return self.FS.FS._element(self.arr[self._index], self.FS.FS)

    def __getitem__(self, indx):
        if type(indx) is int:
            return self.FS.FS._element(self.arr[indx], self.FS.FS)
        elif type(indx) is not tuple:
            arr = self.arr[indx]
            return self._copy(arr, self.FS, (arr.shape, self.this[1][indx], self.this[2]))
        else:
            # indx = (slice0, slice1)
            # slice0 is a slice of curves,
            # slice1 is a slice in time
            # Returned array is indexed of the form [curve, time, (mass,x,y)]
            if self.FS.balanced:
                x = self.arr[:, 1:].reshape(-1, len(self.FS.T), self.dim)[indx[0], indx[1], :]
                if x.ndim == 2:
                    a = self.arr[indx[0], :1]
                else:
                    a = np.tile(self.arr[indx[0], None, :1], (1, x.shape[1]))
                return np.concatenate((a, x), axis=-1)
            else:
                return self.arr.reshape(-1, len(self.FS.T), 1 + self.dim)[indx]

    @property
    def size(self): return self.arr.size
    @property
    def T(self): return self.FS.T
    @property
    def a(self): return self.arr[:, self.FS.FS._slice]

    @property
    def x(self):
        if self.FS.balanced:
            return self.arr[:, 1:].reshape(-1, len(self.FS.T), self.dim)
        else:
            return self.arr.reshape(-1, len(self.FS.T), 1 + self.dim)[:, :, 1:]

    def plot(self, ax=None, update=None, width=1, minwidth=.5, cmap='viridis', alpha=None, paths=True, decimals=6):
        ax = plt.gca() if ax is None else ax
        dim = self.dim
        assert dim == 2
        ZERO = 10 ** -decimals

        # first index is line number, second is time, third is (x,y) coordinate
        if self.FS.balanced:
            mass = np.tile(width * self.arr[:, :1], (1, len(self.T)))
            loc = self.arr[:, 1:].reshape(-1, len(self.T), dim)
        else:
            mass = width * self.arr[:, self.FS.FS._slice]
            loc = self.arr.reshape(-1, len(self.T), 1 + dim)[:, :, 1:]

        if paths:
            # remove duplicate PATHS, not duplicate points
            loc, ind = np.unique(loc.round(decimals), axis=0, return_inverse=True)

            tmp = np.zeros(loc.shape[:2])
            sumarr(mass, tmp, ind)  # remove duplicates from mass
            mass = tmp

            loc = np.concatenate((loc[:, :-1, :], loc[:, 1:, :]), axis=2).reshape(-1, 2, 2)
#             mass = .5 * (mass[:,:-1] + mass[:, 1:]).ravel()
            mass = np.minimum(mass[:, :-1], mass[:, 1:]).ravel()  # ignores jumps to 0 mass
            mass = np.maximum(minwidth, mass) * (mass > ZERO)

            if dim == 1:
                # plot time on the y-axis, position in x, thickness is mass
                raise NotImplementedError
            elif dim == 2:
                if paths == 'Bredies':
                    mass = np.minimum(1, mass * (1 - alpha) / mass.max() + alpha)  # normalise [0,1]
                    # plot position in x/y, thickness is alpha, time is colour, alpha is mass
                    if update is None:
                        update = LineCollection(loc, linewidth=width, cmap=cmap,
                                                norm=plt.Normalize(self.T[0], self.T[-1]))
                        tmp = np.tile(.5 * (self.T[:-1] + self.T[1:]), self.arr.shape[0])  # times
                        update.set_array(tmp)
                        tmp = update.cmap(tmp)  # colors
                        tmp[:, -1] = mass  # set alpha
                        update.set_colors(tmp)
                        ax.add_collection(update)
                    else:
                        update.set(paths=loc, cmap=cmap,
                                   array=np.tile(.5 * (self.T[:-1] + self.T[1:]), self.arr.shape[0]))
                else:
                    # plot position in x/y, thickness is mass, time is colour
                    if update is None:
                        update = LineCollection(loc, linewidth=mass, cmap=cmap, norm=plt.Normalize(self.T[0], self.T[-1]),
                                                alpha=.75 if alpha is None else alpha)
                        update.set_array(np.tile(.5 * (self.T[:-1] + self.T[1:]), self.arr.shape[0]))
                        ax.add_collection(update)
                    else:
                        update.set(paths=loc, linewidth=mass, cmap=cmap,
                                   array=np.tile(.5 * (self.T[:-1] + self.T[1:]), self.arr.shape[0]))
                return update
        else:
            if dim == 1:
                # plot time on the y-axis, position in x, thickness is mass
                raise NotImplementedError
            elif dim == 2:
                # plot position in x/y, thickness is mass, time is colour
                cmap = plt.cm.get_cmap(cmap)
                if update is None:
                    out = []
                    for i in range(len(self.T)):
                        arr = np.concatenate((mass[:, i, None], loc[:, i, :]), axis=1)
                        out.append(ax.scatter(arr[:, 1], arr[:, 2], s=np.maximum(minwidth, arr[:, 0] ** 2), alpha=.75 if alpha is None else alpha,
                                              c=np.tile(cmap(self.T[i]), (arr.shape[0], 1)),
                                              vmin=self.T[0], vmax=self.T[-1]))
                    return out
                else:
                    for i in range(len(update)):
                        arr = np.concatenate((mass[:, i, None], loc[:, i, :]), axis=1)
                        update[i].set(offsets=arr[:, 1:], sizes=np.maximum(minwidth, arr[:, 0] ** 2),
                                      facecolors=np.tile(cmap(self.T[i]), (arr.shape[0], 1)))
                return update

    def asarray(self): return self.arr
    def ravel(self): return self.arr.ravel()


def isbalanced(rho, T=None):
    if hasattr(rho, 'FS'):
        return rho.FS.balanced
    else:
        return rho.shape[1] == 2 * T + 1


def save(filename, DM=None, CB=None):
    args = {}
    if DM is not None:  # Dynamic measure provided
        args['arr'] = DM.arr
        args['T'] = DM.T
        args['balanced'] = DM.FS.balanced
    if CB is not None:  # Convergence values provided
        args['i'] = CB.i
        args['times'] = CB.T
        if hasattr(CB, 'E'):
            args['E'] = CB.E
            args['keys'] = CB.extras()

    if type(filename) is not str:
        filename = path.join(*filename)
    D = path.dirname(filename)
    if len(D) > 0 and not path.exists(D):
        makedirs(path.dirname(filename))
    np.savez(filename, **args)


def load(filename):
    if type(filename) is not str:
        filename = path.join(*filename)
    if not filename.endswith('.npz'):
        filename += '.npz'

    with np.load(filename) as args:
        class callback_info:
            pass  # dummy object to load convergence data into

        ret = []
        if 'arr' in args:
            arr = args['arr']
            FS = DynamicMeasureSpace(CurveSpace(dim=2, T=args['T'], balanced=args['balanced']), n=arr.shape[0])
            DM = FS.element(arr)
            ret.append(DM)
        if 'times' in args:
            CM = callback_info()
            CM.T = args['times']
            ret.append(CM)
        if 'i' in args:
            CM.i = args['i']
        if 'E' in args:
            CM.E = args['E']
            keys = tuple(args['keys'])
            CM.extras = lambda *_: keys
    return tuple(ret)


def example(n=1, T=None, noise=0, balanced=True):
    '''
        min sum_i fidelity_i(rho(t_i)) + alpha|rho|_1 + beta B(rho)
    where B is the balanced Benamou Brenier energy.
    '''

    if T is None:
        T = (51 if n == 2 else 21)
    FS = DynamicMeasureSpace(CurveSpace(dim=2, T=np.linspace(0, 1, T), balanced=balanced))
    if n == 1:
        alpha, beta, datasz = .1, .1, 20
        t = np.arange(datasz).reshape(-1, 1)  # data sampling time
        frequencies = (0 + .2 * t) * np.concatenate((np.cos(t), np.sin(t)), axis=1)

        FS.extend(1)
        GroundTruth = FS.zero()
        GroundTruth[0].a[:] = 1
        GroundTruth[0].x[:] = [[.2 + .6 * t] * 2 for t in FS.T]
    elif n == 2:
        alpha, beta, datasz = .1, .1, 15

        angles = [(i * np.pi / (5 - 1)) % np.pi for i in range(T)]
        scale = np.arange(-(datasz // 2), datasz - (datasz // 2))
        frequencies = np.concatenate([scale[None, :, None] * np.array([np.cos(t), np.sin(t)])
                                      [None, None, :] for t in angles], axis=0)

        FS.extend(3)
        GroundTruth = FS.zero()
        params1 = np.array([.1, .9]), np.linalg.norm([.1 - .5, .9 - .5]) - .1
        params2 = np.array([.5] * 2) + np.array([4, -3]) * .3 / 5, .3
        angles2 = 4.5 * np.pi / 4, -2.94  # original version
#         angles2 = np.pi / 2 + np.arcsin(4 / 5) + (.5 / .8) ** 2 * 2.94, -2.94 # curve at t=.5 is (.5,.5)
        params2 = params2 + (params2[1] * np.array([f(angles2[0] + angles2[1])
                                                    for f in (np.cos, np.sin)]) + params2[0],)

        GroundTruth.a[:] = 1
        # straight diagonal line
        GroundTruth[0].x[:] = [[.2 + .6 * t, .1 + .8 * t] for t in FS.T]
        # circle segment centered at params1[0] of radius params1[1]
        GroundTruth[1].x[:] = params1[1] * np.array([[f((3 + t) * np.pi / 2)
                                                      for f in (np.cos, np.sin)] for t in FS.T]) + params1[0]
        GroundTruth[2].x[:] = [
            params2[1] * np.array([f(angles2[0] + (t / .8) ** 2 * angles2[1]) for f in (np.cos, np.sin)]) + params2[0]
            if t < .8 else
            list((t - .8) / .2 * params2[0] + (1 - t) / .2 * params2[2])
            for t in FS.T]

    else:
        alpha, beta, datasz = .5, .5, 20
        t = np.arange(datasz).reshape(-1, 1)  # data sampling time
        frequencies = (0 + .2 * t) * np.concatenate((np.cos(t), np.sin(t)), axis=1)

        FS.extend(2)
        GroundTruth = FS.zero()
        GroundTruth[0].a[:] = GroundTruth[1].a[:] = 1
        GroundTruth[0].x[:] = [[.2 + .6 * t, .2 + .6 * t] for t in FS.T]
        GroundTruth[1].x[:] = [[.8 - .6 * t, .2 + .6 * t] for t in FS.T]

    if not balanced:  # Phantoms with time-varying masses
        weight = [  # average mass 1
            lambda t: .5 * (1 + 3 * t ** 2),
            lambda t: 1.5 * (1 - t) ** .5,
            lambda t: 1.25 - 3 * (t - .5) ** 2,
        ]
        a = GroundTruth.a
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                a[i, j] = weight[i](FS.T[j])

    from .fidelity import WindowedFourier
    fidelity = WindowedFourier(2 * np.pi * frequencies, data=np.zeros((T, datasz), dtype=COMPLEX))
    fidelity.data = fidelity.fwrd(GroundTruth)

    if noise > 0:
        np.random.seed(123)
        noise_vec = np.random.normal(0, noise, fidelity.data.shape)
        noise_vec *= np.linalg.norm(fidelity.data.ravel()) / np.linalg.norm(noise_vec.ravel())
        fidelity.data += noise * noise_vec

    if balanced:
        OT = pathOT(name='Benamou Brenier', weight=(alpha, beta), T=FS.T)

    else:
        OT = UBpathOT(name='Benamou Brenier', weight=(alpha, beta, beta * 1e-2), T=FS.T)

    return fidelity, OT, GroundTruth
