'''
Created on 29 Oct 2021

@author: Rob Tovey
'''
import numpy as np
from .utils import FLOAT, INT, jit, __params, plt


class FunctionSpace:  # copy from adaptive_spaces.py

    def __init__(self, this=None): self.this = this; self._element = Function

    def update(self, f, inplace=True):
        if f.FS != self:
            if inplace:
                f.FS = self
            else:
                f = self._element(f, self, self.this)
        return f

    def element(self, arr, this=None): return self._element(arr, self, this)

    def zero(self, this=None): return self._element(self._zero(this), self, this)
    def ones(self, c=1, this=None): return self._element(self._ones(c, this), self, this)
    def rand(self, seed=None, this=None): return self._element(self._rand(this, seed), self, this)

    def inner(self, f, g, this=None): return np.dot(f.ravel(), g.ravel())
    def integrate(self, f, this=None): return f.ravel().sum()
    def norm(self, f, p=2, this=None): return np.linalg.norm(f.ravel(), ord=p)
    def distance(self, f, g, this=None): return self.norm(f - g)

    def _zero(self, this, shape=None): return np.zeros(self.shape if shape is None else shape)
    def _ones(self, c, this, shape=None): tmp = np.empty(self.shape if shape is None else shape); tmp.fill(c); return tmp
    def _rand(self, this, seed=None, shape=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.rand(*(self.shape if shape is None else shape))

    def __eq__(self, other): return self is other


class ArraySpace(FunctionSpace):

    def __init__(self, shape):
        '''
        Numpy space with fixed array shape
        '''
        super().__init__((shape,) if np.isscalar(shape) else shape)
        self._element = Array

    @property
    def shape(self): return self.this

    @shape.setter
    def shape(self, value): self.this = value

    @property
    def size(self): return np.prod(self.shape)

    @property
    def ndim(self): return len(self.shape)

    def update(self, f, inplace=True):
        assert f.size == self.size
        if f.this != self.this:
            if inplace:
                f.this = f.arr.shape = self.this
            else:
                f = self._element(f.arr.reshape(self.this), self, self.this)
        return f

    def element(self, arr, this=None):
        this = self.this if this is None else this
        if hasattr(arr, 'this'):
            if this is not arr.this:
                self.update(arr, this)
            return arr
        else:
            arr = arr.asarray() if hasattr(arr, 'asarray') else arr
            if arr.shape != this:
                arr = arr.reshape(this)
            return self._element(arr, self, this)


class DiscreteSpace(ArraySpace):

    def __init__(self, x=None, x0=None, x1=None, dx=None, n=None, interp=None):
        if x is None:
            assert all(t is not None for t in (x0, n))
            dim = max(len(thing) for thing in (x0, x1, dx, n, [1]) if hasattr(thing, '__len__'))
            x0 = np.array([x0] * dim) if np.isscalar(x0) else np.array(x0).ravel()
            # isscalar might not be a good test, np.ndim(n)==0?
            n = np.array([n] * dim, dtype=INT) if np.isscalar(n) else np.array(n).ravel()

            if dx is None:
                assert x1 is not None
                x1 = [x1] * dim if np.isscalar(x1) else x1
                dx = [(x1[i] - x0[i]) / (n[i] - 1) for i in range(len(n))]
            else:
                dx = [dx] * dim if np.isscalar(dx) else dx

            n, x0 = n.astype(INT), x0.astype(FLOAT)
            dx = np.array(dx, dtype=FLOAT).ravel()

            if dim == 1:
                x = (x0 + dx * np.arange(n)).reshape(-1, 1)
                for thing in (x0, dx, n):
                    thing.shape = 1,

            shape, regular_mesh = tuple(n), True
        else:
            if x.shape[0] == x.size:
                x = x.ravel()
                x = x.reshape(-1, 1)
                dim = 1
            else:
                dim = x.shape[1]
            shape, regular_mesh = x.shape[:1], False

        if interp is not None:
            raise NotImplementedError
        super().__init__(shape)
        self.regular_mesh, self.x, self.mesh = regular_mesh, x, (x0, dx, n)
        self.dim, self.interp = dim, interp
        self._element = Discrete

    def to_points(self):
        if self.x is None:
            x0, dx, n = self.mesh
            x = np.meshgrid(*(x0[i] + dx[i] * np.arange(n[i]) for i in range(len(n))), indexing='ij')
            self.x = np.concatenate([xx.reshape(-1, 1) for xx in x], axis=1)
        return self.x

    def __iter__(self):
        if self.x is not None:
            self._multi_index = -1
        else:
            self._multi_index = [0] * (self.dim - 1) + [-1]
        return self

    def __next__(self):
        if self.x is not None:
            self._multi_index += 1
            if self._multi_index == self.x.shape[0]:
                raise StopIteration
            elif self.dim == 1:
                return np.array(self.x[self._multi_index], dtype=FLOAT, ndmin=1)
            else:
                return self.x[self._multi_index]
        else:
            self._multi_index[-1] += 1
            for i in range(self.dim - 1, 0, -1):
                if self._multi_index[i] == self.mesh[2][i]:
                    self._multi_index[i] = 0
                    self._multi_index[i - 1] += 1
            if self._multi_index[0] == self.mesh[2][0]:
                raise StopIteration
            return np.array([self.mesh[0][i] + self._multi_index[i] * self.mesh[1][i]
                          for i in range(self.dim)], dtype=FLOAT)

    def __getitem__(self, indx):
        if self.x is not None:
            return self.x[indx]
        else:  # dimension > 1 and mesh
            if np.isscalar(indx):
                indx = np.unravel_index(indx, self.mesh[2])
            x = [self.mesh[0][i] + indx[i] * self.mesh[1][i] for i in range(self.dim)]
            return np.array(x, dtype=FLOAT, ndmin=1)


class Function:

    def __init__(self, FS, this=None):
        self.FS = FS
        self.this = this

    def update(self, arr=None, FS=None): raise NotImplementedError

    def inner(self, other): return self.FS.inner(self, other, self.this)
    def norm(self): return self.FS.norm(self, this=self.this)
    def distance(self, other): return self.FS.distance(self, other)

    def integrate(self): return self.FS.integrate(self, self.this)

    def copy(self, like=None): raise NotImplementedError

    def __add__(self, other): raise NotImplementedError

    def __sub__(self, other): raise NotImplementedError

    def __mul__(self, other): raise NotImplementedError

    def __truediv__(self, other): return self * (1 / other)

    def __iadd__(self, other): self.update(self + other); return self

    def __isub__(self, other): self.update(self - other); return self

    def __imul__(self, other): self.update(self * other); return self

    def __itruediv__(self, other): self.update(self * (1 / other)); return self

    def __neg__(self, other): raise NotImplementedError

    __radd__ = __add__
#     __rsub__ = __sub__
    __rmul__ = __mul__


class Array(Function):
    '''
    A subspace of functions which is parametrised by an array and the linear algebra of 
    functions is equivalent to the array algebra.   
    '''

    def __init__(self, arr, FS=None, this=None):
        if FS is None:
            FS = ArraySpace(arr.shape)
        super().__init__(FS, this)
        self.arr = arr

    def copy(self, like=None):
        if like is None:
            arr = self.arr.copy()  # deep copy
        elif np.isscalar(like):
            return self.FS.one(like, self.this)  # create a scalar vector
        elif like.size == self.arr.size:
            arr = like if like.shape == self.arr.shape else like.reshape(self.arr.shape)  # same space, different array
        else:
            return self.FS.element(like)
        return self._copy(arr, self.FS, self.this)

    def _copy(self, arr, FS, this): return self.FS._element(arr, FS, this)

    @property
    def size(self): return self.arr.size

    @property
    def shape(self): return self.arr.shape

    def asarray(self): return self.arr

    def ravel(self): return self.arr.ravel() if self.arr.ndim > 1 else self.arr

    def update(self, arr=None, FS=None):
        if hasattr(arr, 'FS'):
            assert FS is None or arr.FS == FS
            arr, FS = arr.arr, arr.FS
        if FS is None:
            if arr is not None:
                self.arr = arr.reshape(self.arr.shape)
        elif arr is None:
            FS.update(self)
        else:
            self.arr, self.FS, self.this = arr, FS, FS.this

    def __add__(self, other):
        if hasattr(other, 'FS'):
            assert self.FS == other.FS
            other = other.arr
        # try this and let numpy generate any errors
        return self._copy(self.arr + other, self.FS, self.this)

    def __sub__(self, other):
        if hasattr(other, 'FS'):
            assert self.FS == other.FS
            other = other.arr
        # try this and let numpy generate any errors
        return self._copy(self.arr - other, self.FS, self.this)

    def __mul__(self, other):
        assert np.isscalar(other)
        return self._copy(self.arr * other, self.FS, self.this)

    def __truediv__(self, other):
        assert np.isscalar(other)
        return self._copy(self.arr / other, self.FS, self.this)

    def __neg__(self): return self._copy(-self.arr, self.FS, self.this)

    __radd__ = __add__
#     __rsub__ = __sub__
    __rmul__ = __mul__


class Discrete(Array):

    def __init__(self, arr, FS=None, this=None):
        if FS is None:
            FS = DiscreteSpace(x0=[0] * arr.ndim, dx=[1] * arr.ndim, n=arr.shape)

        super().__init__(arr, FS, this)

    def plot(self, ax=None, update=None, origin='lower', aspect='auto', N=10, cmap=None, **kwargs):
        ax = plt.gca() if ax is None else ax
        FS = self.FS
        if FS.regular_mesh or FS.dim == 1:
            if FS.dim == 1:
                if update is None:
                    return ax.plot(FS.x, self.arr, **kwargs)[0]
                else:
                    update.set_ydata(self.arr)
                    return update
            elif FS.dim == 2:
                if update is None:
                    extent = sum(([FS.mesh[0][i], FS.mesh[0][i] + FS.mesh[1][i] * FS.mesh[2][i]]
                                  for i in range(2)), [])
                    return ax.imshow(self.arr.T, cmap=cmap, extent=extent, origin=origin, aspect=aspect, **kwargs)
                else:
                    update.set_data(self.arr.T)
                    return update
            else:
                raise NotImplementedError
        else:
            if FS.dim == 2:
                if update is None:
                    if not hasattr(FS, '_tri') or FS._tri is None:
                        from matplotlib import tri
                        FS._tri = tri.Triangulation(FS.x[:, 0], FS.x[:, 1])
                else:
                    for coll in update.collections:
                        ax.collections.remove(coll)
                return ax.tricontourf(FS._tri, self.arr, N, cmap=cmap, **kwargs)

#                 from scipy.interpolate import interp2d as interp
#                 if grid is None:
#                     if not hasattr(FS, '_grid') or FS._grid is None:
#                         extent = FS.x.min(0), FS.x.max(0)
#                         extent = ((extent[0][0], extent[1][0]), (extent[0][1], extent[1][1]))
#                         FS._grid = tuple(np.linspace(*extent[i], 100) for i in range(2))
#                     grid = FS._grid
#                 z = interp(FS.x[:, 0], FS.x[:, 1], self.arr, kind='linear',
#                            copy=False, bounds_error=False)
#                 z = z(*grid)
#
#                 if update is None:
#                     extent = grid[0][0], grid[0][-1], grid[1][0], grid[1][-1]
#                     return ax.imshow(z, *args, extent=extent, origin=origin, **kwargs)
#                 else:
#                     update.set_data(z)
#                     return update
            else:
                raise NotImplementedError

        return ax


class IndexedSpace(ArraySpace):
    '''
    h = f + g 
        if f.ind[i] = g.ind[j] = h.ind[k]
            h.arr[k] = f.arr[i] + g.arr[j]
    f.arr.shape = shape, f.ind.shape = (shape[0],)
    '''

    def __init__(self, shape):
        shape = (shape,) if np.isscalar(shape) else tuple(shape)
        assert len(shape) == 2

        self._shape = shape
        self._ind = np.arange(shape[0], dtype=INT)
        self._maxindex = shape[0]
        self._element = IndexedArray

    @property
    def this(self): return (self._shape, self._ind)

    @this.setter
    def this(self, value):
        if hasattr(value, 'ind'):
            self._shape, self._ind = value.shape, value.ind
        else:
            self._shape, self._ind = value
        assert self._shape[0] == self._ind.shape[0]

    @property
    def shape(self): return self._shape

    @shape.setter
    def shape(self, value): raise ValueError('Length of array can only be updated by updating the index vector')

    @property
    def ind(self): return self._ind

    @ind.setter
    def ind(self, value):
        self._ind = value
        self._shape = (value.shape[0], self._shape[1])

    def update(self, f, inplace=True):
        assert f.shape[1] == self._shape[1]
        if self._ind is not f.this[1]:
            newarr = np.empty(self._shape, dtype=FLOAT)
            _old2newind(f.arr, f.this[1], newarr, self._ind)

            if inplace:
                f.arr, f.this = newarr, self.this
            else:
                f = self._element(newarr, self, self.this)
        return f

    def element(self, arr, this=None):
        this = self.this if this is None else this
        if hasattr(arr, 'this'):
            if this != arr.this:
                self.update(arr, inplace=True)
            return arr
        else:
            arr = arr.asarray() if hasattr(arr, 'asarray') else arr
            arr = arr.reshape(-1, this[0][1])
            return self._element(arr, self, this)

    def align(self, f, g):
        assert f.FS == g.FS
        if f.this == g.this:
            return f, g
        else:
            return self.update(f, False), self.update(g, False)

    def reindex(self, n=1, ind=None):
        if ind is not None:
            # replace old index list with the given one
            self.ind = np.require(ind, INT, ['C', 'A', 'O'])
            if len(ind) > 0:
                self._maxindex = max(self._maxindex, ind[-1])
        else:
            # extend old index list by n
            self.ind = np.concatenate((self._ind, np.arange(self._maxindex, self._maxindex + n)))
            self._maxindex += n

    def extend(self, n=1): self.reindex(n)


@jit(**__params)
def _old2newind(old, oldind, new, newind):
    i, j, K = 0, 0, old.shape[1]
    while i < old.shape[0] and j < new.shape[0]:
        I, J = oldind[i], newind[j]
        if I < J:
            i += 1  # old coordinate has been deleted in new index
        elif I == J:
            for k in range(K):
                new[j, k] = old[i, k]
            i += 1; j += 1
        else:
            for k in range(K):
                new[j, k] = 0  # new coordinate initialised
            j += 1


class IndexedArray(Array):

    def __init__(self, arr, FS, this=None):
        self.arr, self.FS = arr, FS
        self.this = FS.this if this is None else this
        N = self.arr.shape[0]
        if N < self.this[0][0]:
            this = list(self.this)
            this[0] = (N,) + this[0][1:]
            this[1] = this[1][:N]
            self.this = tuple(this)
        elif N > self.this[0][0]:
            raise ValueError

    def __add__(self, other):
        if hasattr(other, 'FS'):
            f, g = self.FS.align(self, other)
            return f._copy(f.arr + g.arr, f.FS, f.this)
        else:
            # try this and let numpy generate any errors
            return self._copy(self.arr + other, self.FS, self.this)

    def __sub__(self, other):
        if hasattr(other, 'FS'):
            f, g = self.FS.align(self, other)
            return f._copy(f.arr - g.arr, f.FS, f.this)
        else:
            # try this and let numpy generate any errors
            return self._copy(self.arr - other, self.FS, self.this)

    __radd__ = __add__

    def __len__(self): return self.arr.shape[0]
    def __iter__(self):
        self._index = -1
        return self
    def __next__(self):
        self._index += 1
        if self._index == self.FS.size:
            raise StopIteration
        else:
            return self.arr[self._index]  # TODO: should be an Array?

    def __getitem__(self, indx): return self.arr[indx]  # TODO: should be an Array?
