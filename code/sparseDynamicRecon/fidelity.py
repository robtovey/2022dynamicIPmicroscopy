'''
Created on 14 Oct 2021

@author: Rob
'''
__util__ = ['utils', 'Fidelity', 'QuadraticFidelity']
__fidelities__ = ['WindowedFourier', 'Fourier', 'Gaussian', 'GaussianGrid', 'AiryDisc']
__all__ = sorted(__util__ + __fidelities__)
import sys, importlib
from .fidelityBin import utils
from .fidelityBin.utils import Fidelity, QuadraticFidelity


def __dir__(): return __all__


def __getattr__(name):
    if name in __util__:
        return globals()[name]
    elif name in __fidelities__:
        mod = importlib.import_module(f".fidelityBin.{name}", __package__)
        return mod.__builtins__['getattr'](mod, name + 'Fidelity')
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if sys.version_info.major == 3 and sys.version_info.minor < 7:  # need PEP 562 for lazy loading
    G = globals()
    for name in __fidelities__:
        G[name] = __getattr__(name)
    del G


if __name__ == '__main__':
    import numpy as np
    from .dynamicModelsBin import DynamicMeasureSpace, CurveSpace
    np.random.seed(100)
    dim, T, epsilon, tau, datasz = 2, 11, 2, 1, 100
    padstr = lambda s, n: ' ' * ((n - len(s)) // 2) + s + ' ' * (n - len(s) - (n - len(s)) // 2)

    TEST = [1, 2, 3, 4, 5, 6]

    for test in TEST:
        for balanced in (True, False):
            FS = DynamicMeasureSpace(CurveSpace(dim=2, T=np.linspace(0, 1, T), balanced=balanced), n=10)

            rho = FS.rand()
            drho = FS.rand()
            if test == 1:
                name = 'Gaussian'
                F = Gaussian(np.random.rand(datasz, 2), 4, 1, np.random.rand(T, datasz))
            elif test == 2:
                name = 'Gaussian vec'
                F = Gaussian(np.random.rand(T, datasz, 2), 4, 1, np.random.rand(T, datasz))
            elif test == 3:
                name = 'Fourier'
                F = Fourier(np.random.rand(datasz, 2), np.random.rand(T, datasz))
                F = WindowedFourier(np.random.rand(datasz, 2), np.random.rand(T, datasz))
            elif test == 4:
                name = 'Fourier vec'
                F = Fourier(np.random.rand(T, datasz, 2), np.random.rand(T, datasz))
                F = WindowedFourier(np.random.rand(T, datasz, 2), np.random.rand(T, datasz))
            elif test == 5:
                name = 'AiryDisc'
                F = AiryDisc(np.random.rand(datasz, 2), 4, 1, np.random.rand(T, datasz))
            elif test == 6:
                name = 'AiryDisc vec'
                F = AiryDisc(np.random.rand(T, datasz, 2), 4, 1, np.random.rand(T, datasz))

            print('-' * 50, '\n%s Fidelity test' % name, '' if balanced else 'unbalanced')
            E0, G0 = F.grad(rho, discrete=True)
            E0, G0 = E0, (drho.arr * G0).sum()

            if abs(F(rho.arr + 1e-9 * drho.arr) - E0 - 1e-9 * G0) * 1e18 < 1e4:
                print('checked')
            else:
                print(''.join(padstr(s, 13)
                              for s in ('E0 = E(x)', 'Et = E(x+ty)', 'err0=Et-E0', 'err0/t', 'err1=err0-tdE', 'err1/t^2')))
                for i in range(0, 10):
                    t = 10 ** (-i)
                    E1 = F(rho.arr + t * drho.arr)
                    print(''.join(padstr('% .3e' % f, 13) for f in (E0, E1, E1 - E0, (E1 - E0) / t, (E1 - E0 - t * G0),
                                                                    (E1 - E0 - t * G0) / t ** 2)))

            print('-' * 50, '\n%s Linearisation test' % name, '' if balanced else 'unbalanced')
            F = F.linearise(rho)
            E0, G0 = F.grad(rho)
            E0, G0 = E0, (drho.arr * G0).sum()
            if abs(F(rho.arr + 1e-9 * drho.arr) - E0 - 1e-9 * G0) * 1e18 < 1e4:
                print('checked')
            else:
                print(''.join(padstr(s, 13)
                              for s in ('E0 = E(x)', 'Et = E(x+ty)', 'err0=Et-E0', 'err0/t', 'err1=err0-tdE', 'err1/t^2')))
                for i in range(0, 10):
                    t = 10 ** (-i)
                    E1 = F(rho.arr + t * drho.arr)
                    print(''.join(padstr('% .3e' % f, 13) for f in (E0, E1, E1 - E0, (E1 - E0) / t, (E1 - E0 - t * G0),
                                                                    (E1 - E0 - t * G0) / t ** 2)))

            print()
