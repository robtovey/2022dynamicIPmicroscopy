'''
Created on 2 Dec 2020

@author: rtovey
'''
import numpy as np
from scipy.optimize import minimize, Bounds
from .utils import jit, __pparams, callback, plotter as plotter_default, ZERO, CURVE_TOL
from .dynamicModelsBin import DynamicMeasureSpace, CurveSpace
from .DynamicPathSelect import bestPath


def SLF(x0, add_atom, local_opt, cback=None):
    if cback is None:
        cback = callback()

    xi = x0.copy()
    cback(0, xi, xi)
    for i in range(cback.iters):
        # Add new atom
        xip1 = add_atom(xi)
        # Locally optimise over full parameter space
        xip2 = local_opt(xip1)
        # Log convergence properties
        if cback(i + 1, xip2, xi):
            break
        else:
            xi = xip2

    return xip2, cback


def black_box(g, u, r, B, C, iters=1000, verbose=False):
    out = minimize(g, u, r, method='L-BFGS-B', options={'maxcor': 30, 'maxiter': iters},
                   jac=True, bounds=B, constraints=C, tol=1e-16, callback=None)
#     print('Message', out.message)
    # if out.message.startswith(b'ABNORMAL'):
    #     print('Local descent warning:', out.message)
    #     u0 = out.x
    #     np.random.seed(1)
    #     du = np.random.rand(u0.size)
    #     E0, G0 = g(u0, r)
    #     G0 = G0.dot(du)
    #     print('Gradient', '% .3e' % G0)
    #     print(('%13s ' * 7) % ('E0 = E(x)', 'Et = E(x+ty)', 'err0 = Et-E0', 'err0/t', 'err1=err0-tdE', 'err1/t', 'err1/t^2'))
    #     for i in range(0, 10):
    #         t = 10 ** (-i)
    #         E1 = g(u0 + t * du, r)[0]
    #         print(('% .3e    ' * 7) % (E0, E1, E1 - E0, (E1 - E0) / t, (E1 - E0 - t * G0),
    #                                 (E1 - E0 - t * G0) / t, (E1 - E0 - t * G0) / t ** 2))
    #     exit()

# #         print('Local descent warning:', out.message)
# #         raise
    if verbose and out.message.startswith(b'ABNORMAL'):
        print('Local descent warning:', out.message)

#     out = minimize(g, u, r, method='TNC', jac=True, bounds=B, constraints=C, tol=0, callback=None)
#     if verbose and out.message.startswith('Linear'):
#         print('Local descent warning:', out.message)

    try:
        g(out.x, r)  # hopefully set the correct minimiser in r
    except TypeError:
        g(out.x, *r)
    return out


def local_linear_oracle(rho, fidelity, OT, bounds, iters=1000):
    nAtoms = rho.shape[0]

    def grad_lin(u, rho):
        rho.x[:] = u.reshape(nAtoms, -1, 2)
        # Linearised data fidelity
        f, df = fidelity.grad(rho)  # linearised derivative on grid
        # OT term
        g, _, dx = OT.grad(rho)  # derivative of non-linearised transport
        dx += df[:, 1:].reshape(dx.shape)

        return f + g, dx.ravel()

    u = rho.x.ravel()
    Xsz = u.size // 2

    B = Bounds(bounds[0][1:] * Xsz, bounds[1][1:] * Xsz, keep_feasible=True)
    black_box(grad_lin, u, rho, B, None, iters=iters)

    # Vector of linear energies for each atom
    E = (np.array([OT(r) for r in rho]) + fidelity(x=rho.x.transpose((1, 0, 2))).sum(axis=0))

    ind = np.argsort(E)
    rho.arr = np.require(rho.arr[ind], requirements='C')

    return rho, E.sum(), E[ind]


def local_linear_oracle_UB(rho, fidelity, OT, bounds, iters=1000):  # unbalanced version
    nAtoms = rho.shape[0]

    def grad_lin(u, rho):
        rho.arr = u.reshape(rho.arr.shape)

        # Linearised data fidelity
        M, dM = OT.toarray(rho, grad=True)
        f, df = fidelity.grad(M)  # linearised derivative on grid
        if not (np.isscalar(dM) and (dM == 1)):
            df *= dM  # chain rule for coordinate transform
        # OT term
        g, darr = OT.grad(rho)  # derivative of non-linearised transport

        return f + g, (df + darr).ravel()

    u = rho.ravel()

    B = Bounds(bounds[0] * nAtoms, bounds[1] * nAtoms, keep_feasible=True)
    black_box(grad_lin, u, rho, B, None, iters=iters)

    # Vector of linear energies for each atom
    E = (np.array([OT(r) for r in rho]) +
         (fidelity(x=rho.x.transpose((1, 0, 2))) * OT.toarray(rho)[:,::3].T).sum(axis=0))

    ind = np.argsort(E)
    rho.arr = np.require(rho.arr[ind], requirements='C')

    return rho, E.sum(), E[ind]


def path_linesearch(rho0, rho1, constraint, energy):
    rho0.FS.extend(max(0, rho0.shape[0] + rho1.shape[0] - rho0.FS.shape[0]))
    start = np.concatenate((rho0.arr, rho1.arr), axis=0)
    start[rho0.shape[0]:, 0] = 0  # set mass of rho1 to 0

    nextrho = rho0.FS.element(start)
    step = np.zeros(start.shape[0])  # line-search direction

    def interp_measures(t):
        Rho = nextrho.copy()
        Rho.a[:, 0] += t * step
        return Rho

    def min_quadratic(F):
        'Find minimum of quadratic using Lagrange interpolation formula'
        f = [F(t / 2) for t in range(3)]
        r = 3 * f[0] - 4 * f[1] + f[2], f[0] - 2 * f[1] + f[2]
        if abs(r[1]) < .1 * (abs(r[0]) + 1e-5):
            return 0
        else:
            return (3 * f[0] - 4 * f[1] + f[2]) / (f[0] - 2 * f[1] + f[2]) / 4

#         # lineasearch all curves at once
#         step[:rho0.shape[0]], step[rho0.shape[0]:] = -rho0.a, constraint * rho1.a / (ZERO + rho1.a.sum())
#         nextrho = interp_measures(max(0, min(1, min_quadratic(lambda t: energy(interp_measures(t))))))

    # linesearch each curve individually
    for i in range(rho0.shape[0], step.size):
        step[:i], step[i] = -nextrho.a[:i, 0], constraint
        nextrho = interp_measures(max(0, min(1, min_quadratic(lambda t: energy(interp_measures(t))))))

    return nextrho


def path_linesearch_UB(rho0, rho1, constraint, energy, OT):
    rho0.FS.extend(max(0, rho0.shape[0] + rho1.shape[0] - rho0.FS.shape[0]))
    start = np.concatenate((OT.toarray(rho0), OT.toarray(rho1)), axis=0)

    nextrho = start.copy()
    nextrho[rho0.shape[0]:,::3] = 0  # set mass of rho1 to 0
    step = np.zeros(nextrho[:,::3].shape)  # line-search direction

    def interp_measures(t):
        Rho = nextrho.copy()
        Rho[:,::3] += t * step
        return Rho

    def min_quadratic():
        'Find minimum of quadratic using Lagrange interpolation formula'
        f = [energy(OT.fromarray(interp_measures(t / 2))) for t in range(3)]
        r = 3 * f[0] - 4 * f[1] + f[2], f[0] - 2 * f[1] + f[2]
        if abs(r[1]) < .1 * abs(r[0]):
            return np.sign(r[0] * r[1])  # it's going to be cut off in [0,1] anyway
        else:
            return r[0] / r[1] / 4

    # linesearch each curve individually
    for i in range(rho0.shape[0], step.shape[0]):
        step[:i], step[i] = -nextrho[:i,::3], constraint * start[i,::3]
        nextrho = interp_measures(max(0, min(1, min_quadratic())))

    return rho0.FS.element(OT.fromarray(nextrho))


def Energy(fidelity, OT):
    if hasattr(OT, 'toarray'):
        def energy(rho): return fidelity(OT.toarray(rho)) + OT(rho)
    else:
        def energy(rho): return fidelity(rho) + OT(rho)
    return energy


def Extras(pms, energy):
    def extras(X=None, x=None, d=None, **kwargs):
        keys = 'gap', 'E', 'E0', 'dof', 'step'
        if X is None:
            return keys
        elif pms['E'] < 0:
            # gap must be set in add_atom
            pms['E'] = pms['E0'] = energy(X)
        out = [pms[k] for k in keys[:-1]] + [d]

        if 'levels' in pms:
            if not pms['stepped'] and d <= ZERO:
                pms['levels'] += 1
                pms['level_its'] = 0
            pms['stepped'] = False
            if pms['levels'] > pms['maxlevel']:
                pms['stop'] = True

        pms['gap'] = -1
        return out
    return extras


def Curve_Grads(fidelity, OT):

    def grad(u, rho):  # returns function value and gradient of energy
        rho.arr = u.reshape(rho.shape)

        f, df = fidelity.grad(rho, discrete=True)  # full derivative on grid
        g, da, dx = OT.grad(rho)  # full derivative on grid

        df[:, 0] += da
        df[:, 1:] += dx.reshape(dx.shape[0], -1)
        return f + g, df.ravel()

    def grad_weights(u, rho):  # returns function value and gradient of energy
        raise NotImplementedError('Have not checked fidelity derivatives')

        rho.a[:, 0] = u

        f, df = fidelity.linearise(rho, energy=True)  # differentiate
        df = df(x=rho.x.transpose((1, 0, 2))).sum(axis=0)  # evaluate on grid
        g, da, _ = OT.grad(rho)  # full derivative on grid

        return f + g, df + da

    def grad_locs(u, rho):  # returns function value and gradient of energy
        raise NotImplementedError('Have not checked fidelity derivatives')

        rho.x[:] = u.reshape(rho.x.shape)

        f, df = fidelity.grad(rho, discrete=True)  # full derivative on grid
        g, _, dx = OT.grad(rho)  # full derivative on grid

        dx += df[:, 1:].reshape(dx.shape)

        return f + g, dx.ravel()

    return grad, grad_weights, grad_locs


def gap(E0, E1, constraint):
    #     return E0 - constraint * E1  # exact
    #     return E0- min(E0, constraint * E1)  # nonnegative
    return max(0, -E1 - 1e-10 * constraint)  # normalised gap, assuming sliding weights for E0


@jit(cache=True)
def _filtercurves(array, thresh):
    I = 1
    for j in range(1, array.shape[0]):  # first row is already correct
        row = array[j]
        if row[0] < thresh:
            continue
        for i in range(I):
            FLAG = True
            for ii in range(1, array.shape[1]):
                if abs(array[i, ii] - row[ii]) > thresh:
                    # row j is not the same location as row i
                    FLAG = False
                    break
            if FLAG:
                break
#             FLAG = False
        if FLAG:
            array[i, 0] += row[0]
        else:
            for ii in range(array.shape[1]):
                array[I, ii] = row[ii]
            I += 1
    return I


@jit(cache=True)
def _filtercurves_UB(array, thresh):
    T = array.shape[1]
    I = 1
    for j in range(1, array.shape[0]):  # first row is already correct
        for i in range(I):  # compare with previous rows
            FLAG = True
            for t in range(T):  # compare entries block-wise
                if ((abs(array[j, t, 1] - array[i, t, 1]) > thresh)
                        or (abs(array[j, t, 2] - array[i, t, 2]) > thresh)):
                    # row j at time t is not in the same location as row i at t
                    FLAG = False
                if not FLAG:
                    break  # don't check remaining times
            if FLAG:  # found matching row
                break
        if FLAG:
            for t in range(T):
                array[i, t, 0] += array[j, t, 0]  # sum the masses of curves
        else:
            array[I,:,:] = array[j,:,:]
            I += 1
    return I


def filtercurves(rho, thresh=CURVE_TOL):
    ind = (rho.a.max(1) > thresh)
    arr = rho.arr[ind]
    if arr.shape[0] == 0:
        N = 0
    elif rho.FS.balanced:
        N = _filtercurves(arr, thresh)
    else:
        N = _filtercurves_UB(arr.reshape(arr.shape[0], -1, 3), thresh)
    rho.arr = arr[:N].reshape(N, rho.shape[1])
    rho.FS.reindex(ind=rho.FS._ind[:N])
    rho.this = rho.FS.this
    return rho


def FWFactory1(fidelity, OT, get_paths, pms=None, atoms=None, bounds=None, constraint=1, opt='joint',
               opt_support=False, GT=None, mv=None, doPlot=True, iters=100, iters_blackbox=None, quiet=False, plotter=plotter_default, window=None):
    if atoms is None:
        if GT is None:
            atoms = DynamicMeasureSpace(CurveSpace(dim=2, T=OT.T, balanced=OT.balanced)).zero()
        else:
            atoms = DynamicMeasureSpace(GT.FS.FS).zero()
    else:
        assert atoms.FS.balanced == OT.balanced

    if bounds is None:
        # in order: amplitude, x, y
        bounds = [0, 0, 0], [constraint, 1, 1]
    if window is None:
        window = bounds[0][1:], bounds[1][1:]

    pms = {} if pms is None else pms
    pms.update({'gap':-1, 'E':-1, 'E0': 0, 'dof': 0, 'i': 1, 'maxiters':iters,
                'ax1': (np.inf, -np.inf), 'ax2': (np.inf, -np.inf), 'stop': False})

    energy = Energy(fidelity, OT)
    extras = Extras(pms, energy)
    doPlot, mv = plotter(GT, pms, mv, paths=True, doPlot=doPlot, window=window)

    cback = callback(iters=iters, frequency=(1.1 if iters / mv.fps > 5.1 else 1), quiet=quiet, record=('extras', 'stepsize'), extras=extras,
                     stop=lambda *_: pms['stop'], plot={'call': doPlot, 'fig': mv.fig})

    if OT.balanced:
        iters_blackbox = 1000 if iters_blackbox is None else iters_blackbox
        grad, grad_weights, grad_locs = Curve_Grads(fidelity, OT)

        def local_opt(rho):
            if rho.shape[0] == 0:
                return rho
            nextrho, C = rho.copy(), None

            if opt == 'joint':
                u = nextrho.ravel()  # weights plus locations
                Xsz = nextrho.x.size // 2
                B = Bounds([0] * nextrho.shape[0] + bounds[0][1:] * Xsz,
                           [bounds[1][0]] * nextrho.shape[0] + bounds[1][1:] * Xsz,
                           keep_feasible=True)
                out = black_box(grad, u, nextrho, B, C, iters=iters_blackbox)
                pms.update(E=grad(out.x, nextrho)[0], nfev=out.nfev, nit=out.nit)

            elif opt == 'weights':
                u = nextrho.a[:, 0]
                B = Bounds([0] * u.size, [bounds[1][0]] * u.size, keep_feasible=True)
                out = black_box(grad_weights, u, nextrho, B, C, iters=iters_blackbox)
                pms.update(E=grad_weights(out.x, nextrho)[0], nfev=out.nfev, nit=out.nit)

            elif opt == 'sequential':
                u = nextrho.x.ravel()
                Xsz = u.size // 2
                B = Bounds(bounds[0][1:] * Xsz, bounds[1][1:] * Xsz, keep_feasible=True)
                out = black_box(grad_locs, u, nextrho, B, C, iters=iters_blackbox)
                nfev, nit = out.nfev, out.nit
                u = nextrho.a
                B = Bounds([0] * u.size, [bounds[1][0]] * u.size, keep_feasible=True)
                out = black_box(grad_weights, u, nextrho, B, C, iters=iters_blackbox)
                pms.update(E=grad_weights(out.x, nextrho)[0], nfev=nfev + out.nfev, nit=nit + out.nit)

            filtercurves(nextrho)  # remove nul/duplicate curves
            pms['dof'] = nextrho.arr.size
            return nextrho

        def add_atom(rho):
            F = fidelity.linearise(rho)
            paths = get_paths(rho, F, bounds)  # find optimal atom assuming constraint=1
            nextrho = DynamicMeasureSpace(rho.FS.FS, n=paths.shape[0]).ones()
            nextrho.x[:] = paths

            nextrho, E0, E1 = local_linear_oracle(nextrho, F, OT, bounds, iters_blackbox)
            pms['gap'] = max(pms['gap'], gap(E0, E1.min(), constraint))

            nextrho = path_linesearch(rho, nextrho, constraint, energy)
            pms['E0'] = energy(nextrho)

            if 'levels' in pms:
                pms['level_its'] += 1
                # Extra check to avoid getting stuck:
                b = (pms['level_its'] > 100) and (pms['levels'] < pms['maxlevel'])
                if pms['gap'] < ZERO or b:
                    pms['levels'] += 1
                    pms['stepped'] = True
                    pms['level_its'] = 0
                    if pms['levels'] > pms['maxlevel']:
                        pms['stop'] = True

            pms['i'] += 1
            return nextrho
    else:
        iters_blackbox = 10000 if iters_blackbox is None else iters_blackbox
        assert opt == 'joint'

        def grad(u, rho):  # returns function value and gradient of energy
            rho.arr = u.reshape(rho.shape)

            M, dM = OT.toarray(rho, grad=True)
            f, df = fidelity.grad(M, discrete=True)  # full derivative on grid
            if not (np.isscalar(dM) and (dM == 1)):
                df *= dM  # chain rule for coordinate transform
            g, darr = OT.grad(rho)  # full derivative on grid

            return f + g, (df + darr).ravel()

        def local_opt(rho):
            if rho.shape[0] == 0:
                return rho
            nextrho, C = rho.copy(), None

            u = nextrho.ravel()  # weights plus locations
            rows = nextrho.shape[0]
            B = Bounds(bbounds[0] * rows, bbounds[1] * rows, keep_feasible=True)
            out = black_box(grad, u, nextrho, B, C, iters=iters_blackbox)
            pms.update(E=grad(out.x, nextrho)[0], nfev=out.nfev, nit=out.nit)

            filtercurves(nextrho)  # remove nul/duplicate curves
            pms['dof'] = nextrho.arr.size
            return nextrho

        bbounds = bounds[0] * fidelity.T, bounds[1] * fidelity.T  # full constraints for one atom

        def add_atom(rho):
            F = fidelity.linearise(OT.toarray(rho))
            paths = get_paths(rho, F, bounds)  # find optimal atom assuming constraint=1
            nextrho = DynamicMeasureSpace(rho.FS.FS, n=paths.shape[0]).ones()
            nextrho.arr = paths

            if opt_support and rho.shape[0] > 0:
                oldpaths = paths
                mesh = np.ascontiguousarray(rho.x.transpose(1, 0, 2))  # use the current support
                H = np.tile(np.linspace(0, 1, 10), (rho.T.size, 1))  # and hopefully a good mass-resolution
                # find optimal atom assuming constraint=1
                paths = bestPath(F(x=mesh), mesh, OT.kernel[2], rho.T, masses=H, nAtoms=paths.shape[0])
                nextrho = DynamicMeasureSpace(rho.FS.FS, n=2 * paths.shape[0]).ones()
                nextrho.arr = np.concatenate([oldpaths, paths], axis=0)

            nextrho, E0, E1 = local_linear_oracle_UB(nextrho, F, OT, bbounds, iters_blackbox)
            pms['gap'] = max(pms['gap'], gap(E0, E1.min(), constraint))

            nextrho = path_linesearch_UB(rho, nextrho, constraint, energy, OT)
            pms['E0'] = energy(nextrho)

            if 'levels' in pms:
                pms['level_its'] += 1
                # Extra check to avoid getting stuck:
                b = (pms['level_its'] > 100) and (pms['levels'] < pms['maxlevel'])
                if pms['gap'] < ZERO or b:
                    pms['levels'] += 1
                    pms['stepped'] = True
                    pms['level_its'] = 0
                    if pms['levels'] > pms['maxlevel']:
                        pms['stop'] = True

            pms['i'] += 1
            return nextrho

    return atoms, add_atom, local_opt, cback, mv


def FWFactory2(atoms, add_atom, local_opt, cback, mv):
    atoms = local_opt(atoms)
    add_atom(atoms.copy())  # pre-compile functions, hopefully quite quick
    atoms, cback = SLF(atoms, add_atom, local_opt, cback)
    mv.finish()
    return atoms, cback
