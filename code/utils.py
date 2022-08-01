import numpy as np
from numba import jit, prange
from sparseDynamicRecon.fidelity import GaussianGrid
from sparseDynamicRecon.utils import time, timefrmt
from sparseDynamicRecon import (DynamicMeasureSpace, CurveSpace, UBpathOT, save, load, MovieMaker,
                                uniformDynamicFW, uniformRadiusDynamicFW)

np.random.seed(123)
##################################################
############## code for kernels ##################
##################################################
# Unbalanced transport example:
# In the code below sigma is an array of shape (n. atoms, 3*T), representing a measure on
# unbalanced paths. The values of each row are (h[0],x[0],y[0],...,h[T],x[T],y[T]) where the mass
# of a curve at time j is h[j]^p for some p>1. The path energy that we are minimising is:
#     energy(sigma) = sum_{i=1}^N sum_{j=1}^T (  alpha[j] * (h[i,j-1]^p + h[i,j]^p)/2
#                + beta[j] * ( (x[i,j]-x[i,j-1])^2 + (y[i,j]-y[i,j-1])^2 ) * (h[i,j-1]^p+h[i,j]^p)/2
#                + gamma[j] * |h[i,j-1]^p - h[i,j]|^p  ).
# The function grad computes d(energy)/d(sigma). The final utility is the 'step function' which
# gives the alternative formula
#     energy(sigma) = sum_{i=1}^N sum_{j=1}^T step(r[i,j],h[i,j]^p,r[i,j-1],h[i,j-1]^p,t[j]-t[j-1])
# In this case
#     step(r1,m1,r2,m2,dt) = (alpha * dt + (beta/dt) * |r1-r2|_p^p) * (m1 + m2)/2
#                            + gamma/(2*dt)*|m1^(1/p)-m2^(1/p)|^p
# Note that the first two functions are smooth functions of the h parameterisation whereas the
# third is a non-smooth function of mass = h^p.
_p = 2.0
_Tsz = 10
_alpha = _beta = _gamma = np.ones(_Tsz, dtype='f8')
_weight = np.ones(3, dtype='f8')


@jit('void(f8[:,:],f8[:])', parallel=True, nopython=True, cache=True)
def energy(sigma, out):
    '''
    The return variable is <out> given by
        out[i] = energy(sigma[i])
    where 
        energy(h0, x0, y0, ... , hT, xT, yT) = 
            sum_{j=1}^T ( alpha[j-1] + beta[j-1] * ((xj-x[j-1])**2 + (yj-y[j-1])**2) ) * (h[j-1]^p + h[j]^p)
                        + gamma[j-1] * abs(h[j-1] - h[j])^p
    '''
    for i in prange(sigma.shape[0]):  # for each curve in sigma
        s = 0
        for j in range(_Tsz):
            J = 3 * j
            h, x, y = sigma[i, J], sigma[i, J + 1], sigma[i, J + 2]
            H, X, Y = sigma[i, J + 3], sigma[i, J + 4], sigma[i, J + 5]

            s += (_alpha[j] + _beta[j] * ((x - X) ** 2 + (y - Y) ** 2)) * (h ** _p + H ** _p)
            s += _gamma[j] * abs(h - H) ** _p
        out[i] = s


@jit('void(f8[:,:],f8[:],f8[:,:])', parallel=True, nopython=True, cache=True)
def grad(sigma, out, outgrad):
    '''
    The return variables <out> and <outgrad> are given by
        out[i] = energy(sigma[i])
        outgrad[i,j] = d(out[i])/d(sigma[i,j])
    where 
        energy(h0, x0, y0, ... , hT, xT, yT) = 
            sum_{j=1}^T ( alpha[j-1] + beta[j-1] * ((xj-x[j-1])**2 + (yj-y[j-1])**2) ) * (h[j-1]^p + h[j]^p)
                        + gamma[j-1] * abs(h[j-1] - h[j])^p
    '''
    for i in prange(sigma.shape[0]):
        s = 0.0
        for j in range(_Tsz):
            J = 3 * j
            h, H = sigma[i, J], sigma[i, J + 3]
            dx = sigma[i, J + 1] - sigma[i, J + 4]
            dy = sigma[i, J + 2] - sigma[i, J + 5]

            tmp = _alpha[j] + _beta[j] * (dx ** 2 + dy ** 2)
            hH = (h ** _p + H ** _p)

            s += tmp * hH
            s += _gamma[j] * abs(h - H) ** _p

            signed = _p * _gamma[j] * abs(h - H) ** (_p - 1)
            if H > h:
                signed = -signed
            outgrad[i, J + 0] += _p * tmp * h ** (_p - 1) + signed
            outgrad[i, J + 3] += _p * tmp * H ** (_p - 1) - signed

            tmp = 2 * _beta[j] * hH
            outgrad[i, J + 1] += tmp * dx
            outgrad[i, J + 2] += tmp * dy
            outgrad[i, J + 4] -= tmp * dx
            outgrad[i, J + 5] -= tmp * dy
        out[i] = s


@jit('void(f8[:,:],f8[:],f8[:,:],f8[:],f8,f8[:,:,:,:])', parallel=True, nopython=True, cache=True)
def step_mesh(x, m, X, M, dt, out):
    '''
    With alpha = weight[0]*dt/2, beta = weight[1]/(2*dt), and gamma = weight[2]/(2*dt^(p-1)), the 
    return variable <out> is given by
        out[i0,i1,j0,j1] = step(x[i0],m[i1],X[j0],M[j1],dt)
    where
        step(x,m,X,M,dt) = (alpha + beta * ( (x[0]-X[0])**2 + (x[1]-X[1])**2 )) * (m+M)
                           + gamma * (m^(1/p)-M^(1/p))
    '''
    alpha = _weight[0] * .5 * dt
    beta = _weight[1] / (2 * dt)
    gamma = _weight[2] / (2 * dt ** (_p - 1))
    for i0 in prange(x.shape[0]):
        xx, yy = x[i0, 0], x[i0, 1]
        for i1 in range(m.shape[0]):
            h = m[i1] ** (1 / _p)
            for j0 in range(X.shape[0]):
                r = (xx - X[j0, 0]) ** 2 + (yy - X[j0, 1]) ** 2
                for j1 in range(M.shape[0]):
                    H = M[j1] ** (1 / _p)
                    out[i0, i1, j0, j1] = ((alpha + beta * r) * (m[i1] + M[j1])
                                           +gamma * abs(h - H) ** _p)


def get_kernels(weight, T, p):
    '''
    Utility for recompiling energy, grad, and step_mesh with correct weights/times/p value.
    '''
    DT = T[1:] - T[:-1]
    alpha, beta, gamma = weight[0] * .5 * DT, weight[1] / (2 * DT), weight[2] / (2 * DT ** (p - 1))
    Tsz = DT.size

    for thing in ('p', 'Tsz', 'alpha', 'beta', 'gamma', 'weight'):
        globals()['_' + thing] = locals()[thing]
    for thing in (energy, grad, step_mesh):
        thing.recompile()

    return (energy, grad, step_mesh)


def toarray(sigma, grad, p):
    '''
    Mapping from smooth to mass representation, i.e.
        sigma[i] = (h[0], x[0], y[0],..., h[T], x[T], y[T])
    and the returned measure is
        rho[i] = (h[0]^p, x[0], y[0], ..., h[T]^p, x[T], y[T])
    The other optional return value is the derivative:
        d(rho)[i] = (p*h[0]^(p-1), x[0], y[0], ..., p*h[T]^(p-1), x[T], y[T])
    
    rho, (drho) = toarray(sigma, grad, p)
    Arguments:
    ----------
        sigma: numpy array of shape (N atoms, 3*T)
            The measure to be converted
        grad: bool
            If true, returns (rho, drho), otherwise returns just rho
        p: float
            Value of p to be used
    '''
    M = sigma.copy()
    M[:,::3] = sigma[:,::3] ** p
    if grad:
        dM = np.ones(sigma.shape)
        dM[:,::3] = p * sigma[:,::3] ** (p - 1)
        return M, dM
    else:
        return M


def fromarray(rho, p):
    '''
    Inverse of toarray, i.e.
        rho[i] = (m[0], x[0], y[0],..., m[T], x[T], y[T])
    and the returned measure is
        sigma[i] = (m[0]^(1/p), x[0], y[0], ..., m[T]^(1/p), x[T], y[T])
    
    sigma = fromarray(rho, p)
    Arguments:
    ----------
        rho: numpy array of shape (N atoms, 3*T)
            The measure to be converted
        p: float
            Value of p to be used
    '''
    sigma = rho.copy()
    sigma[:,::3] = np.maximum(0, sigma[:,::3]) ** (1 / p)  # the max-threshold shouldn't be needed but for rounding errors
    return sigma


class myOT(UBpathOT):
    '''
    This class bundles the previous functions into an 'optimal transport object', it shouldn't 
    need changing. 
    '''
    def toarray(self, rho, grad=False): return toarray(self._parse_rho(rho), grad, _p)

    def fromarray(self, M): return fromarray(M, _p)

    def update_kernel(self, **kwargs):
        # This function is called whenever the OT parameters are modified
        self._ker_params = {k: kwargs.get(k, v) for k, v in self._ker_params.items()}
        # sigma is stored as an array of the form:
        #    sigma[i,:] = [h(0), x(0), y(0), ..., h(1), x(1), y(1)]
        self._shape = -1, 3 * self.T.size

        # Setting a maximum velocity for particles can reduce computation time:
        self.velocity = self._ker_params['velocity']

        self.kernel = get_kernels(np.array(self._ker_params['weight']).astype('float32'),
                                  self._ker_params['T'], self._ker_params['norm'])

##################################################
############## code for plotting #################
##################################################


def get_plotter(data, T, F, threshold):
    '''
    This function returns another which is able to plot the reconstruction and energy while the algorithm is running.
    '''
    def plotter(GT, pms, mv, paths=False, doPlot=True, window=None):
        if not doPlot:  # don't do any plotting
            return (lambda *_: None), (MovieMaker(dummy=True) if mv is None else mv)
        elif mv is None or mv.doplot == 0:  # plot but don't save
            mv = MovieMaker(dummy=False)
        else:  # plotting and saving, clear figure first
            mv.fig.clear()
        if window is None:  # spatial window to plot
            window = [0, 0], [1, 1]
        xlim = window[0][1], window[1][1]
        ylim = window[0][0], window[1][0]

        tic = time()
        midT = T.size // 2
        extent = [window[0][1], window[1][1], window[0][0], window[1][0]]
        title_spec = {'fontsize': 20}
        from matplotlib.cm import ScalarMappable
        arr2alpha = lambda arr: np.maximum(0, np.minimum(1, (10 / (1 - .75 * threshold)) * (arr - .75 * threshold)))

        def doPlot(cback, i, rho, fig, plt):
            if pms.get('axes', None) is None:
                fig.clear()
                pms['axes'] = fig.subplots(2, 3)
                for ax in pms['axes'].ravel()[:-1]:
                    ax.set_xlim(xlim); ax.set_ylim(ylim)

            recon_data = F.fwrd(rho)

            ax = pms['axes'][0, 0]
            ax.set_title('Recon slice t=%f' % T[midT], title_spec)
            if 'Recon slice' not in pms:
                pms['Recon slice'] = ax.imshow(recon_data[midT].T, vmin=0, vmax=.5 * data[midT].max(), origin='lower',
                                               extent=extent, cmap='gray_r')
            else:
                pms['Recon slice'].set_data(recon_data[midT].T)

            ax = pms['axes'][1, 0]
            ax.set_title('Data slice t=%f' % T[midT], title_spec)
            if 'Data slice' not in pms:
                pms['Data slice'] = ax.imshow(data[midT].T, vmin=0, vmax=.5 * data[midT].max(), origin='lower',
                                              extent=extent, cmap='gray_r')

            ax = pms['axes'][0, 1]
            ax.set_title('Recon time-map', title_spec)
            if 'Recon time-map' not in pms:
                pms['Recon time-map'] = ax.imshow(recon_data.argmax(0).T, alpha=arr2alpha(recon_data.max(0).T),
                                                  vmin=0, vmax=T.size, origin='lower', extent=extent, cmap='viridis')
            else:
                pms['Recon time-map'].set_alpha(arr2alpha(recon_data.max(0).T))
                pms['Recon time-map'].set_data(recon_data.argmax(0).T)

            ax = pms['axes'][1, 1]
            ax.set_title('Data time-map', title_spec)
            if 'Data time-map' not in pms:
                pms['Data time-map'] = ax.imshow(data.argmax(0).T, alpha=arr2alpha(data.max(0).T),
                                                 vmin=0, vmax=T.size, origin='lower', extent=extent, cmap='viridis')

            ax = pms['axes'][0, 2]
            ax.set_title('Recon paths', title_spec)
            if 'Recon paths' not in pms:
                pms['Recon paths'] = rho.plot(ax=ax, paths=True, decimals=4, minwidth=3,
                                              width=6, alpha=1, cmap='viridis')
                cbar = fig.colorbar(ScalarMappable(cmap=pms['Recon paths'].cmap,
                                                   norm=pms['Recon paths'].norm), ax=ax, pad=0.01)
                cbar.ax.set_ylabel('t', rotation=0)
            else:
                pms['Recon paths'] = rho.plot(ax=ax, decimals=4, minwidth=3, width=6,
                                              alpha=1, update=pms['Recon paths'])

            ax = pms['axes'][1, 2]
            if 'energy plot' not in pms:
                pms['energy plot'] = ax.plot([1], [1])[0]
                ax.set_yscale('log'); ax.set_title('Energy convergence', title_spec)
                ax.set_xlabel('iterations'); ax.set_ylabel(r'$E(\rho_n) - \min_N(E(\rho_N))$')

            else:
                x, y = cback.i[:cback._i], cback.E[:cback._i, 1]
                if len(y) == 2:
                    Y = np.sort(y)
                    pms['ax2'] = Y  # = (y.min(), y.max())
                else:
                    Y = np.sort(np.partition(y, 2)[:2])
                ax.set_title('Energy convergence, %.5f' % pms['E'], title_spec)
                if Y[1] > 2 * Y[0]:
                    Y = .5 * Y[0], .5 * Y[0]
                else:
                    Y = 1.5 * Y[0] - .5 * Y[1], .5 * (Y[1] - Y[0])
                pms['energy plot'].set_data(1 + x, y - Y[0])
                pms['ax2'] = min(pms['ax2'][0], max(1e-8, Y[1]) if y.size > 1 else Y), max(pms['ax2'][1], y[-1])
                ax.set_xlim(1, x[-1] + .5); ax.set_ylim(.95 * pms['ax2'][0], 1.05 * pms['ax2'][1])
                ax.set_title('Energy after %s is %e' % (timefrmt(time() - tic), pms['E']), title_spec)

            if i < 2:
                fig.tight_layout()
            mv.update()
        return doPlot, mv
    return plotter


##################################################
################ code for filenames ##############
##################################################
def get_files(directory, data, weight, iters, p, FWHM):
    directory = (directory,) if type(directory) is str else tuple(directory)
    fname = ('-' + '-'.join(str(d) for d in data.shape + tuple(weight))
             +'-' + '-'.join(str(d) for d in (iters, p, FWHM) if d is not None)
             ).replace('.', '_')  # reformat decimals, e.g. 1.2 becomes 1_2
    return  {'recon': directory + ('recon' + fname + '.npz',),
             'vid': directory + ('vid' + fname + '.mp4',),
             'gif': directory + ('gif' + fname + '.gif',),
             'trace': directory + ('trace' + fname + '.gif',)}


def reweight_files(weight0, weight1, directory, data, iters, p, FWHM):
    directory = (directory,) if type(directory) is str else tuple(directory)
    w = lambda i, weight: tuple(w[i[j]] for j, w in enumerate(weight))
    files = []

    from itertools import product
    from os import rename, path
    for i in product(*[range(len(w)) for w in weight0]):
        for I in iters:
            fname0 = get_files(directory, data, w(i, weight0), I, p, FWHM)
            fname1 = get_files(directory, data, w(i, weight1), I, p, FWHM)

            for k in fname0:
                if path.exists(path.join(*fname0[k])):
                    files.append(path.join(*fname1[k]) + '123')  # so that we don't overwrite old files with new ones first
                    rename(path.join(*fname0[k]), files[-1])
    for file in files:
        rename(file, file[:-3])


##################################################
############## code for reconstruction ###########
##################################################
def get_single_run(directory, weight, dx, FWHM, max_velocity, threshold=0):
    '''
    Encodes basic properties of the data and parameters into a function for performing 
    reconstructions.
    
    single_run = get_single_run(directory, weight, dx, FWHM, max_velocity, threshold)
    
    Arguments:
    ----------
    directory: string or tuple of strings
        results and videos will be saved in <directory> or os.path.join(<directory>)
    weight: two floats
        The returned function single_run will also have a <weight> parameter, denote these values
        by <weight2>. The parameters used by the algorithm are
            alpha = weight[0] * weight2[0]
            beta  = weight[1] * weight2[1] / (2*dt)
        The values of <weight2> will appear in output filenames, so should be close to 1 in 
        magnitude. The values of <weight> are only used in this scaling.
    dx: float
        The grid-spacing beween pixels  
    FWHM: float
        Full-width-half-maximum of the Gaussian point-spread function in units of pixels
    max_velocity: float or np.inf
        Constrain search for new curves to jumps smaller than (max_velocity*dt/dx) pixels
    threshold: float (optional, default=0)
        Parameter used for get_plotter, visualisation ignores all signal below the threshold
    
    Returns single_run.    
    
    fname, recon, cback = single_run(data, T, weight, iters, p, LOAD=True, doPlot=False, fig=None)
    
    Arguments:
    ----------
    data: float type numpy array with shape (NT,Nx,Ny)
        Preprocessed data for reconstruction, first axis is time
    T: float type numpy array with shape (NT,)
        Time points corresponding to data
    weight: two floats
        Fine-tuning of alpha and beta weights as seen previously.
    iters: int or float
        int(iters) is the number of iterations of the algorithm which will be performed.
        The decimal part of <iters> is a code indicating variation of algorithm (this is very 
        lazy...). For example:
            iters=12.0: Run 12 iterations of the standard algorithm
            iters=12.1: In the insertion step, search for new curves with 10 non-zero intensities 
                        instead of binary curves (for unbalanced alg. only)
            iters=12.2: In the insertion step, start by searching for slowest moving curves rather
                        than those on a coarse grid. I think this is slightly faster than the 
                        default but should have no effect on the final reconstruction.
            iters=12.2: In the insertion step, start by searching for slowest moving curves rather
                        than those on a coarse grid. I think this is slightly faster than the 
                        default but should have no effect on the final reconstruction.
            iters=12.3: In the insertion step, only search for balanced curves rather than binary intensities
            iters=12.9: In each insertion step, 10000 new curves are added rather than the default of 100
    p: float
        Value of p to use in energy
    LOAD: bool, default is True
        If True, then algorithm will first try to load the reconstruction rather than re-computing it
    doPlot: bool, default is False
        If True, then the get_plotter function is used to record a video demonstrating the 
        convergence during the process of the algorithm.
    fig: matplotlib figure or (the default) None
        Figure to be used for recording reconstruction during the running of the algorithm
        
    Returns:
    --------
    fname: dictionary of strings
        Filenames for the reconstruction and various videos: 
            recon: The saved parameters of the reconstruction and convergence properties
            vid: The video recorded during reconstruction (if applicable)
            gif: The de-noised data presented as a gif
            trace: The original data overlayed with the tracking result
    recon: instance of sparseDynamicRecon.dynamicModelsBin.DynamicMeasure
        The measure on paths representation of the reconstruction. Some useful attributes:
            recon.arr = [ [h[0,0],x[0,0],y[0,0],...], [h[1,0],x[1,0],y[1,0],...],  ] is an array of
                the trajectories with masses and positions
            recon.x = [ [x[0,0], y[0,0], ...], [x[1,0], y[1,0], ...], ... ]
            recon.a = [ [h[0,0], h[0,1], ...], [h[1,0], h[1,1], ...], ... ]
            recon.T = [t[0], t[1], ... ]
            recon.shape = recon.arr.shape = (N atoms, 3*recon.T.size)
    cback: instance of sparseDynamicRecon.utils.callback
        Effectively a store of convergence properties of the algorithm at different numbers 
        throughout the reconstruction process. Some useful attributes:
            cback.i = [i0, i1, ...] iteration numbers at which observations where made
            cback.T = [t0, t1, ...] time in seconds spent in algorithm up to iteration i
            c.extras() = ('gap', 'E', 'E0', 'dof', 'step') are the labels of quantities recorded:
                gap: estimated primal-dual gap, non-negative indicator of quality of reconstruction
                E: the energy after iteration i
                E0: the energy after the insertion step of iteration i (before the sliding step)
                dof: total number of parameters in recon.arr at the end of iteration i, equal to
                     (N atoms)* 3*T
                step: the L1 vector norm between recon.arr between iteration i-1 and i, typically 
                      proportional to the gap
            c.E = 2D numpy array of shape (len(cback.i), len(c.extras())). The array of recorded
                values at iterations indicated in cback.i, ordered in columns as in c.extras().
    
    '''
    base_weight = np.array(weight, dtype=float)
    radius = FWHM * dx / (4 * np.log(2)) ** .5  # the function exp(-|x|^2/(2*radius^2)) = 0.5 at x = FWHM*dx/2

    def single_run(data, T, weight, iters, p, LOAD=True, doPlot=False, fig=None):
        fname = get_files(directory, data, weight, iters, p, FWHM)

        iters, p = float(iters), float(p)  # for command-line arguments
        recon = None
        if LOAD:
            try:
                recon, cback = load(fname['recon'])
            except:
                pass

        if recon is None:
            space = DynamicMeasureSpace(CurveSpace(dim=2, T=T, balanced=False))
            F = GaussianGrid((dx, dx), radius, 0, data=data, ZERO=1e-10)
            OT = myOT(weight=(base_weight * np.array(weight, dtype=float)).astype('float32'),
                      norm=p, velocity=max_velocity, T=space.T)
            plotter = (get_plotter(data, T, F, threshold) if doPlot else
                       (lambda *_, **__: ((lambda *_: None), MovieMaker(dummy=True))))
            kwargs = dict(# default parameters
                bounds=([0, 0, 0], [1, dx * (data.shape[1] - 1), dx * (data.shape[2] - 1)]),
                iters=int(iters), iters_blackbox=1000,  # number of iterations
                nAtoms=100,  # number of new atoms to add at every iteration
                masses=1,  # default number of non-zero values considered for initialisation
                # plotting parameters:
                window=([0, 0], [dx * data.shape[1], dx * data.shape[2]]),
                plotter=plotter,
                mv=MovieMaker(filename=fname['vid'], fps=4, fig=fig) if doPlot else None,
                quiet=True,
                opt_support=False,
            )

            if np.isclose(iters % 1, .2):
                recon, cback = uniformRadiusDynamicFW(F, OT, **kwargs,
                                                      levels=int(np.ceil(np.log2(data.shape[1]))),
                                                      maxvel=max_velocity,
                                                      )
            else:
                if np.isclose(iters % 1, .1):
                    kwargs['masses'] = 10
                elif np.isclose(iters % 1, .3):
                    kwargs['masses'] = 0
                elif np.isclose(iters % 1, .9):
                    kwargs['nAtoms'] = 10000
                recon, cback = uniformDynamicFW(F, OT, **kwargs,
                                                levels=4,  # start with a grid size 2^4x2^4
                                                maxlevel=int(np.ceil(np.log2(data.shape[1]))),
                                                )
            save(fname['recon'], DM=recon, CB=cback)
            F.gif(fname['gif'], OT.toarray(recon), scale=4 * 255)
            F.gif(fname['trace'], recon.copy(OT.toarray(recon)), scale=4 * 255, trace=5)
        recon.arr = toarray(recon.arr, False, p)  # always save 'smooth' parametrisation but return 'real' one

        return fname, recon, cback

    return single_run


# python utils.py Human_small.npz 1 1 1 3 2
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run and save a single reconstruction')
    parser.add_argument('datafile', type=str,
                        help='File containing raw data')
    parser.add_argument('weight', type=str, nargs=3,
                        help='Weights in OT cost (alpha, beta, and gamma)')
    parser.add_argument('iters', type=str,
                        help='Maximum number of iterations (and variation of algorithm in decimal part)')
    parser.add_argument('p', type=str,
                        help='Power of regularity in mass conservation')
    args = parser.parse_args()

    with np.load(args.datafile, allow_pickle=True) as f:
        data, T, kwargs = f['data'], f['T'], f['extras'].item(0)

    single_run = get_single_run(**kwargs)
    single_run(data, T, args.weight, args.iters, args.p, LOAD=True)
