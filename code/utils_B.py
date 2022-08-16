import numpy as np
from numba import jit, prange
from sparseDynamicRecon.fidelity import GaussianGrid
from sparseDynamicRecon import (DynamicMeasureSpace, CurveSpace, pathOT,
                                save, load, MovieMaker, uniformDynamicFW, uniformRadiusDynamicFW)
from utils import get_plotter, get_files

np.random.seed(123)
##################################################
############## code for kernels ##################
##################################################
# Balanced transport example:
# In the code below sigma is an array of shape (n. atoms, 1+2*T), representing a measure on
# balanced paths. The values of each row are (mass,x[0],y[0],...,x[T],y[T]). The path energy that
# we are minimising is:
#     energy(sigma) = sum_{i=1}^N ( alpha + sum_{j=1}^T beta[j]*((x[i,j]-x[i,j-1])^2+(y[i,j]-y[i,j-1])^2 ) * mass[i].
# The function grad computes d(energy)/d(sigma). The final utility is the 'step function' which
# gives the alternative formula
#     energy(sigma) = sum_{i=1}^N sum_{j=1}^T step(r[i,j],r[i,j-1],t[j]-t[j-1]) * mass[i]
# In this case
#     step(r1,r2,dt) = alpha * dt + (beta/(2*dt)) * |r1-r2|^2
_Tsz = 10
_alpha = 1.0; _beta = _gamma = np.ones(_Tsz, dtype='f8')
_weight = np.ones(3, dtype='f8')


@jit('void(f8[:,:],f8[:])', parallel=True, nopython=True)
def energy(sigma, out):
    '''
    The return variable is <out> given by
        out[i] = energy(sigma[i])
    where 
        energy(mass, x0, y0, ... , xT, yT) = mass * ( alpha + sum_{j=1}^T beta[j-1] * ((xj-x[j-1])**2 + (yj-y[j-1])**2) )  
    '''
    for i in prange(sigma.shape[0]):  # for each curve in sigma
        s = _alpha
        for j in range(_Tsz):
            J = 2 * j
            x, y = sigma[i, J + 1], sigma[i, J + 2]
            X, Y = sigma[i, J + 3], sigma[i, J + 4]

            s += _beta[j] * ((x - X) ** 2 + (y - Y) ** 2)
        out[i] = sigma[i, 0] * s


@jit('void(f8[:,:],f8[:],f8[:,:])', parallel=True, nopython=True)
def grad(sigma, out, outgrad):
    '''
    The return variables <out> and <outgrad> are given by
        out[i] = energy(sigma[i])
        outgrad[i,j] = d(out[i])/d(sigma[i,j])
    where 
        energy(mass, x0, y0, ... , xT, yT) = mass * ( alpha + sum_{j=1}^T beta[j-1] * ((xj-x[j-1])**2 + (yj-y[j-1])**2) )  
    '''
    for i in prange(sigma.shape[0]):
        s = _alpha
        J = 0
        for j in range(_Tsz):
            for k in range(2):
                tmp = sigma[i, 1 + J] - sigma[i, 3 + J]  # = dx
                s += _beta[j] * tmp ** 2  # add to energy

                tmp *= 2 * _beta[j]  # rescale for gradient
                outgrad[i, J] += tmp
                outgrad[i, 2 + J] -= tmp

                J += 1  # J = 2*j + k
        out[i] = s  # not scaled by mass of curve!


@jit('void(f8[:,:],f8[:,:],f8,f8[:,:])', parallel=True, nopython=True)
def step_mesh(x, X, dt, out):
    '''
    With alpha = weight[0]*dt and beta = weight[1]/(2*dt), the return variable <out> is given by
        out[i,j] = step(x[i],X[j],dt)
    where
        step(x,X,dt) = alpha + beta * ( (x[0]-X[0])**2 + (x[1]-X[1])**2 ) 
    '''
    alpha = _weight[0] * dt
    beta = _weight[1] / (2 * dt)
    for i in prange(x.shape[0]):
        xx, yy = x[i, 0], x[i, 1]
        for j in range(X.shape[0]):
            out[i, j] = alpha + beta * ((X[j, 0] - xx) ** 2 + (X[j, 1] - yy) ** 2)


def get_kernels(weight, T):
    '''
    Utility for recompiling energy, grad, and step_mesh with correct weights/times.
    '''
    DT = T[1:] - T[:-1]
    alpha, beta = weight[0] * DT.sum(), weight[1] / (2 * DT)
    Tsz = DT.size

    for thing in ('Tsz', 'alpha', 'beta', 'weight'):
        globals()['_' + thing] = locals()[thing]
    for thing in (energy, grad, step_mesh):
        thing.recompile()

    return (energy, grad, step_mesh)


class myOT(pathOT):
    '''
    This class bundles the previous functions into an 'optimal transport object', it shouldn't need changing. 
    '''
    def update_kernel(self, **kwargs):
        # This function is called whenever the OT parameters are modified
        self._ker_params = {k: kwargs.get(k, v) for k, v in self._ker_params.items()}
        # sigma is stored as an array of the form:
        #    sigma[i,:] = [h(0), x(0), y(0), ..., x(1), y(1)]
        self._shape = -1, 1 + 2 * self.T.size

        # Setting a maximum velocity for particles can reduce computation time:
        self.velocity = self._ker_params['velocity']

        self.kernel = get_kernels(np.array(self._ker_params['weight']).astype('float32'),
                                  self._ker_params['T'])


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
    directory: string
        results and videos will be saved in ./results/<directory>
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
    
    fname, recon, cback = single_run(data, T, weight, iters, LOAD=True, fig=None)
    
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
            iters=10.0: Run 10 iterations of the standard algorithm
            iters=10.1: In the insertion step, search for new curves with 10 non-zero intensities 
                        instead of binary curves (for unbalanced alg. only)
            iters=10.2: In the insertion step, start by searching for slowest moving curves rather
                        than those on a coarse grid. I think this is slightly faster than the 
                        default but should have no effect on the final reconstruction.
            iters=10.9: In each insertion step, 10000 new curves are added rather than the default of 100
    
    '''
    base_weight = np.array(weight, dtype=float)
    radius = FWHM * dx / (4 * np.log(2)) ** .5  # the function exp(-|x|^2/(2*radius^2)) = 0.5 at x = FWHM*dx/2

    def single_run(data, T, weight, iters, LOAD=True, doPlot=False, fig=None):
        fname = get_files(directory, data, weight, iters, None, FWHM)

        iters = float(iters)
        recon = None
        if LOAD:
            try:
                recon, cback = load(fname['recon'])
            except:
                pass
        if recon is None:
            space = DynamicMeasureSpace(CurveSpace(dim=2, T=T, balanced=True))
            F = GaussianGrid((dx, dx), radius, 0, data=data, ZERO=1e-10)
            OT = myOT(weight=(base_weight * np.array(weight, dtype=float)
                              ).astype('float32'), velocity=max_velocity, T=space.T)
            plotter = (get_plotter(data, T, F, threshold) if doPlot else
                       (lambda *_, **__: ((lambda *_: None), MovieMaker(dummy=True))))
            kwargs = dict(# default parameters
                bounds=([0, 0, 0], [1, dx * (data.shape[1] - 1), dx * (data.shape[2] - 1)]),
                iters=int(iters), iters_blackbox=1000,  # number of iterations
                nAtoms=100,  # number of new atoms to add at every iteration
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
                    raise  # all reconstructions should be balanced!
                elif np.isclose(iters % 1, .3):
                    kwargs['masses'] = 0
                    raise  # all reconstructions should be balanced!
                elif np.isclose(iters % 1, .9):
                    kwargs['nAtoms'] = 10000
                recon, cback = uniformDynamicFW(F, OT, **kwargs,
                                                levels=4,  # start with a grid size 2^4x2^4
                                                maxlevel=int(np.ceil(np.log2(data.shape[1]))),
                                                )
            # # Don't record videos automatically:
            # F.gif(fname['gif'], OT.toarray(recon), scale=4 * 255)
            # F.gif(fname['trace'], recon.copy(OT.toarray(recon)), scale=4 * 255, trace=5)

        return fname, recon, cback

    return single_run


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run and save a single reconstruction')
    parser.add_argument('datafile', type=str,
                        help='File containing raw data')
    parser.add_argument('weight', type=str, nargs=2,
                        help='Weights in OT cost (alpha and beta)')
    parser.add_argument('iters', type=str,
                        help='Maximum number of iterations (and variation of algorithm in decimal part)')
    args = parser.parse_args()

    with np.load(args.datafile, allow_pickle=True) as f:
        data, T, kwargs = f['data'], f['T'], f['extras'].item(0)

    single_run = get_single_run(**kwargs)
    single_run(data, T, args.weight, args.iters, LOAD=True)
