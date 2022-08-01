'''
Created on 14 Feb 2022

@author: Rob Tovey
'''
import sys, importlib
from .utils import MovieMaker
from .dynamicModelsBin import save, load, example, DynamicMeasureSpace, CurveSpace
from .OTPathUtils import pathOT, UBpathOT

__util__ = ['MovieMaker', 'save', 'load', 'example', 'DynamicMeasureSpace', 'CurveSpace', 'pathOT', 'UBpathOT']
__submod__ = ['balanced_FW', 'unbalanced_FW', 'dynamicModelsBin', 'DynamicPathSelect',
              'FunctionSpaces', 'optimisation_algs', 'OTPathUtils', 'utils']
__fidbin__ = ['WindowedFourierFidelity', 'FourierFidelity',
              'GaussianFidelity', 'GaussianGridFidelity', 'AiryDiscFidelity']
__all__ = sorted(__util__ + __submod__)


def __dir__(): return sorted(__all__ + __fidbin__)


def __getattr__(name):
    if name in __util__:
        # return importlib.import_module("." + name, __name__)
        return globals()[name]
    elif name in __submod__:
        return importlib.import_module("." + name, __name__)
    elif name in __fidbin__:
        mod = importlib.import_module(f".fidelityBin.{name[:-8]}", __name__)
        return mod.__builtins__['getattr'](mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if sys.version_info.major == 3 and sys.version_info.minor < 7:  # need PEP 562 for lazy loading
    G = globals()
    for name in __submod__ + __fidbin__:
        G[name] = __getattr__(name)


def randomDynamicFW(fidelity, OT, **kwargs):
    '''
    Frank-Wolfe algorithm for dynamic problems using random locations

    Approximates minimisers of the variational problem:  
        min_u E(u) = fidelity(u) + OT(u)
    where u is a measure on paths, the fidelity is smooth, and OT is a penalty on paths.

    The only parameters different between random locations and uniform grids are
        random : meshsize
        uniform : levels, maxlevel

    Parameters
    ----------
    fidelity : instance of fidelity.Fidelity object
        Represents smooth data term
    OT : instance of OTPathUtils.pathOT object
        Represents a cost on paths
    nAtoms : int (default 1)
        Number of curves to add in each iteration
    meshsize : int (default 1)
        First integer refers to number of random locations sampled for each time-point. 
        If meshsize<nAtoms, then trajectories are purely random. Otherwise, the best paths
        through those random locations will be returned.
        If the problem is unbalanced, then the second integer is the number of random masses.
    masses : int (default 1)
        Only used for unbalanced transport. Indicates number of random masses to consider.
        If masses=0, then all curves will be initialised with constant mass 1
        If masses>0, then that number of random masses are optimised over.
    atoms : instance of dynamicModelsBin.DynamicMeasure object (default 0)
        The starting point of the algorithm
    constraint : float (default 1)
        The mass constraint for each curve in the reconstruction
    bounds : ([m,a,b], [M,A,B]) all floats (default [0,0,0], [constraint, 1, 1])
        The mass of each curve is constrained in the range [m,M]
        The x (respectively y) coordinates are constrained between [a,A] (resp. [b,B])
        Dimensions other than 2 are currently untested.
    opt : str (default 'joint')
        For balanced examples opt dictates the type of sliding step. 'joint' optimises both
        locations and weights, 'weights' only updates the weights, and 'sequential' optimises
        first the locations then the weights.
    GT : instance of dynamicModelsBin.DynamicMeasure object (default None)
        If provided, the ground-truth will be included as an overlay in plots
    mv : instance of utils.MovieMaker object (default None)
        Object which allows figures to be recorded as a video
    doPlot : bool (default True)
        Turns off plotting if doPlot=False (even if mv is given).
        Default is to plot (whether or not mv is given)
    iters : int (default 100)
        Number of Frank-Wolfe iterations to perform 
    quiet : bool (default False)
        Flag for whether to print the approximate primal-dual gap to console


    Returns
    -------
    atoms : instance of dynamicModelsBin.DynamicMeasure object
        The final reconstruction
    E : instance of utils.callback object
        A container for tracking energy, gap, stepsize etc. with respect to iteration number
        and time

    '''
    if OT.balanced:
        from . import balanced_FW
        f = balanced_FW.randomDynamicFW
    else:
        from . import unbalanced_FW
        f = unbalanced_FW.UBrandomDynamicFW
    return f(fidelity, OT, **kwargs)


def uniformDynamicFW(fidelity, OT, **kwargs):
    '''
    Frank-Wolfe algorithm for dynamic problems using uniform grids

    Approximates minimisers of the variational problem:  
        min_u E(u) = fidelity(u) + OT(u)
    where u is a measure on paths, the fidelity is smooth, and OT is a penalty on paths.

    The only parameters different between uniform grids and random locations are
        random : meshsize
        uniform : levels, maxlevel

    Parameters
    ----------
    fidelity : instance of fidelity.Fidelity object
        Represents smooth data term
    OT : instance of OTPathUtils.pathOT object
        Represents a cost on paths
    nAtoms : int (default 1)
        Number of curves to add in each iteration
    levels : int (default 1)
        Discretisation starts with a mesh of size (2**levels+1)**d in dimension d
    maxlevel : int (default 7)
        Number of levels will increase when the algorithm fails to find a new curve on the 
        discrete mesh. The maximum mesh considered is of size (2**maxlevel+1)**d.
    masses : int (default 1)
        Only used for unbalanced transport. Indicates number of equi-spaced masses to consider.
        If the bounds on mass are [0,constraint], then it is the number of *non-zero* masses. 
        If masses=0, then all curves will be initialised with constant mass 1
        If masses>0, then np.linspace(m,M,masses+1) are optimised over
    atoms : instance of dynamicModelsBin.DynamicMeasure object (default 0)
        The starting point of the algorithm
    constraint : float (default 1)
        The mass constraint for each curve in the reconstruction
    bounds : ([m,a,b], [M,A,B]) all floats (default [0,0,0], [constraint, 1, 1])
        The mass of each curve is constrained in the range [m,M]
        The x (respectively y) coordinates are constrained between [a,A] (resp. [b,B])
        Dimensions other than 2 are currently untested.
    opt : str (default 'joint')
        For balanced examples opt dictates the type of sliding step. 'joint' optimises both
        locations and weights, 'weights' only updates the weights, and 'sequential' optimises
        first the locations then the weights.
    GT : instance of dynamicModelsBin.DynamicMeasure object (default None)
        If provided, the ground-truth will be included as an overlay in plots
    mv : instance of utils.MovieMaker object (default None)
        Object which allows figures to be recorded as a video
    doPlot : bool (default True)
        Turns off plotting if doPlot=False (even if mv is given).
        Default is to plot (whether or not mv is given)
    iters : int (default 100)
        Number of Frank-Wolfe iterations to perform 
    quiet : bool (default False)
        Flag for whether to print the approximate primal-dual gap to console


    Returns
    -------
    atoms : instance of dynamicModelsBin.DynamicMeasure object
        The final reconstruction
    E : instance of utils.callback object
        A container for tracking energy, gap, stepsize etc. with respect to iteration number
        and time
    '''
    if OT.balanced:
        from . import balanced_FW
        f = balanced_FW.uniformDynamicFW
    else:
        from . import unbalanced_FW
        f = unbalanced_FW.UBuniformDynamicFW
    return f(fidelity, OT, **kwargs)


def uniformRadiusDynamicFW(fidelity, OT, **kwargs):
    if OT.balanced:
        from . import balanced_FW
        f = balanced_FW.uniformRadiusDynamicFW
    else:
        from . import unbalanced_FW
        f = unbalanced_FW.UBuniformRadiusDynamicFW
    return f(fidelity, OT, **kwargs)
