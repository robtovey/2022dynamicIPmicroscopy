'''
Created on 12 Apr 2021

@author: Rob Tovey
'''
import numpy as np
from .DynamicPathSelect import bestPath, regularBestPath
from .optimisation_algs import FWFactory1, FWFactory2


def randomDynamicFW(fidelity, OT, nAtoms=1, meshsize=1, **kwargs):
    assert OT.balanced

    def get_paths(rho, F, bounds):
        if meshsize < nAtoms:
            paths = np.random.rand(nAtoms, F.T, 2)  # random locations
            for i in range(2):
                paths[..., i] = bounds[0][1 + i] + paths[..., i] * (bounds[1][1 + i] - bounds[0][1 + i])
        else:
            mesh = np.random.rand(F.T, meshsize, 2)
            for i in range(2):
                mesh[..., i] = bounds[0][1 + i] + mesh[..., i] * (bounds[1][1 + i] - bounds[0][1 + i])
            # paths = bestPath(F(x=mesh), mesh, OT.kernel[2], rho.T, nAtoms=nAtoms)
            paths = midTimePath(bestPath, F(x=mesh), mesh, OT.kernel[2], rho.T, nAtoms=nAtoms)
        return paths

    return FWFactory2(*FWFactory1(fidelity, OT, get_paths, **kwargs))


def uniformDynamicFW(fidelity, OT, nAtoms=1, levels=1, maxlevel=7, **kwargs):
    assert OT.balanced
    pms = {'levels': levels, 'maxlevel': maxlevel, 'stepped': False}

    def get_paths(rho, F, bounds):
        mesh = np.meshgrid(*[np.linspace(bounds[0][i], bounds[1][i], 2 ** pms['levels'] + 1)
                             for i in range(1, 3)], indexing='ij')
        mesh = np.concatenate([m[:, :, None] for m in mesh], axis=2)
        return regularBestPath(F(x=mesh.reshape(-1, 2)), mesh, OT.kernel[2], rho.T, nAtoms=nAtoms, vel=OT.velocity)

    return FWFactory2(*FWFactory1(fidelity, OT, get_paths, pms=pms, **kwargs))


def uniformRadiusDynamicFW(fidelity, OT, nAtoms=1, levels=1, maxvel=None, **kwargs):
    assert OT.balanced
    N = 2**levels
    maxvel = OT.velocity if maxvel is None else maxvel
    # This is a bit of a hack, 'levels' argument refers to resolution but
    # 'levels' parameter is the radius of search.
    pms = {'levels': 2, 'maxlevel': np.inf, 'stepped': False}

    def get_paths(rho, F, bounds):
        # velocity of 1 pixel per time interval:
        pxps = max(bounds[1][i] - bounds[0][i] for i in range(1, 3)) / (N * (OT.T[1] - OT.T[0]))
        vel = min(maxvel, pxps * 2**pms['levels'])
        pms['maxlevel'] = np.ceil(np.log2(min(maxvel / pxps, N)))
        # a stupid check if the initial level parameter is too large
        pms['levels'] = min(pms['levels'], pms['maxlevel'])

        mesh = np.meshgrid(*[np.linspace(bounds[0][i], bounds[1][i], N + 1)
                             for i in range(1, 3)], indexing='ij')
        mesh = np.concatenate([m[:, :, None] for m in mesh], axis=2)
        return regularBestPath(F(x=mesh.reshape(-1, 2)), mesh, OT.kernel[2], rho.T, nAtoms=nAtoms, vel=vel)

    return FWFactory2(*FWFactory1(fidelity, OT, get_paths, pms=pms, **kwargs))


if __name__ == '__main__':
    from .utils import MovieMaker
    from .dynamicModelsBin import example
    np.random.seed(100)

    # minimial energy problem 2 is 0.4236619774839504
    # minimial energy problem 3 is 0.9773918544127473
    problem, RECORD, ALGS, NOISE = 2, False, (2,), False

    fidelity, OT, GroundTruth = example(problem)
    dim = 2
    iters = (100 if problem == 2 else 10)

    for alg in ALGS:
        for nAtoms in [1]:
            for meshsize in [15]:
                if NOISE and alg == 1:
                    for p, params in enumerate(([1, .2], [1, .6], [3, .6])):
                        fidelity, OT, GroundTruth = example(problem, noise=params[1])
                        OT.scale_kernel(params[0])

                        record = '2D_paths%d_%d_%d_%d' % (problem, p, nAtoms, meshsize) if RECORD else None
                        mv = MovieMaker(filename=record, fps=4, dummy=record is None)
                        recon, cback = randomDynamicFW(fidelity, OT, constraint=10, GT=GroundTruth,
                                                       nAtoms=nAtoms, meshsize=meshsize ** 2, iters=iters, mv=mv)

                        print('\nFinal energies', OT(GroundTruth), cback.E[:, 1].min(), '\n')

                elif alg == 1:
                    fidelity, OT, GroundTruth = example(problem)
                    record = '2D_paths%d_%d_%d' % (problem, nAtoms, meshsize) if RECORD else None
                    mv = MovieMaker(filename=record, fps=4, dummy=record is None)
                    recon, cback = randomDynamicFW(fidelity, OT, nAtoms=nAtoms, meshsize=meshsize ** 2,
                                                   constraint=10, GT=GroundTruth, iters=iters, mv=mv)

                    print('\nFinal energies', OT(GroundTruth), cback.E[:, 1].min(), '\n')

            if NOISE and alg == 2:
                for p, params in enumerate(([1, .2], [1, .6], [3, .6])):
                    fidelity, OT, GroundTruth = example(problem, noise=params[1])
                    OT.scale_kernel(params[0])

                    record = '2D_uniformpaths%d_%d_%d' % (problem, p, nAtoms) if RECORD else None
                    mv = MovieMaker(filename=record, fps=4, dummy=record is None)
                    recon, cback = uniformDynamicFW(fidelity, OT, constraint=10, GT=GroundTruth,
                                                    nAtoms=nAtoms, levels=4, maxlevel=8, iters=iters, mv=mv)
                    print('\nFinal energies', p, nAtoms, cback.E[:, 1].min(), '\n')
            elif alg == 2:
                fidelity, OT, GroundTruth = example(problem)

                record = '2D_uniformpaths%d_%d' % (problem, nAtoms) if RECORD else None
                mv = MovieMaker(filename=record, fps=4, dummy=record is None)
                recon, cback = uniformDynamicFW(fidelity, OT, constraint=10, GT=GroundTruth,
                                                nAtoms=nAtoms, levels=4, maxlevel=8, iters=iters, mv=mv)

                print('\nFinal energies', OT(GroundTruth), cback.E[:, 1].min(), '\n')

            if NOISE and alg == 3:
                for p, params in enumerate(([1, .2], [1, .6], [3, .6])):
                    fidelity, OT, GroundTruth = example(problem, noise=params[1])
                    OT.scale_kernel(params[0])

                    record = None
                    mv = MovieMaker(filename=record, fps=4, dummy=record is None)
                    recon, cback = uniformRadiusDynamicFW(fidelity, OT, constraint=10, GT=GroundTruth,
                                                          nAtoms=nAtoms, levels=8, maxvel=10, iters=iters, mv=mv)
                    print('\nFinal energies', p, nAtoms, cback.E[:, 1].min(), '\n')
            elif alg == 3:
                fidelity, OT, GroundTruth = example(problem)

                record = None
                mv = MovieMaker(filename=record, fps=4, dummy=record is None)
                recon, cback = uniformRadiusDynamicFW(fidelity, OT, constraint=10, GT=GroundTruth,
                                                      nAtoms=nAtoms, levels=8, maxvel=10, iters=iters, mv=mv)

                print('\nFinal energies', OT(GroundTruth), cback.E[:, 1].min(), '\n')

    from matplotlib import pyplot as plt; plt.show()
    print('\n\nfinished')
