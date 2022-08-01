'''
Created on 14 Jul 2021

@author: Rob Tovey
'''
import numpy as np
from .DynamicPathSelect import midTimePath, bestPath, regularBestPath
from .optimisation_algs import FWFactory1, FWFactory2


def UBrandomDynamicFW(fidelity, OT, nAtoms=1, masses=1, meshsize=1, **kwargs):
    assert not OT.balanced

    def get_paths(rho, F, bounds):
        if meshsize < nAtoms:
            paths = np.random.rand(nAtoms, F.T, 3)  # random locations and masses
            for i in range(3):
                paths[..., i] = bounds[0][i] + paths[..., i] * (bounds[1][i] - bounds[0][i])
        else:
            mesh = np.random.rand(F.T, meshsize, 2)
            for i in range(2):
                mesh[..., i] = bounds[0][1 + i] + mesh[..., i] * (bounds[1][1 + i] - bounds[0][1 + i])
            H = np.random.rand(F.T, masses) if masses > 0 else np.ones((F.T, 1))
            H = bounds[0][0] + H * (bounds[1][0] - bounds[0][0])
            # paths = bestPath(F(x=mesh), mesh, OT.kernel[2], rho.T, masses=H, nAtoms=nAtoms)
            paths = midTimePath(bestPath, F(x=mesh), mesh,
                                OT.kernel[2], rho.T, masses=H, nAtoms=nAtoms)
        return paths.reshape(nAtoms, -1)

    return FWFactory2(*FWFactory1(fidelity, OT, get_paths, **kwargs))


def UBuniformDynamicFW(fidelity, OT, nAtoms=1, masses=1, levels=1, maxlevel=7, **kwargs):
    assert not OT.balanced
    pms = {'levels': levels, 'maxlevel': maxlevel, 'stepped': False, 'level_its':0}

    def get_paths(rho, F, bounds):
        mesh = np.meshgrid(*[np.linspace(bounds[0][i], bounds[1][i], 2 ** pms['levels'] + 1)
                             for i in range(1, 3)], indexing='ij')
        mesh = np.concatenate([m[:,:, None] for m in mesh], axis=2)
        H = np.linspace(bounds[0][0], bounds[1][0], masses + 1) if masses > 0 else np.array([1], dtype='float64')
        # return regularBestPath(F(x=mesh.reshape(-1, 2)), mesh, OT.kernel[2], rho.T, masses=H, nAtoms=nAtoms, vel=OT.velocity)
        return midTimePath(regularBestPath, F(x=mesh.reshape(-1, 2)), mesh, OT.kernel[2], rho.T, masses=H, nAtoms=nAtoms, vel=OT.velocity)

    return FWFactory2(*FWFactory1(fidelity, OT, get_paths, pms=pms, **kwargs))


def UBuniformRadiusDynamicFW(fidelity, OT, nAtoms=1, masses=1, levels=1, maxvel=None, **kwargs):
    assert not OT.balanced
    N = 2 ** levels
    maxvel = OT.velocity if maxvel is None else maxvel
    pms = {'levels': 2, 'maxlevel': np.inf, 'stepped': False, 'level_its':0}
    # starting with levels=2 allows maximum velocity of roughly 2^2 pixels per frame

    def get_paths(rho, F, bounds):
        # velocity of 1 pixel per time interval:
        pxps = max(bounds[1][i] - bounds[0][i] for i in range(1, 3)) / (N * (OT.T[1] - OT.T[0]))
        vel = min(maxvel, pxps * 2 ** pms['levels'])
        pms['maxlevel'] = np.ceil(np.log2(min(maxvel / pxps, N)))
        # a stupid check if the initial level parameter is too large
        pms['levels'] = min(pms['levels'], pms['maxlevel'])

        mesh = np.meshgrid(*[np.linspace(bounds[0][i], bounds[1][i], N + 1)
                             for i in range(1, 3)], indexing='ij')
        mesh = np.concatenate([m[:,:, None] for m in mesh], axis=2)
        H = np.linspace(bounds[0][0], bounds[1][0], masses + 1) if masses > 0 else np.array([1], dtype='float64')
        # return regularBestPath(F(x=mesh.reshape(-1, 2)), mesh, OT.kernel[2], rho.T, masses=H, nAtoms=nAtoms, vel=vel)
        return midTimePath(regularBestPath, F(x=mesh.reshape(-1, 2)), mesh, OT.kernel[2], rho.T, masses=H, nAtoms=nAtoms, vel=vel)

    return FWFactory2(*FWFactory1(fidelity, OT, get_paths, pms=pms, **kwargs))


if __name__ == '__main__':
    from .utils import MovieMaker
    from .dynamicModelsBin import example, save
    problem, RECORD, ALGS = 2, False, (1,)

    fidelity, OT, GroundTruth = example(problem, balanced=False)
    dim = 2

    for alg in ALGS:
        for nAtoms in [1]:
            if alg == 1:
                for meshsize in [(10, 10)]:
                    M = meshsize[0] ** 2, meshsize[1]
                    iters = (100 if problem == 2 else 10)
                    record = '2D_UB%d_%d_%d_%d' % ((problem, nAtoms) + meshsize) if RECORD else None
                    mv = MovieMaker(filename=record, fps=5, dummy=record is None)
                    np.random.seed(100)
                    recon, cback = UBrandomDynamicFW(fidelity, OT, constraint=10, GT=GroundTruth,
                                                     nAtoms=nAtoms, meshsize=M[0], masses=M[1], iters=iters, mv=mv)

                    if RECORD:
                        save(record, DM=recon, CB=cback)
                    print(GroundTruth.norm(), recon.norm())
                    print('\nFinal energies', OT(GroundTruth), cback.E[:, 1].min(), '\n')

            elif alg == 2:
                iters = (100 if problem == 2 else 10)
                record = '2D_uniformUB%d_%d' % (problem, nAtoms) if RECORD else None
                mv = MovieMaker(filename=record, fps=min(5, int(np.ceil(iters / 10))), dummy=record is None)
                np.random.seed(100)
                recon, cback = UBuniformDynamicFW(fidelity, OT, masses=10, constraint=10, GT=GroundTruth,
                                                  nAtoms=nAtoms, levels=3, maxlevel=7, iters=iters, mv=mv, doPlot=True)

                if RECORD:
                    save(record, DM=recon, CB=cback)
                print('\nFinal energies', OT(GroundTruth), cback.E[:, 1].min(), '\n')

#     from matplotlib.pyplot import show; show()

    print('\n\nfinished')
