'''
Created on 22 Mar 2021

@author: Rob Tovey
'''
import warnings
import numpy as np
from .utils import jit, prange, __params, __pparams


def midTimeSelector(L):
    # return 0
    # return L.shape[0] - 1
    return np.random.choice(L.shape[0])


def midTimePath(func, lin, mesh, kernel, T, *args, nAtoms=1, **kwargs):
    '''
    func = bestPath or regularBestPath
    t = int between 0 and T.size-1

    returns best <nAtoms> paths with distinct values at time T[t]
    '''
    regular = (mesh.shape[0] != lin.shape[0]) or (func == regularBestPath)
    t = midTimeSelector(lin)

    if t == 0:
        paths, E = func(lin, mesh, kernel, T, *args, nAtoms=-1, **kwargs)
    elif t == T.size - 1:
        paths, E = func(np.ascontiguousarray(lin[::-1]),  # reverse time
                        mesh if regular else np.ascontiguousarray(mesh[::-1]),
                        kernel, T.max() - T[::-1],  # preserve time intervals but not sign
                        * args, nAtoms=-1, **kwargs)
        paths = paths[:,::-1]  # re-invert time
    else:
        P1, E1 = func(lin[:t + 1],  # up to and including time t
                      mesh if regular else mesh[:t + 1],
                      kernel, T[:t + 1],
                      *args, nAtoms=-1, **kwargs)
        P2, E2 = func(np.ascontiguousarray(lin[:t - 1:-1]),  # reverse time up to including t
                      mesh if regular else np.ascontiguousarray(mesh[:t - 1:-1]),
                      kernel, T.max() - T[:t - 1:-1],  # preserve time intervals but flip
                      * args, nAtoms=-1, **kwargs)
        paths = np.concatenate((P1[:,:-1], P2[:,::-1]), axis=1)
        E = E1 + E2 - lin[t]  # remove double counting of weight at node t
    Eind = np.argsort(E)
    return paths[Eind[:nAtoms]]


def bestPath(lin, mesh, kernel, T, masses=None, nAtoms=1):
    '''
    lin is a list of 1D arrays of size n[k]
    mesh is a list of 2D arrays of shape (n[k], 2)
    kernel computes the cost of connecting two points
    masses is an array of permitted masses
    '''

    N = max(L.size for L in lin)
    TT = len(lin)
    assert N == min(L.size for L in lin)  # not fully implemented yet

    if masses is None:  # balanced transport
        E = [np.empty(N) for _ in range(2)]
        kernel_buf = np.empty((N, N))
        paths = [np.zeros((N, TT, 2)) for _ in range(2)]

        _bestPath(lin, mesh, kernel, T, *E, *paths, kernel_buf)
        E, paths = (E[1], paths[1]) if TT % 2 == 0 else (E[0], paths[0])

    else:  # unbalanced transport
        M = masses.shape[1]
        E = [np.empty((N, M)) for _ in range(2)]
        kernel_buf = np.empty((N, M, N, M))
        paths = [np.zeros((N, M, TT, 3)) for _ in range(2)]

        _bestPath_UB(lin, mesh, masses, kernel, T, *E, *paths, kernel_buf)
        E, paths = (E[1], paths[1]) if TT % 2 == 0 else (E[0], paths[0])
        ind = range(N), E.argmin(axis=1)
        E, paths = E[ind], paths[ind[0], ind[1]]  # only best final mass

    E, paths = E[:lin[-1].size], paths[:lin[-1].size]  # cut off excess buffer
    if nAtoms > 0:
        paths = paths[np.argsort(E)]  # sorted so the first row has smallest energy
        return paths[:nAtoms]
    else:
        return paths, E


@jit(**__pparams)
def __bestPath(L, M, C, oldE, oldP, newE, newP, t):
    N = M.shape[0]
    for i in prange(N):  # find best path going through M[i]
        bestE, bestJ = 10 ** 15, 0  # = infinity, 0
        for j in range(N):  # find j with minimal energy
            currentE = oldE[j] + C[i, j]
            if currentE < bestE:
                bestE, bestJ = currentE, j

        # copy over new curve
        newP[i,:t] = oldP[bestJ,:t]
        newP[i, t] = M[i]
        newE[i] = bestE + L[i]


@jit(**__params)
def _bestPath(lin, mesh, kernel, T, E0, E1, path0, path1, ker_buf):
    '''
    If p is a path 
        E(p) = sum_t lin[t][p[t]] + kernel(p[t],p[t+1])
    '''
    path0[:, 0] = mesh[0]
    E0[:] = lin[0]

    isEven = False
    for t in range(1, len(lin)):
        if isEven:
            oldE, newE = E1, E0
            oldP, newP = path1, path0
        else:
            oldE, newE = E0, E1
            oldP, newP = path0, path1
        isEven = not isEven

        kernel(mesh[t], mesh[t - 1], abs(T[t] - T[t - 1]), ker_buf)  # compute cost to join each curve
        __bestPath(lin[t], mesh[t], ker_buf, oldE, oldP, newE, newP, t)


@jit(**__pparams)
def __bestPath_UB(L, M, H, C, oldE, oldP, newE, newP, t):
    N = M.shape[0]
    for i0 in prange(N):  # find best path going through M[i0]
        for m0 in prange(H.size):  # find best curve through (H[m0], M[i0])
            bestE, bestI, bestM = 10 ** 15, 0, 0  # = infinity, 0, 0
            for i1 in range(N):  # find i1 with minimal energy
                for m1 in range(H.size):
                    currentE = oldE[i1, m1] + C[i0, m0, i1, m1]
                    if currentE < bestE:
                        bestE, bestI, bestM = currentE, i1, m1

            # copy over new curve
            newP[i0, m0,:t] = oldP[bestI, bestM,:t]
            newP[i0, m0, t, 0] = H[m0]
            newP[i0, m0, t, 1:] = M[i0]
            newE[i0, m0] = bestE + H[m0] * L[i0]


def _bestPath_UB(lin, mesh, H, kernel, T, E0, E1, path0, path1, ker_buf):
    '''
    If p = (h,x) is a path 
        E(p) = sum_t h[t]*lin[t][x[t]] + kernel(p[t], p[t+1])
    '''
    path0[:,:, 0, 0] = H[0, None,:]
    path0[:,:, 0, 1:] = mesh[0][:, None]
    E0[:] = H[0, None,:] * lin[0][:, None]

    isEven = False
    for t in range(1, len(lin)):
        if isEven:
            oldE, newE = E1, E0
            oldP, newP = path1, path0
        else:
            oldE, newE = E0, E1
            oldP, newP = path0, path1
        isEven = not isEven

        kernel(mesh[t], H[t], mesh[t - 1], H[t - 1], abs(T[t] - T[t - 1]), ker_buf)  # compute cost to join each curve
        __bestPath_UB(lin[t], mesh[t], H[t], ker_buf, oldE, oldP, newE, newP, t)


def regularBestPath(lin, mesh, kernel, T, masses=None, nAtoms=1, vel=None):
    '''
    lin is a list of 1D arrays of size N^2
    mesh is a single 2D array of shape (N,N, 2)
    kernel computes the cost of connecting two points
    '''
    N, TT = mesh.shape[0], len(T)
    N2 = N ** 2
    assert np.allclose(T[1:] - T[:-1], T[1] - T[0])
    mesh = mesh.reshape(N2, 2)
    radius = N if vel in (None, np.inf) else min(N, int(np.ceil(vel * (T[1] - T[0]) * N / mesh.ptp())))

    M = 0 if masses is None else len(masses)
    balanced = (M == 0)

    if balanced:  # balanced transport
        E = np.empty(N2), np.empty(N2)
        paths = np.zeros((N2, TT, 2)), np.zeros((N2, TT, 2))
        # transport computation is simplified assuming kernel(x,y) = kernel(0,|x-y|)
        kernel_buf = np.empty((1, N2))
        kernel(np.zeros((1, 2)), mesh, abs(T[1] - T[0]), kernel_buf)
        kernel_buf.shape = N, N

    else:
        E = [np.empty((N2, M)) for _ in range(2)]
        paths = [np.zeros((N2, M, TT, 3)) - 1 for _ in range(2)]
        kernel_buf = np.empty((1, M, N2, M))  # TODO: this doesn't parallelise at the moment
        kernel(np.zeros((1, 2)), masses, mesh, masses, abs(T[1] - T[0]), kernel_buf)
        kernel_buf.shape = M, N, N, M

    if GPU and (_regularBestPath_cl is not None):
        # use opencl implementation
        step = int(np.ceil(N2 / min(cl.device_info.MAX_WORK_GROUP_SIZE, N2, 1024)))
        groups = int(np.ceil(N2 / step)),

        if balanced:
            args = ([cl.Buffer(ctx, cl_mf.READ_ONLY | cl_mf.COPY_HOST_PTR, hostbuf=thing)
                     for thing in (lin, mesh, kernel_buf, T)] +
                    [cl.Buffer(ctx, cl_mf.READ_WRITE, size=thing.nbytes)
                     for thing in E + paths] +
                    [np.int32(n) for n in (step, radius, N, T.size)])
            event = _regularBestPath_cl.regBestPath(queue, groups, groups, *args)
        else:
            args = ([cl.Buffer(ctx, cl_mf.READ_ONLY | cl_mf.COPY_HOST_PTR, hostbuf=thing)
                     for thing in (lin, mesh, masses, kernel_buf, T)] +
                    [cl.Buffer(ctx, cl_mf.READ_WRITE, size=thing.nbytes)
                     for thing in E + paths] +
                    [np.int32(n) for n in (step, radius, N, T.size, masses.size)])
            event = _regularBestPath_cl.regBestPath_UB(queue, groups, groups, *args)

        event.wait()
        E, paths = E[0], paths[0]
        cl.enqueue_copy(queue, E, args[5 + int(TT % 2 == 0) - balanced])
        cl.enqueue_copy(queue, paths, args[7 + int(TT % 2 == 0) - balanced])

    else:  # use numba implementation

        if balanced:
            _regularBestPath(lin, mesh, kernel_buf, T, *E, *paths, radius)
        else:
            _regularBestPath_UB(lin, mesh, masses, kernel_buf, T, *E, *paths, radius)
        E, paths = (E[1], paths[1]) if TT % 2 == 0 else (E[0], paths[0])
    if not balanced:  # only look at best final mass
        ind = range(N2), E.argmin(axis=1)
        E, paths = E[ind], paths[ind[0], ind[1]]

    if nAtoms > 0:
        paths = paths[np.argsort(E)]  # sorted so the first row has smallest energy
        return paths[:nAtoms]
    else:
        return paths, E


@jit(**__pparams)
def _regularBestPath(lin, M, C, T, E0, E1, path0, path1, rad):
    '''
    If p is a path 
        E(p) = sum_t lin[t][p[t]] + kernel(p[t],p[t+1])

    '''
    N, N2 = C.shape[0], M.shape[0]
    for i in prange(N2):
        path0[i, 0] = M[i]  # current curve i
        E0[i] = lin[0, i]  # current energy of curve i

    isEven = False
    for t in range(1, T.size):
        if isEven:
            oldE, newE = E1, E0
            oldP, newP = path1, path0
        else:
            oldE, newE = E0, E1
            oldP, newP = path0, path1

        for i0 in prange(N):
            for i1 in range(N):  # find best path going through M[i,ii]
                i = i1 + i0 * N
                bestE, bestJ = 10 ** 2, 0
                jj0 = max(0, i1 - rad); jj1 = min(N, i1 + rad + 1)
                for j in range(max(0, i0 - rad), min(N, i0 + rad + 1)):
                    for jj in range(jj0, jj1):  # find j with minimal energy
                        currentE = oldE[j * N + jj] + C[abs(i0 - j), abs(i1 - jj)]
                        if currentE < bestE:
                            bestE, bestJ = currentE, j * N + jj

                # copy over new curve
                newP[i,:t] = oldP[bestJ,:t]
                newP[i, t] = M[i]
                newE[i] = bestE + lin[t, i]

        isEven = not isEven


try:
    import pyopencl as cl;  # warnings.filterwarnings('ignore', category=cl.CompilerWarning)
    ctx = cl.create_some_context(interactive=False); queue = cl.CommandQueue(ctx)
    cl_mf = cl.mem_flags

    _regularBestPath_cl = cl.Program(ctx, '''
    __kernel void regBestPath(
        __global const double *lin,  __global const double *M,  __global const double *C,
        __global const double *T,  __global double *E0,  __global double *E1,
        __global double *path0,  __global double *path1,
        int step, int rad, int N, int Tsz)
    {
        int I = get_global_id(0);
        int I0 = step*I; int I1 = min(I0+step, N*N);
        int i, i0, i1, bestJ, j, jj, j0, j1, jj0, jj1;
        __global double *oldE, *newE, *oldP, *newP;
        double bestE, currentE;
    
    
        for (i=I0; i<I1; i++){
            path0[2*Tsz*i+0+0] = M[2*i];
            path0[2*Tsz*i+0+1] = M[2*i+1];
            E0[i] = lin[i];
        }
    
        bool isEven = false;
        for (int t=1; t<Tsz; t++){
            if (isEven) {
                oldE = E1; newE = E0;
                oldP = path1; newP = path0;
            } else {
                oldE = E0; newE = E1;
                oldP = path0; newP = path1;
            }
    
            barrier(CLK_LOCAL_MEM_FENCE);
    
            for (i=I0; i<I1; i++){
                i0 = i/N; i1 = i-i0*N;
                j0 = max(0,i0-rad);   j1 = min(N, i0+rad+1);
                jj0 = max(0,i1-rad); jj1 = min(N, i1+rad+1);
                bestE = 10000; bestJ = i;
    
                for (j=j0; j<j1; j++){
                    for (jj=jj0; jj<jj1; jj++){
                        currentE = oldE[j*N+jj] + C[abs(i0-j)*N + abs(i1-jj)];
                        if (currentE < bestE){
                            bestE = currentE;
                            bestJ = j*N + jj;
                        }
                    }
                }
    
                i0 = 2*Tsz*i; i1 = 2*Tsz*bestJ;
                for (j=0; j<2*t; j++){
                    newP[i0 + j] = oldP[i1 + j];
                }
                newP[i0+2*t] = M[2*i];
                newP[i0+2*t+1] = M[2*i+1];
                newE[i] = bestE + lin[N*N*t + i];
            }
    
            isEven = !isEven;
        }
    }
    '''
                                     +
                                     '''    
    __kernel void regBestPath_UB(
        __global const double *lin,  __global const double *M, 
        __global const double *H, __global const double *C,
        __global const double *T,  __global double *E0,  __global double *E1,
        __global double *path0,  __global double *path1,
        int step, int rad, int N, int Tsz, int Hsz)
    {
        int I = get_global_id(0);
        int I0 = step*I; int I1 = min(I0+step, N*N);
        int i, i0, i1, bestJ, j, jj, j0, j1, jj0, jj1, bestM, m, mm;
        __global double *oldE, *newE, *oldP, *newP;
        double bestE, currentE;
        
        /*
        path0.shape = [N**2, Hsz, Tsz, 3]
        path0[i0,i1,i2,i3] = path0[i3 + 3*(i2 + Tsz*(i1+Hsz*i0))]
        
        C.shape = [Hsz, N, N, Hsz]
        C[i0,i1,i2,i3] = C[i3 + Hsz*(i2 + N*(i1+N*i0))]

        */
    
        for (i=I0; i<I1; i++){
            for (m=0; m<Hsz; m++){
                path0[0 + 3*Tsz*(m + Hsz*i)] = H[m];
                path0[1 + 3*Tsz*(m + Hsz*i)] = M[2*i];
                path0[2 + 3*Tsz*(m + Hsz*i)] = M[1 + 2*i];
                E0[m + i*Hsz] = H[m]*lin[i];
            }
        }
    
        bool isEven = false;
        for (int t=1; t<Tsz; t++){
            if (isEven) {
                oldE = E1; newE = E0;
                oldP = path1; newP = path0;
            } else {
                oldE = E0; newE = E1;
                oldP = path0; newP = path1;
            }
    
            barrier(CLK_LOCAL_MEM_FENCE);
    
            for (i=I0; i<I1; i++){
                i0 = i/N; i1 = i-i0*N;
                j0 = max(0,i0-rad);   j1 = min(N, i0+rad+1);
                jj0 = max(0,i1-rad); jj1 = min(N, i1+rad+1);
                for (m=0; m<Hsz; m++){
                    i0 = i/N; i1 = i-i0*N;
                    bestE = 10000; bestJ = i; bestM = 0;
                    for (j=j0; j<j1; j++){
                        for (jj=jj0; jj<jj1; jj++){
                            for (mm=0; mm<Hsz; mm++){
                                currentE = oldE[mm + Hsz*(jj + j*N)] + C[mm + Hsz*(abs(i1-jj) + N*(abs(i0-j) + N*m))];
                                if (currentE < bestE){
                                    bestE = currentE;
                                    bestJ = j*N + jj;
                                    bestM = mm;
                                }
                    }   }   }   

                    if (H[bestM]==0){
                        bestJ = i;
                    }
                    i0 = 3*Tsz*(m+Hsz*i); i1 = 3*Tsz*(bestM+Hsz*bestJ);
                    for (j=0; j<3*t; j++){
                        newP[i0 + j] = oldP[i1 + j];
                    }
                    newP[3*t + i0] = H[m];
                    newP[1 + 3*t + i0] = M[2*i];
                    newP[2 + 3*t + i0] = M[1 + 2*i];
                    newE[m + Hsz*i] = bestE + H[m]*lin[i + N*N*t];
                }   
            }
    
            isEven = !isEven;
        }
    }    
    ''').build()
    GPU = True  # default use opencl implementation
except ImportError:
    GPU, _regularBestPath_cl = False, None


@jit(**__pparams)
def _regularBestPath_UB(lin, M, H, C, T, E0, E1, path0, path1, rad):
    '''
    If p is a path
        E(p) = sum_t lin[t][p[t]] + C(p[t],p[t+1])
    '''
    N, N2 = C.shape[1], M.shape[0]
    for i in prange(N2):
        for m in range(H.size):
            path0[i, m, 0, 0] = H[m]  # current mass i
            path0[i, m, 0, 1:] = M[i]  # current position i
            E0[i, m] = H[m] * lin[0, i]  # current energy of curve i

    isEven = False
    for t in range(1, T.size):
        if isEven:
            oldE, newE = E1, E0
            oldP, newP = path1, path0
        else:
            oldE, newE = E0, E1
            oldP, newP = path0, path1
        isEven = not isEven

        for i in prange(N2):  # find best path going through M[i,ii]
            i0 = i // N
            i1 = i - i0 * N
            j00 = max(0, i0 - rad); j01 = min(N, i0 + rad + 1)
            j10 = max(0, i1 - rad); j11 = min(N, i1 + rad + 1)
            for m in range(H.size):
                bestE, bestJ, bestM = 10 ** 15, 0, 0
                for j in range(j00, j01):
                    for jj in range(j10, j11):  # find j with minimal energy
                        J = j * N + jj
                        for mm in range(H.size):
                            currentE = oldE[J, mm] + C[m, abs(i0 - j), abs(i1 - jj), mm]
                            if currentE < bestE:
                                bestE, bestJ, bestM = currentE, j * N + jj, mm

                # copy over new curve
                newP[i, m,:t] = oldP[bestJ, bestM,:t]
                newP[i, m, t, 0] = H[m]
                newP[i, m, t, 1:] = M[i]
                newE[i, m] = bestE + H[m] * lin[t, i]
