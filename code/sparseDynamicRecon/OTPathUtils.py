'''
Created on 15 Jul 2021

@author: Rob Tovey
'''
from .utils import jit, prange, __pparams, FLOAT
import numpy as np


def path_kernels(name=None, norm=None, weight=None, T=None, **_):
    '''
    Returns:
    --------
        energy(rho) = cost of path rho = [mass, x0, y0, x1, y1,...]
            rho is an array of shape [1 + len(T)*dim]

        grad(rho) = (cost, d(cost)/drho) where cost = energy(rho)
            rho is an array of shape [1 + len(T)*dim]

        step_mesh(x,y,dt)[i,j] = energy for step between (x[i],y[j]) in time dt
            x is an array of shape [n,dim]
            y is an array of shape [m,dim]
            
    '''
    assert T is not None
    DT = T[1:] - T[:-1]
    # TODO: implement mass terms in balanced transport
    if name in (None, 'Dynamic p=1'):
        raise NotImplementedError('No mass terms, old format')
        '''
        step(x,y,dt) = min_z \int_0^dt |z'(t)|^{norm}/(norm) dt s.t. z(0)=x, z(dt)=y
                    = |y-x|^{norm}/((norm)*dt^{norm-1})
                    
        step_grad(x,y,dt) = sign(x-y) (|y-x|/dt)^{norm-1}
        '''
        assert norm is not None
        if norm == 1:
            def step(x, y, dt, out):
                n = x.shape[1]
                for i in range(x.shape[0]):
                    s = 0.0
                    for k in range(n):
                        s += abs(x[i, k] - y[i, k])
                    out[i] = s

            def step_mesh(x, y, dt, out):
                n = x.shape[1]
                for i in range(x.shape[0]):
                    for j in range(y.shape[0]):
                        s = 0.0
                        for k in range(n):
                            s += abs(x[i, k] - y[j, k])
                        out[i, j] = s

            def step_grad(x, y, dt, out, outgrad):
                n = x.shape[0]
                for j in range(y.shape[0]):
                    s = 0
                    for k in range(n):
                        tmp = x[k] - y[j, k]
                        if tmp > 0:
                            s += tmp
                            outgrad[j, k] = 1
                        else:
                            s -= tmp
                            outgrad[j, k] = -1
                    out[j] = s
        else:
            def step(x, y, dt, out):
                scale = 1 / (norm * dt ** (norm - 1))
                n = x.shape[1]
                for i in range(x.shape[0]):
                    s = 0.0
                    for k in range(n):
                        s += abs(x[i, k] - y[i, k]) ** norm
                    out[i] = s * scale

            def step_mesh(x, y, dt, out):
                scale = 1 / (norm * dt ** (norm - 1))
                n = x.shape[1]
                for i in range(x.shape[0]):
                    for j in range(y.shape[0]):
                        s = 0.0
                        for k in range(n):
                            s += abs(x[i, k] - y[j, k]) ** norm
                        out[i, j] = s * scale

            def step_grad(x, y, dt, out, outgrad):
                scale0, scale1 = 1 / (norm * dt ** (norm - 1)), dt ** (1 - norm)
                n = x.shape[0]
                for j in range(y.shape[0]):
                    s = 0
                    for k in range(n):
                        tmp = x[k] - y[j, k]
                        s += abs(tmp) ** norm
                        if tmp > 0:
                            outgrad[j, k] = scale1 * abs(tmp) ** (norm - 1)
                        else:
                            outgrad[j, k] = -scale1 * abs(tmp) ** (norm - 1)
                    out[j] = scale0 * s
    elif name == 'Benamou Brenier':
        '''
        alpha, beta = weight[0], weight[1]
        step(x,y,dt) = alpha*dt + beta/(2*dt) * |x-y|^2
                    
        step_grad(x,y,dt) = ( alpha*dt + beta/(2*dt) * |x-y|^2,
                                  beta/dt * (x-y))
        '''
        scale0, scale1 = weight[0] * (T[-1] - T[0]), weight[1] / (2 * DT)
        scale2 = 2 * scale1

        def energy(rho, out):
            for i in prange(rho.shape[0]):
                s = scale0
                for j in range(DT.size):
                    s += scale1[j] * (abs(rho[i, 1 + 2 * j] - rho[i, 3 + 2 * j]) ** 2
                                     +abs(rho[i, 2 + 2 * j] - rho[i, 4 + 2 * j]) ** 2)
                out[i] = rho[i, 0] * s

        def grad(rho, out, outgrad):
            for i in prange(rho.shape[0]):
                s = scale0
                K = 0
                for j in range(DT.size):
                    for k in range(2):
                        tmp = rho[i, 1 + K] - rho[i, 3 + K]
                        s += scale1[j] * tmp ** 2  # add to energy

                        tmp *= scale2[j]  # rescale for gradient
                        outgrad[i, K] += tmp
                        outgrad[i, 2 + K] -= tmp

                        K += 1  # K = 2*j + k
                out[i] = s

        def step_mesh(x, y, dt, out):
            scale0 = weight[0] * dt
            scale1 = weight[1] / (2 * dt)
            for i in prange(x.shape[0]):
                for j in range(y.shape[0]):
                    out[i, j] = scale0 + scale1 * (
                        abs(x[i, 0] - y[j, 0]) ** 2 + abs(x[i, 1] - y[j, 1]) ** 2)

    return (jit('void(T[:,:],T[:])'.replace('T', 'f8'), inline='always', **__pparams)(energy),
            jit('void(T[:,:],T[:],T[:,:])'.replace('T', 'f8'), inline='always', **__pparams)(grad),
            jit('void(T[:,:],T[:,:],T,T[:,:])'.replace('T', 'f8'), inline='always', **__pparams)(step_mesh))


def UBpath_kernels(name=None, norm=None, weight=1, T=None, **_):
    '''
    Returns:
    --------
        energy(rho) = cost of path rho = [m0, x0, y0, m1, x1, y1,...]
            rho is an array of shape [len(T)*(1+dim)]

        grad(rho) = (cost, d(cost)/drho) where cost = energy(rho)
            rho is an array of shape [len(T)*(1+dim)]

        step_mesh(x,a,y,b,dt)[i,j] = energy for step between (x[i0],a[i1],y[j0],b[j1]) in time dt
            x is an array of shape [n0,dim]
            a is an array of shape [n1]
            y is an array of shape [m0,dim]
            b is an array of shape [m1]
    '''
    if np.isscalar(weight):
        weight = weight ** .5, 1 / weight ** .5

    assert T is not None
    DT = T[1:] - T[:-1]

    if name == 'Benamou Brenier':
        '''        
        step((a,x),(b,y),dt) = int_0^{dt} [alpha + beta/2*|z'|^2 +  gamma/2*(h'/h)^2] * h(t)
            for some z(0)=x, z(dt)=y, h(0)=a, h(dt)=b.
        
        This is really hard to solve so we switch to H(t) = sqrt(h(t)) and assume 
        the mass term is just alpha*(h(1)+h(0))/2.

        alpha, beta, gamma = weight[0], weight[1], weight[2]
        delta = sqrt(weight[2]/weight[1])
        step(x,y,a,b,dt) = (.5*alpha*dt + 2*gamma/dt) * (a^2+b^2) - (4*gamma/dt)*a*b*cos( |x-y|/(2*delta) ) )
                        
        step_grad(x,y,a,b,dt) = ( (alpha*dt + 4*gamma/dt)*a + 4*gamma/dt*b*cos(...)) ,
                                  2*gamma/delta/dt * a*b * (x-y)/|x-y| * sin(...))
        '''
        assert abs(DT - DT[0]).max() < 1e-10
        dt = DT[0]
        alpha, gamma, delta = weight[0], weight[2], (weight[2] / weight[1]) ** .5

        scale0 = .5 * alpha * dt + 2 * gamma / dt
        scale1 = 4 * gamma / dt
        scale2 = .5 / delta
        thresh = (np.pi / scale2) ** 2  # |x-y|/(2*delta) < pi <=> |x-y|^2 < thresh
        ZERO = (0.01 / scale2) ** 2  # small-angle approximation of cos is valid

        def energy(rho, out):
            for i in prange(rho.shape[0]):
                s = 0
                x = 0; y = 0
                b = rho[i, 0]; dx = rho[i, 1]; dy = rho[i, 2]
                for j in range(DT.size):
                    J = 3 * j
                    a = b; x += dx; y += dy
                    b = rho[i, J + 3]
                    dx = rho[i, J + 4] - x
                    dy = rho[i, J + 5] - y

                    s += scale0 * (a ** 2 + b ** 2)  # pure mass term

                    # d = |x-y|^2
                    d = dx ** 2 + dy ** 2
                    if d >= thresh:  # cos(...) = -1
                        s += scale1 * a * b
                    elif d <= ZERO:  # small-angle approximation of cos
                        # cos(t) = 1-t^2/2,   t^2 = scale2^2 * d
                        s -= scale1 * a * b * (1 - .5 * scale2 ** 2 * d)
                    else:
                        s -= scale1 * a * b * np.cos(scale2 * np.sqrt(d))

                out[i] = s

        def grad(rho, out, outgrad):
            for i in prange(rho.shape[0]):
                s = 0
                x = 0; y = 0
                b = rho[i, 0]; dx = rho[i, 1]; dy = rho[i, 2]
                for j in range(DT.size):
                    J = 3 * j
                    a = b; x += dx; y += dy
                    b = rho[i, J + 3]
                    dx = rho[i, J + 4] - x
                    dy = rho[i, J + 5] - y

                    s += scale0 * (a ** 2 + b ** 2)  # pure mass term
                    outgrad[i, J] += 2 * scale0 * a
                    outgrad[i, J + 3] += 2 * scale0 * b

                    d = dx ** 2 + dy ** 2
                    if d >= thresh:  # cos(...) = -1
                        s += scale1 * a * b
                        outgrad[i, J] += scale1 * b
                        outgrad[i, J + 3] += scale1 * a

                    elif d <= ZERO:  # small-angle approximation of cos
                        # cos(t) = 1-t^2/2,   t^2 = scale2^2 * d
                        tmp = scale1 * (1 - .5 * scale2 ** 2 * d)
                        s -= tmp * a * b
                        outgrad[i, J] -= b * tmp
                        outgrad[i, J + 3] -= a * tmp

                        tmp = scale1 * a * b * scale2 ** 2
                        outgrad[i, J + 1] -= tmp * dx
                        outgrad[i, J + 2] -= tmp * dy
                        outgrad[i, J + 4] += tmp * dx
                        outgrad[i, J + 5] += tmp * dy

                    else:
                        d = np.sqrt(d)
                        EXP = scale1 * np.exp(1j * scale2 * d)  # = cos(...) + i*sin(...)
                        s -= a * b * EXP.real
                        outgrad[i, J] -= b * EXP.real
                        outgrad[i, J + 3] -= a * EXP.real

                        tmp = a * b * EXP.imag * scale2 / d
                        outgrad[i, J + 1] -= tmp * dx
                        outgrad[i, J + 2] -= tmp * dy
                        outgrad[i, J + 4] += tmp * dx
                        outgrad[i, J + 5] += tmp * dy

                out[i] = s

        def step_mesh(x, a, y, b, dt, out):
            s0 = .5 * alpha * dt + 2 * gamma / dt
            s1 = 4 * gamma / dt
            for i0 in prange(x.shape[0]):
                for i1 in prange(a.size):
                    aa = a[i1]
                    for j0 in range(y.shape[0]):
                        d = (x[i0, 0] - y[j0, 0]) ** 2 + (x[i0, 1] - y[j0, 1]) ** 2
                        if d >= thresh:
                            for j1 in range(b.size):
                                out[i0, i1, j0, j1] = (
                                        s0 * (aa + b[j1]) + s1 * np.sqrt(aa * b[j1])
                                    )
                        elif d <= ZERO:  # small-angle approximation of cos
                            # cos(t) = 1-t^2/2,   t^2 = scale2^2 * d
                            cos = s1 * (1 - .5 * scale2 ** 2 * d)
                            for j1 in range(b.size):
                                out[i0, i1, j0, j1] = (
                                        s0 * (aa + b[j1]) - np.sqrt(aa * b[j1]) * cos
                                    )
                        else:
                            cos = s1 * np.cos(scale2 * np.sqrt(d))
                            for j1 in range(b.size):
                                out[i0, i1, j0, j1] = (
                                        s0 * (aa + b[j1]) - np.sqrt(aa * b[j1]) * cos
                                    )

    elif name == 'Lazy Benamou Brenier':
        '''        
        step((a,x),(b,y),dt) = int_0^{dt} [alpha + beta/2*|x'|^2 +  gamma/2*(h'/h)^2] * h(t)
        step((a,x),(b,y),dt) = int_0^{dt} [alpha + beta/2*|x'|^2]g(t)^2 +  gamma/2*(g'(t))^2]
        g(t) = sqrt(h(t))
        
        This is really hard to solve so we switch to a(t) = sqrt(h(t)) and assume 
        the mass term is just alpha*(h(1)+h(0))/2.

        alpha, beta, gamma = weight[0], weight[1], weight[2]
        step(x,y,a,b,dt) = (alpha*dt + beta/(2*dt)*|x-y|^2) * (a**2+b**2)/2 + gamma/(2*dt)*(a-b)^2 )
        
        step_grad(x,y,a,b,dt) = ( (alpha*dt + beta|x-y|^2/(2*dt))*a + gamma/(2*dt) * (a-b) ,
                                  beta*(a**2+b**2)/(2*dt)*(x-y))
        '''
        alpha, beta, delta = weight[0], weight[1], (weight[2] / weight[1]) ** .5

        def step(X, Y, dt, out):
            s0, s1, s2 = .5 * weight[0] * dt, weight[1] / (4 * dt), weight[2] / (2 * dt)
            N, n = X.shape[0], X.shape[1]
            for i in range(N):
                a = X[i, 0]; b = Y[i, 0]
                d = 0.0
                for j in range(1, n):
                    d += abs(X[i, j] - Y[i, j]) ** 2

                out[i] = (s0 + s1 * d) * (a ** 2 + b ** 2) + s2 * (a - b) ** 2

        def step_mesh(x, a, y, b, dt, out):
            s0, s1, s2 = .5 * weight[0] * dt, weight[1] / (4 * dt), weight[2] / (2 * dt)
            n = x.shape[1]
            for i0 in range(x.shape[0]):
                for i1 in range(a.size):
                    aa = a[i1]
                    for j0 in range(y.shape[0]):
                        d = 0.0
                        for k in range(n):
                            d += abs(x[i0, k] - y[j0, k]) ** 2

                        for j1 in range(b.size):
                            out[i0, i1, j0, j1] = (s0 + s1 * d) * (aa + b[j1]) + s2 * (np.sqrt(aa) - np.sqrt(b[j1])) ** 2

        def step_grad(X, Y, dt, out, outgrad1, outgrad2):
            s0, s1, s2 = .5 * weight[0] * dt, weight[1] / (4 * dt), weight[2] / (2 * dt)
            N, n = Y.shape[0], Y.shape[1]

            a = X[0]
            for j in range(N):
                b = Y[j, 0]
                d = 0.0
                for k in range(1, n):
                    d += abs(X[k] - Y[j, k]) ** 2

                out[j] = (s0 + s1 * d) * (a ** 2 + b ** 2) + s2 * (a - b) ** 2
                outgrad1[j, 0] = 2 * (s0 + s1 * d) * a + 2 * s2 * (a - b)
                outgrad2[j, 0] = 2 * (s0 + s1 * d) * b + 2 * s2 * (b - a)
                for k in range(1, n):
                    outgrad1[j, k] = 2 * s1 * (a ** 2 + b ** 2) * (X[k] - Y[j, k])

            for j in range(N):
                for k in range(1, n):
                    outgrad2[j, k] = -outgrad1[j, k]

    elif name == 'Linear BB':
        '''        
        This is an even lazier version of Lazy Benamou-Brenier which parametrises with h,
        rather than sqrt(h). We still use the quadratic attraction between time-points 
        which is physically strange. The main advantage is that it is equivalent to 
        the balanced transport implementation when masses are constrained.

        alpha, beta, gamma = weight[0], weight[1], weight[2]
        step(x,y,a,b,dt) = (alpha*dt + beta/(2*dt)*|x-y|^2) * (a+b)/2 + gamma/(2*dt)*(a-b)^2 )
        
        step_grad(x,y,a,b,dt) = ( (alpha*dt + beta|x-y|^2/(2*dt))/2 + gamma/(2*dt) * (a-b) ,
                                  beta*(a+b)/(2*dt)*(x-y))
        '''
        def step(X, Y, dt, out):
            s0, s1, s2 = .5 * weight[0] * dt, weight[1] / (4 * dt), weight[2] / (2 * dt)
            N, n = X.shape[0], X.shape[1]
            for i in range(N):
                a = X[i, 0]; b = Y[i, 0]
                d = 0.0
                for j in range(1, n):
                    d += abs(X[i, j] - Y[i, j]) ** 2

                out[i] = (s0 + s1 * d) * (a + b) + s2 * (a - b) ** 2

        def step_mesh(x, a, y, b, dt, out):
            s0, s1, s2 = .5 * weight[0] * dt, weight[1] / (4 * dt), weight[2] / (2 * dt)
            n = x.shape[1]
            for i0 in range(x.shape[0]):
                for i1 in range(a.size):
                    aa = a[i1]
                    for j0 in range(y.shape[0]):
                        d = 0.0
                        for k in range(n):
                            d += abs(x[i0, k] - y[j0, k]) ** 2

                        for j1 in range(b.size):
                            out[i0, i1, j0, j1] = (s0 + s1 * d) * (aa + b[j1]) + s2 * (aa - b[j1]) ** 2

        def step_grad(X, Y, dt, out, outgrad1, outgrad2):
            s0, s1, s2 = .5 * weight[0] * dt, weight[1] / (4 * dt), weight[2] / (2 * dt)
            N, n = Y.shape[0], Y.shape[1]

            a = X[0]
            for j in range(N):
                b = Y[j, 0]
                d = 0.0
                for k in range(1, n):
                    d += abs(X[k] - Y[j, k]) ** 2

                out[j] = (s0 + s1 * d) * (a + b) + s2 * (a - b) ** 2
                outgrad1[j, 0] = (s0 + s1 * d) + 2 * s2 * (a - b)
                outgrad2[j, 0] = (s0 + s1 * d) + 2 * s2 * (b - a)
                for k in range(1, n):
                    outgrad1[j, k] = 2 * s1 * (a + b) * (X[k] - Y[j, k])

            for j in range(N):
                for k in range(1, n):
                    outgrad2[j, k] = -outgrad1[j, k]

#     return (step), (step_mesh), (step_grad)
    return (jit('void(T[:,:],T[:])'.replace('T', 'f8'), **__pparams)(energy),
            jit('void(T[:,:],T[:],T[:,:])'.replace('T', 'f8'), **__pparams)(grad),
            jit('void(T[:,:],T[:],T[:,:],T[:],T,T[:,:,:,:])'.replace('T', 'f8'), **__pparams)(step_mesh))


class pathOT:
    balanced = True

    def __init__(self, T, **kwargs):
        self._ker_params = {'name':None, 'norm':2, 'weight':1, 'T':T, 'velocity':np.inf}
        self.update_kernel(**kwargs)

    def update_kernel(self, **kwargs):
        if 'name' in kwargs:
            self._ker_params['norm'] = None
        self._ker_params = {k:kwargs.get(k, v) for k, v in self._ker_params.items()}
        self.kernel = path_kernels(**self._ker_params)
        self.velocity = self._ker_params['velocity']
        self._shape = -1, 1 + 2 * self.T.size

    def scale_kernel(self, s):
        W = self._ker_params['weight']
        if not hasattr(W, '__len__'):
            self.update_kernel(weight=s * W)
            return
        if np.isscalar(s):
            s = (s,) * len(W)
        self.update_kernel(weight=tuple(w * s[i] for i, w in enumerate(W)))

    @property
    def weight(self): return self._ker_params['weight']
    @property
    def T(self): return self._ker_params['T']

    def _parse_rho(self, rho):
        if hasattr(rho, 'asarray'):
            rho = rho.asarray()
        return rho.reshape(self._shape)

    def __call__(self, rho):
        rho = self._parse_rho(rho)
        if rho.shape[0] == 0:
            return 0
        buf = np.empty(rho.shape[0])
        self.kernel[0](rho, buf)
        return buf.sum()

    def grad(self, rho):
        rho = self._parse_rho(rho)
        mass = rho[:, 0].copy()
        dRho = np.empty(rho.shape[0])
        dX = np.zeros((rho.shape[0], rho.shape[1] - 1))
        self.kernel[1](rho, dRho, dX)
        return dRho.dot(mass), dRho, (dX * mass[:, None]).reshape(mass.size, -1, 2)


class UBpathOT(pathOT):
    balanced = False

    def update_kernel(self, **kwargs):
        if 'name' in kwargs:
            self._ker_params['norm'] = None
        self._ker_params = {k:kwargs.get(k, v) for k, v in self._ker_params.items()}
        self.kernel = UBpath_kernels(**self._ker_params)
        self.velocity = self._ker_params['velocity']
        self._shape = -1, 3 * self.T.size

    def __call__(self, rho):
        rho = self._parse_rho(rho)
        if rho.shape[0] == 0:
            return 0
        buf = np.empty(rho.shape[0])
        self.kernel[0](rho, buf)
        return buf.sum()

    def toarray(self, rho, grad=False):
        '''
        It is possible to use a non-linear scaling of mass in the 
        parametrisation of rho. For example, in the Benamou-Brenier
        case, it is convenient to have
            mass = rho.a ^ 2.
            
        This function returns an array with same shape as rho, but 
        corrected mass. If grad==True, the derivative of this transform 
        is also returned.
        '''
        rho = self._parse_rho(rho)
        name = self._ker_params['name']

        if type(name) is str and 'Benamou Brenier' in name:
            M = rho.copy()
            M[:,::3] *= M[:,::3]
            if grad:
                dM = np.ones(rho.shape)
                dM[:,::3] = 2 * rho[:,::3]
                return M, dM
            else:
                return M
        else:
            if grad:
                return rho, 1
            else:
                return rho

    def fromarray(self, M):
        '''
        The inverse of self.toarray
        '''
        name = self._ker_params['name']

        if type(name) is str and 'Benamou Brenier' in name:
            rho = M.copy()
            rho[:,::3] = np.sqrt(rho[:,::3])
            return rho
        else:
            return M

    def grad(self, rho):
        rho = self._parse_rho(rho)
        E = np.empty(rho.shape[0], dtype=FLOAT)
        dRho = np.zeros(rho.shape, dtype=FLOAT)

        self.kernel[1](rho, E, dRho)
        return E.sum(), dRho

