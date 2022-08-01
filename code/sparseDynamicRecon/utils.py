'''
Created on 2 Dec 2020

@author: rtovey
'''
from os import path, makedirs
from time import time
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib import pyplot as plt, ticker
from numba import jit, prange
INT, FLOAT, COMPLEX = np.dtype('int32'), np.dtype('float64'), np.dtype('complex128')
ZERO, CURVE_TOL = 1e-10, 1e-10
__params = {'nopython': True, 'parallel': False, 'fastmath': False, 'boundscheck': False, 'cache': True}
# __params = {'forceobj':True}
__pparams = __params.copy(); __pparams['parallel'] = True


def Norm(x):
    if isinstance(x, np.ndarray):
        return np.linalg.norm(x.ravel())
    elif hasattr(x, 'norm'):
        return x.norm()
    elif hasattr(x, 'asarray'):
        return np.linalg.norm(x.asarray().ravel())
    else:
        raise NotImplementedError


def Distance(x, y):
    if isinstance(x, np.ndarray):
        return np.linalg.norm((x - y).ravel())
    elif hasattr(x, 'distance'):
        return x.distance(y)
    elif hasattr(y, 'distance'):
        return y.distance(x)
    else:
        return Norm(x - y)


def timefrmt(t):
    if t < 100:  # 100s
        return '%5.2fs' % t
    elif t < 60 * 60:  # 1h
        return '%2dm%2ds' % (t // 60, int(t % 60))
    elif t < 60 * 60 * 24:  # 1day
        return '%2dh%2dm' % (t // (60 * 60), (t % (60 * 60)) // 60)
    else:
        day = 60 * 60 * 24
        return '%2dd%2dh' % (t // day, (t % day) // (60 * 60))
    pass


class MinorSymLogLocator(ticker.Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """

    def __init__(self): pass

    def __call__(self):
        'Return the locations of the ticks'
        linthresh = self.axis.get_transform().linthresh
        majorlocs = self.axis.get_majorticklocs(); n = len(majorlocs)
        lims = self.axis.get_view_interval()  # I know that I don't care about the axis below 0

        # Ticks below major ticks
        if n > 0:
            minorlocs = [k * majorlocs[0] / 10 for k in range(-9, 10) if k * majorlocs[0] / 10 > lims[0]]
        else:
            minorlocs = []

        # Ticks between major ticks
        for i in range(n - 1):
            m, M = majorlocs[i], majorlocs[i + 1]
            if M <= linthresh:
                step = .1 * max(M - m, 10 ** np.floor(np.log10(linthresh)))
                minorlocs.extend([m + k * step for k in range(1, 10)])
            else:
                step = (M - m) / 9
                minorlocs.extend([m + k * step for k in range(1, 9)])

        # Ticks above major ticks
        if n > 2:
            m, M = majorlocs[-1], majorlocs[-1] ** 2 / majorlocs[-2]
        elif n == 1:
            m, M = majorlocs[-1], 10 * majorlocs[-1]
        else:
            m, M = lims
        step = (M - m) / 9
        minorlocs.extend([m + k * step for k in range(1, 9) if m + k * step < lims[1]])

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))


class callback:

    def __init__(self, iters=100, frequency=1, quiet=False,
                 stop=0, record={}, extras=None, plot=None):
        self.iters = iters
        # All values are only recorded every <frequency> iterations
        self.frequency = frequency
        # Only print convergence properties if <quiet> is False
        self.quiet = quiet
        # Terminate if the normalised distance between iterations is smaller than this value
        self.stop = stop

        self.plot = plot
        if plot is not None:
            if plot is True:
                plot['fig'] = plt.gcf()
            else:  # plot is MovieMaker
                plot['fig'] = plot.get('fig', None)

        # <record> is a dictionary noting which values should be recorded.
        # Valid keys are iterates, and extras.
        if type(record) is str:
            self.save = {record: True}
        elif type(record) is not dict:
            self.save = {thing: True for thing in record}
        else:
            self.save = record.copy()
        for thing in ('iterates', 'extras'):
            if thing not in self.save:
                self.save[thing] = False

        self._i = 0
        # tot = total number of recorded time-points
        self.i, j = [], 0
        for i in range(self.iters + 1):
            if i == j:
                self.i.append(i)
                if type(self.frequency) is int:
                    j = min(j + self.frequency, self.iters)
                else:
                    j = min(max(j + 1, int(j * self.frequency)), self.iters)

        self.i = np.array(self.i, dtype='int32')
        self.X = [] if self.save['iterates'] else None
        # Every algorithm stores step-sizes and time points
        self.D = np.empty(self.i.shape, dtype=FLOAT)
        self.T = np.zeros(self.i.shape, dtype=FLOAT)

        # <extras> is a function of x which returns a vector of values to be stored
        # <extras>() returns an integer indicating the number of values.
        # The energy will always be prepended to each row
        if self.save['extras']:
            assert extras is not None
            self.extras = extras
            self.E = np.empty((self.i.size, len(self.extras())), dtype=FLOAT) if self.save['extras'] else None

    def __call__(self, i, X, x, extras={}):
        I = self._i
        if i >= self.i[I]:  # record new point
            self.i[I], self._i = i, I + 1
            self._record(I, X, x, extras)

            if self.plot is not None:
                self.plot['call'](self, i, X, self.plot['fig'], plt)

            self._tic = time()  # exclude plotting time in time recording

        b = False  # decide whether to terminate
        if i >= self.iters:
            b = True
        elif np.isscalar(self.stop):
            n = Distance(X, x) / max(1e-10, Norm(X))
            b = (n <= self.stop) and (i > 1)
        else:
            b = self.stop(i, X, x)

        if b:  # if terminating then tidy up arrays
            if I < len(self.i) and self._i != I + 1:
                self.i[I], self._i = i, I + 1
                self._record(I, X, x, extras)
                self.i = self.i[:I + 1]
                self.T = self.T[:I + 1]
                if self.save['extras']:
                    self.E = self.E[:I + 1]
            if self.plot is not None:
                self.plot['call'](self, i, X, self.plot['fig'], plt)

            for thing in (self.D, self.T, self.E):
                if thing is not None:
                    thing[self._i:] = thing[self._i - 1]
            self.T -= self.T[0]  # measure time from start, not just time
            self.end = self._i
            if not self.quiet:
                print()  # print new line to keep things tidy
        return b

    def _record(self, I, X, x, extras):
        self.T[I] = 0 if I == 0 else (self.T[I - 1] + time() - self._tic)
        if self.save['iterates']:
            self.X.append(X.copy())
        self.D[I] = Distance(X, x)
        if self.save['extras']:
            self.E[I, :] = self.extras(X, x, self.D[I], **extras)

        if not self.quiet:
            if I == 0:
                print(' Iter   |x^n-x^{n-1}|       %s' % ('' if self.E is None else 'Energy gap'))
            if self.E is None:
                print('%4d\t% 1.5e\t' % (self.i[I], self.D[I]))
            else:
                print('%4d\t% 1.5e\t% 1.5e' % (self.i[I], self.D[I], self.E[I, 0]))


class MovieMaker:
    def __init__(self, filename=None, fps=5, fig=None, dummy=False):
        from matplotlib import animation

        if dummy:
            self.fig = None
        else:
            self.fig = fig
            self.fig = self.getFig()  # initialises a default figure if fig was None

        if dummy:  # no plotting at all
            self.doplot = 0
        elif filename is None:  # plotting without save
            self.doplot = 1
            self.fig.show()
        else:  # plotting with save
            self.doplot = 2
            if type(filename) is not str:
                filename = path.join(*filename)
            D = path.dirname(filename)
            if len(D) > 0 and not path.exists(D):
                makedirs(D, exist_ok=True)
            if not filename.endswith('.mp4'):
                filename = filename + '.mp4'
            self.writer = animation.writers['ffmpeg'](fps=fps, metadata={'title': filename})
            self.writer.setup(self.fig, filename, dpi=100)

        self.fps = fps
        plt.pause(0.1)

    def getFig(self):
        if self.fig is None:
            from matplotlib.pyplot import figure
            try:  # try to move figure to top-left corner of screen
                F = figure('MovieMaker', figsize=(18, 10))
                window = F.canvas.manager.window
                if hasattr(window, 'move'):
                    window.move(0, 0)
                elif hasattr(window, 'SetPosition'):
                    window.SetPosition((0, 0))
                elif hasattr(window, 'wm_geometry'):
                    window.wm_geometry('+0+0')
            except:
                pass

            return F
        else:
            return self.fig

    def update(self):
        if self.doplot > 0:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            if self.doplot > 1:
                self.writer.grab_frame()

    def finish(self):
        if self.doplot > 0:
            plt.pause(0.1)
            if self.doplot > 1:
                self.update()
                self.writer.finish()


def plotter(GT, pms, mv, paths=False, doPlot=True, window=None):
    if not doPlot:  # don't do any plotting
        return (lambda *_: None), (MovieMaker(dummy=True) if mv is None else mv)
    elif mv is None or mv.doplot == 0:  # plot but don't save
        mv = MovieMaker(dummy=False)
    else:  # plotting and saving, clear figure first
        mv.fig.clear()
    if window is None:  # spatial window to plot
        window = [0, 0], [1, 1]

    tic = time()

    def doPlot(cback, i, rho, fig, plt):
        if pms.get('axes', None) is None:
            fig.clear()
            gs = GridSpec(2, 3)
            pms['axes'] = plt.subplot(gs[:, :2]), plt.subplot(gs[0, -1]), plt.subplot(gs[1, -1])

        title_spec = {'fontsize': 30}

        ax = pms['axes'][0]
        title = 'Reconstruction, %3d iters, %s' % (i, timefrmt(time() - tic))
        if pms.get('recon plot', None) is None:
            if GT is not None:
                pms['scale'] = GT.norm() ** .25
                GT.plot(ax=ax, paths=True, alpha=1, minwidth=1, width=4)
            pms['recon plot'] = rho.plot(ax=ax, paths=paths, alpha=.3)
            ax.set_xlim(window[0][1], window[1][1]); ax.set_ylim(window[0][0], window[1][0])
            ax.set_xlabel('x'); ax.set_ylabel('y')
            if paths:
                cbar = fig.colorbar(ScalarMappable(cmap=pms['recon plot'].cmap,
                                                   norm=pms['recon plot'].norm), ax=ax, pad=0.01)
            else:
                cbar = fig.colorbar(ScalarMappable(cmap=pms['recon plot'][0].cmap,
                                                   norm=pms['recon plot'][0].norm), ax=ax, pad=0.01)
            cbar.ax.set_ylabel('t', rotation=0)
        pms['recon plot'] = rho.plot(ax=ax, width=100 / pms.get('scale', 1),
                                     minwidth=1, paths=paths, update=pms['recon plot'])
        ax.set_title(title, title_spec)

        ax = pms['axes'][1]
        if pms.get('stepsize plot', None) is None:
            pms['stepsize plot'] = ax.plot([1], [1])[0]
            ax.set_yscale('log'); ax.set_title('Stepsize convergence', title_spec)
            ax.set_xlabel('iterations'); ax.set_ylabel(r'$|\rho_n - \rho_{n-1}|$')

        else:
            x, y = cback.i[:cback._i], cback.D[:cback._i]
            pms['stepsize plot'].set_data(x, y)
            pms['ax1'] = min(pms['ax1'][0], .9 * y[-1] if y.size > 1 else rho.norm()), max(pms['ax1'][1], 1.1 * y[-1])
            ax.set_xlim(1, x[-1] + .5); ax.set_ylim(max(1e-8, pms['ax1'][0]), pms['ax1'][1])
        if i > 40:
            ax.set_xscale('symlog', linthresh=40); ax.xaxis.set_minor_locator(MinorSymLogLocator())

        ax = pms['axes'][2]
        if pms.get('energy plot', None) is None:
            pms['energy plot'] = ax.plot([1], [1])[0]
            ax.set_yscale('log'); ax.set_title('Energy convergence', title_spec)
            ax.set_xlabel('iterations');  # ax.set_ylabel(r'$E(\rho_n) - \min_N(E(\rho_N))$')

        else:
            if 'gap' in pms:
                x, y = cback.i[1:cback._i], np.maximum(cback.E[1:cback._i, 0], ZERO)
                ax.set_title('Gap plot, energy = %.5f' % pms['E'], title_spec)

                pms['energy plot'].set_data(x, y)
                pms['ax2'] = ZERO, max(pms['ax2'][1], y[-1])
                ax.set_xlim(1, x[-1] + .5); ax.set_ylim(.95 * pms['ax2'][0], 1.05 * pms['ax2'][1])

            else:
                x, y = cback.i[:cback._i], cback.E[:cback._i, 0]
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
                pms['ax2'] = min(pms['ax2'][0], max(ZERO, Y[1]) if y.size > 1 else Y), max(pms['ax2'][1], y[-1])
                ax.set_xlim(1, x[-1] + .5); ax.set_ylim(.95 * pms['ax2'][0], 1.05 * pms['ax2'][1])

        if i < 2:
            fig.tight_layout()
        elif i > 40 and ax.get_xscale() != 'symlog':
            for ax in pms['axes'][1:]:
                ax.set_xscale('symlog', linthresh=40); ax.xaxis.set_minor_locator(MinorSymLogLocator())

        mv.update()
    return doPlot, mv
