from builtins import range
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .plotHandler import BasePlotter

#mag_sun = -27.1 # apparent r band magnitude of the sun. this sets the band for the magnitude limit.
# see http://www.ucolick.org/~cnaw/sun.html for apparent magnitudes in other bands.
mag_sun = -26.74 # apparent V band magnitude of the Sun (our H mags translate to V band)
km_per_au = 1.496e8
m_per_km = 1000


class MetricVsH(BasePlotter):
    """
    Plot metric values versus H.
    Marginalize over metric values in each H bin using 'npReduce'.
    """
    def __init__(self):
        self.plotType = 'MetricVsH'
        self.objectPlotter = False
        self.defaultPlotDict = {'title': None, 'xlabel': 'H (mag)', 'ylabel': None, 'label': None,
                                'npReduce': None, 'nbins': None, 'albedo': None,
                                'Hmark': None, 'HmarkLinestyle': ':', 'figsize': None}
        self.minHrange=1.0

    def __call__(self, metricValue, slicer, userPlotDict, fignum=None):
        if 'linestyle' not in userPlotDict:
            userPlotDict['linestyle'] = '-'
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)
        fig = plt.figure(fignum, figsize=plotDict['figsize'])
        Hvals = slicer.slicePoints['H']
        reduceFunc = plotDict['npReduce']
        if reduceFunc is None:
            reduceFunc = np.mean
        if Hvals.shape[0] == 1:
            # We have a simple set of values to plot against H.
            # This may be due to running a summary metric, such as completeness.
            mVals = metricValue[0].filled()
        elif len(Hvals) == slicer.shape[1]:
            # Using cloned H distribution.
            # Apply 'npReduce' method directly to metric values, and plot at matching H values.
            mVals = reduceFunc(metricValue.filled(), axis=0)
        else:
            # Probably each object has its own H value.
            hrange = Hvals.max() - Hvals.min()
            minH = Hvals.min()
            if hrange < self.minHrange:
                hrange = self.minHrange
                minH = Hvals.min() - hrange/2.0
            nbins = plotDict['nbins']
            if nbins is None:
                nbins = 30
            stepsize = hrange  / float(nbins)
            bins = np.arange(minH, minH + hrange + stepsize/2.0, stepsize)
            # In each bin of H, calculate the 'npReduce' value of the corresponding metricValues.
            inds = np.digitize(Hvals, bins)
            inds = inds-1
            mVals = np.zeros(len(bins), float)
            for i in range(len(bins)):
                match = metricValue[inds == i]
                if len(match) == 0:
                    mVals[i] = slicer.badval
                else:
                    mVals[i] = reduceFunc(match.filled())
            Hvals = bins
        plt.plot(Hvals, mVals, color=plotDict['color'], linestyle=plotDict['linestyle'],
                label=plotDict['label'])
        if 'xMin' in plotDict:
            plt.xlim(left = plotDict['xMin'])
        if 'xMax' in plotDict:
            plt.xlim(right = plotDict['xMax'])
        if 'yMin' in plotDict:
            plt.ylim(bottom = plotDict['yMin'])
        if 'yMax' in plotDict:
            plt.ylim(top = plotDict['yMax'])
        # Convert Hvals to diameter, using 'albedo'
        albedo = plotDict['albedo']
        y = 1.0
        if albedo is not None:
            ax = plt.axes()
            ax2 = ax.twiny()
            Hmin, Hmax = ax.get_xlim()
            dmax = 2.0 * np.sqrt(10**((mag_sun - Hmin - 2.5*np.log10(albedo))/2.5))
            dmin = 2.0 * np.sqrt(10**((mag_sun - Hmax - 2.5*np.log10(albedo))/2.5))
            dmax = dmax * km_per_au * m_per_km
            dmin = dmin * km_per_au * m_per_km
            ax2.set_xlim(dmax, dmin)
            ax2.set_xscale('log')
            ax2.set_xlabel('D (m)', labelpad=-10, horizontalalignment='right')
            ax2.grid(False)
            plt.sca(ax)
            y = 1.1
        plt.grid(True)
        if plotDict['Hmark'] is not None:
            plt.axvline(x=plotDict['Hmark'], color='r',
                        linestyle=plotDict['HmarkLinestyle'], alpha=0.3)
        plt.title(plotDict['title'], y=y)
        plt.xlabel(plotDict['xlabel'])
        plt.ylabel(plotDict['ylabel'])
        plt.tight_layout()
        return fig.number


class MetricVsOrbit(BasePlotter):
    """
    Plot metric values (at a particular H value) vs. orbital parameters.
    Marginalize over metric values in each orbital bin using 'npReduce'.
    """
    def __init__(self, xaxis='q', yaxis='e'):
        self.plotType = 'MetricVsOrbit_%s%s' %(xaxis, yaxis)
        self.objectPlotter = False
        self.defaultPlotDict = {'title': None, 'xlabel': xaxis, 'ylabel': yaxis,
                                'xaxis': xaxis, 'yaxis': yaxis,
                                'label': None, 'cmap': cm.viridis,
                                'npReduce': None,
                                'nxbins': None, 'nybins': None, 'levels': None,
                                'Hval': None, 'Hwidth': None, 'figsize': None}

    def __call__(self, metricValue, slicer, userPlotDict, fignum=None):
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)
        fig = plt.figure(fignum, figsize=plotDict['figsize'])
        xvals = slicer.slicePoints['orbits'][plotDict['xaxis']]
        yvals = slicer.slicePoints['orbits'][plotDict['yaxis']]
        # Set x/y bins.
        nxbins = plotDict['nxbins']
        nybins = plotDict['nybins']
        if nxbins is None:
            nxbins = 100
        if nybins is None:
            nybins = 100
        if 'xbins' in plotDict:
            xbins = plotDict['xbins']
        else:
            xbinsize = (xvals.max() - xvals.min())/float(nxbins)
            xbins = np.arange(xvals.min(), xvals.max() + xbinsize/2.0, xbinsize)
        if 'ybins' in plotDict:
            ybins = plotDict['ybins']
        else:
            ybinsize = (yvals.max() - yvals.min())/float(nybins)
            ybins = np.arange(yvals.min(), yvals.max() + ybinsize/2.0, ybinsize)
        nxbins = len(xbins)
        nybins = len(ybins)
        # Identify the relevant metricValues for the Hvalue we want to plot.
        Hvals = slicer.slicePoints['H']
        Hwidth = plotDict['Hwidth']
        if Hwidth is None:
            Hwidth = 1.0
        if len(Hvals) == slicer.shape[1]:
            if plotDict['Hval'] is None:
                Hidx = int(len(Hvals) / 2)
                Hval = Hvals[Hidx]
            else:
                Hval = plotDict['Hval']
                Hidx = np.where(np.abs(Hvals - Hval) == np.abs(Hvals - Hval).min())[0]
                Hidx = Hidx[0]
        else:
            if plotDict['Hval'] is None:
                Hval = np.median(Hvals)
                Hidx = np.where(np.abs(Hvals - Hval) <= Hwidth/2.0)[0]
            else:
                Hval = plotDict['Hvals']
                Hidx = np.where(np.abs(Hvals - Hval) <= Hwidth/2.0)[0]
        if len(Hvals) == slicer.shape[1]:
            mVals = np.swapaxes(metricValue, 1, 0)[Hidx].filled()
        else:
            mVals = metricValue[Hidx].filled()
        # Calculate the npReduce'd metric values at each x/y bin.
        if 'colorMin' in plotDict:
            badval = plotDict['colorMin'] - 1
        else:
            badval = slicer.badval
        binvals = np.zeros((nybins, nxbins), dtype='float') + badval
        xidxs = np.digitize(xvals, xbins) - 1
        yidxs = np.digitize(yvals, ybins) - 1
        reduceFunc = plotDict['npReduce']
        if reduceFunc is None:
            reduceFunc = np.mean
        for iy in range(nybins):
            ymatch = np.where(yidxs == iy)[0]
            for ix in range(nxbins):
                xmatch = np.where(xidxs[ymatch] == ix)[0]
                matchVals = mVals[ymatch][xmatch]
                if len(matchVals) > 0:
                    binvals[iy][ix] = reduceFunc(matchVals)
        xi, yi = np.meshgrid(xbins, ybins)
        if 'colorMin' in plotDict:
            vMin = plotDict['colorMin']
        else:
            vMin = binvals.min()
        if 'colorMax' in plotDict:
            vMax = plotDict['colorMax']
        else:
            vMax = binvals.max()
        nlevels = plotDict['levels']
        if nlevels is None:
            nlevels = 200
        levels = np.arange(vMin, vMax, (vMax-vMin)/float(nlevels))
        plt.contourf(xi, yi, binvals, levels, extend='max',
                     zorder=0, cmap=plotDict['cmap'])
        cbar = plt.colorbar()
        label = plotDict['label']
        if label is None:
            label = ''
        cbar.set_label(label + ' @ H=%.1f' %(Hval))
        plt.title(plotDict['title'])
        plt.xlabel(plotDict['xlabel'])
        plt.ylabel(plotDict['ylabel'])
        return fig.number

class MetricVsOrbitPoints(BasePlotter):
    """
    Plot metric values (at a particular H value) as function of orbital parameters,
    using points for each metric value.
    """
    def __init__(self, xaxis='q', yaxis='e'):
        self.plotType = 'MetricVsOrbit'
        self.objectPlotter = False
        self.defaultPlotDict = {'title': None, 'xlabel': xaxis, 'ylabel': yaxis,
                                'label': None, 'cmap': cm.viridis,
                                'xaxis': xaxis, 'yaxis': yaxis,
                                'Hval': None, 'Hwidth': None,
                                'foregroundPoints': True, 'backgroundPoints': False,
                                'figsize': None}

    def __call__(self, metricValue, slicer, userPlotDict, fignum=None):
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)
        fig = plt.figure(fignum, figsize=plotDict['figsize'])
        xvals = slicer.slicePoints['orbits'][plotDict['xaxis']]
        yvals = slicer.slicePoints['orbits'][plotDict['yaxis']]
        # Identify the relevant metricValues for the Hvalue we want to plot.
        Hvals = slicer.slicePoints['H']
        Hwidth = plotDict['Hwidth']
        if Hwidth is None:
            Hwidth = 1.0
        if len(Hvals) == slicer.shape[1]:
            if plotDict['Hval'] is None:
                Hidx = int(len(Hvals) / 2)
                Hval = Hvals[Hidx]
            else:
                Hval = plotDict['Hval']
                Hidx = np.where(np.abs(Hvals - Hval) == np.abs(Hvals - Hval).min())[0]
                Hidx = Hidx[0]
        else:
            if plotDict['Hval'] is None:
                Hval = np.median(Hvals)
                Hidx = np.where(np.abs(Hvals - Hval) <= Hwidth/2.0)[0]
            else:
                Hval = plotDict['Hvals']
                Hidx = np.where(np.abs(Hvals - Hval) <= Hwidth/2.0)[0]
        if len(Hvals) == slicer.shape[1]:
            mVals = np.swapaxes(metricValue, 1, 0)[Hidx]
        else:
            mVals = metricValue[Hidx]
        if 'colorMin' in plotDict:
            vMin = plotDict['colorMin']
        else:
            vMin = mVals.min()
        if 'colorMax' in plotDict:
            vMax = plotDict['colorMax']
        else:
            vMax = mVals.max()
        if plotDict['backgroundPoints']:
            # This isn't quite right for the condition .. but will do for now.
            condition = np.where(mVals == 0)
            plt.plot(xvals[condition], yvals[condition], 'r.', markersize=4, alpha=0.5, zorder=3)
        if plotDict['foregroundPoints']:
            plt.scatter(xvals, yvals, c=mVals, vmin=vMin, vmax=vMax,
                        cmap=plotDict['cmap'], s=15, alpha=0.8, zorder=0)
            cbar = plt.colorbar()
            label = plotDict['label']
            if label is None:
                label = ''
        cbar.set_label(label + ' @ H=%.1f' %(Hval))
        plt.title(plotDict['title'])
        plt.xlabel(plotDict['xlabel'])
        plt.ylabel(plotDict['ylabel'])
        return fig.number
