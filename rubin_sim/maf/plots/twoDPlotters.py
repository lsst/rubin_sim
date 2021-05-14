import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from .plotHandler import BasePlotter
from .perceptual_rainbow import makePRCmap

__all__ = ['TwoDMap', 'VisitPairsHist']

perceptual_rainbow = makePRCmap()

class TwoDMap(BasePlotter):
    def __init__(self):
        self.plotType = 'TwoD'
        self.objectPlotter = False
        self.defaultPlotDict = {'title': None, 'xlabel': None, 'ylabel': None, 'label': None,
                                'logScale': False, 'cbarFormat': None, 'cbarTitle': 'Count',
                                'cmap': perceptual_rainbow,
                                'percentileClip': None, 'colorMin': None, 'colorMax': None,
                                'zp': None, 'normVal': None,
                                'cbar_edge': True, 'nTicks': None, 'aspect': 'auto',
                                'xextent': None, 'origin': None}

    def __call__(self, metricValue, slicer, userPlotDict, fignum=None):
        """
        Parameters
        ----------
        metricValue : numpy.ma.MaskedArray
        slicer : rubin_sim.maf.slicers.BaseSpatialSlicer
            (any spatial slicer)
        userPlotDict: dict
            Dictionary of plot parameters set by user (overrides default values).
        fignum : int
            Matplotlib figure number to use (default = None, starts new figure).

        Returns
        -------
        int
           Matplotlib figure number used to create the plot.
        """
        if 'Healpix' in slicer.slicerName:
            self.defaultPlotDict['ylabel'] = 'Healpix ID'
        elif 'Opsim' in slicer.slicerName:
            self.defaultPlotDict['ylabel'] = 'Field ID'
            self.defaultPlotDict['origin'] = 'lower'
        elif 'User' in slicer.slicerName:
            self.defaultPlotDict['ylabel'] = 'User Field ID'

        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        # Don't clobber with None
        for key in userPlotDict:
            if userPlotDict[key] is not None:
                plotDict[key] = userPlotDict[key]

        if plotDict['xextent'] is None:
            plotDict['xextent'] = [0, metricValue[0, :].size]

        if plotDict['logScale']:
            norm = colors.LogNorm()
        else:
            norm = None

        # Mask out values below the color minimum so they show up as white
        if plotDict['colorMin'] is not None:
            lowVals = np.where(metricValue.data < plotDict['colorMin'])
            metricValue.mask[lowVals] = True

        figure = plt.figure(fignum)
        ax = figure.add_subplot(111)
        yextent = slicer.spatialExtent
        xextent = plotDict['xextent']
        extent = []
        extent.extend(xextent)
        extent.extend(yextent)
        image = ax.imshow(metricValue, vmin=plotDict['colorMin'], vmax=plotDict['colorMax'],
                          aspect=plotDict['aspect'], cmap=plotDict['cmap'], norm=norm,
                          extent=extent,
                          interpolation='none', origin=plotDict['origin'])
        cb = plt.colorbar(image)

        ax.set_xlabel(plotDict['xlabel'])
        ax.set_ylabel(plotDict['ylabel'])
        ax.set_title(plotDict['title'])
        cb.set_label(plotDict['cbarTitle'])

        # Fix white space on pdf's
        if plotDict['cbar_edge']:
            cb.solids.set_edgecolor("face")
        return figure.number

class VisitPairsHist(BasePlotter):
    """
    Given an opsim2dSlicer, figure out what fraction of observations are in singles, pairs, triples, etc.
    """
    def __init__(self):
        self.plotType = 'TwoD'
        self.objectPlotter = False
        self.defaultPlotDict = {'title': None, 'xlabel': 'N visits per night per field',
                                'ylabel': 'N Visits', 'label': None, 'color': 'b',
                                'xlim': [0, 20], 'ylim': None}

    def __call__(self, metricValue, slicer, userPlotDict, fignum=None):
        """
        Parameters
        ----------
        metricValue : numpy.ma.MaskedArray
        slicer : rubin_sim.maf.slicers.TwoDSlicer
        userPlotDict: dict
            Dictionary of plot parameters set by user (overrides default values).
        fignum : int
            Matplotlib figure number to use (default = None, starts new figure).

        Returns
        -------
        int
           Matplotlib figure number used to create the plot.
        """
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        # Don't clobber with None
        for key in userPlotDict:
            if userPlotDict[key] is not None:
                plotDict[key] = userPlotDict[key]

        maxVal = metricValue.max()
        bins = np.arange(0.5, maxVal + 0.5, 1)

        vals, bins = np.histogram(metricValue, bins)
        xvals = (bins[:-1] + bins[1:]) / 2.

        figure = plt.figure(fignum)
        ax = figure.add_subplot(111)
        ax.bar(xvals, vals * xvals, color=plotDict['color'], label=plotDict['label'])
        ax.set_xlabel(plotDict['xlabel'])
        ax.set_ylabel(plotDict['ylabel'])
        ax.set_title(plotDict['title'])
        ax.set_xlim(plotDict['xlim'])
        ax.set_ylim(plotDict['ylim'])

        return figure.number
