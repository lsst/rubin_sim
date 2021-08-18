from builtins import zip
import numpy as np
import matplotlib.pyplot as plt
from rubin_sim.maf.utils import percentileClipping

from .plotHandler import BasePlotter

__all__ = ['OneDBinnedData']

class OneDBinnedData(BasePlotter):
    def __init__(self):
        self.plotType = 'BinnedData'
        self.objectPlotter = False
        self.defaultPlotDict = {'title': None, 'label': None, 'xlabel': None, 'ylabel': None,
                                'filled': False, 'alpha': 0.5, 'linestyle': '-', 'linewidth': 1,
                                'logScale': False, 'percentileClip': None,
                                'xMin': None, 'xMax': None, 'yMin': None, 'yMax': None,
                                'fontsize': None, 'figsize': None, 'grid': True}

    def __call__(self, metricValues, slicer, userPlotDict, fignum=None):
        """
        Plot a set of oneD binned metric data.
        """
        if slicer.slicerName != 'OneDSlicer':
            raise ValueError('OneDBinnedData plotter is for use with OneDSlicer')
        if 'bins' not in slicer.slicePoints:
            errMessage = 'OneDSlicer must contain "bins" in slicePoints metadata.'
            errMessage += ' SlicePoints only contains keys %s.' % (slicer.slicePoints.keys())
            raise ValueError(errMessage)
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)
        fig = plt.figure(fignum, figsize=plotDict['figsize'])
        # Plot the histogrammed data.
        leftedge = slicer.slicePoints['bins'][:-1]
        width = np.diff(slicer.slicePoints['bins'])
        if plotDict['filled']:
            plt.bar(leftedge, metricValues.filled(), width, label=plotDict['label'],
                    linewidth=0, alpha=plotDict['alpha'], log=plotDict['logScale'],
                    color=plotDict['color'])
        else:
            good = np.where(metricValues.mask == False)
            x = np.ravel(list(zip(leftedge[good], leftedge[good] + width[good])))
            y = np.ravel(list(zip(metricValues[good], metricValues[good])))
            if plotDict['logScale']:
                plt.semilogy(x, y, label=plotDict['label'], color=plotDict['color'],
                             linestyle=plotDict['linestyle'], linewidth=plotDict['linewidth'],
                             alpha=plotDict['alpha'])
            else:
                plt.plot(x, y, label=plotDict['label'], color=plotDict['color'],
                         linestyle=plotDict['linestyle'], linewidth=plotDict['linewidth'],
                         alpha=plotDict['alpha'])
        if 'ylabel' in plotDict:
            plt.ylabel(plotDict['ylabel'], fontsize=plotDict['fontsize'])
        if 'xlabel' in plotDict:
            plt.xlabel(plotDict['xlabel'], fontsize=plotDict['fontsize'])
        # Set y limits (either from values in args, percentileClipping or compressed data values).
        if plotDict['percentileClip'] is not None:
            yMin, yMax = percentileClipping(metricValues.compressed(),
                                            percentile=plotDict['percentileClip'])
            if plotDict['yMin'] is None:
                plotDict['yMin'] = yMin
            if plotDict['yMax'] is None:
                plotDict['yMax'] = yMax

        if plotDict['grid']:
            plt.grid(plotDict['grid'], alpha=0.3)

        if plotDict['yMin'] is None and metricValues.filled().min() == 0:
            plotDict['yMin'] = 0

        # Set y and x limits, if provided.
        if plotDict['yMin'] is not None:
            plt.ylim(bottom=plotDict['yMin'])
        if plotDict['yMax'] is not None:
            plt.ylim(top=plotDict['yMax'])
        if plotDict['xMin'] is not None:
            plt.xlim(left=plotDict['xMin'])
        if plotDict['xMax'] is not None:
            plt.xlim(right=plotDict['xMax'])
        plt.title(plotDict['title'])
        return fig.number
