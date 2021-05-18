from builtins import zip
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from .plotHandler import BasePlotter
from .perceptual_rainbow import makePRCmap

__all__ = ['TwoDSubsetData', 'OneDSubsetData']

perceptual_rainbow = makePRCmap()

class TwoDSubsetData(BasePlotter):
    """
    Plot 2 axes from the slicer.sliceColList, identified by
    plotDict['xaxis']/['yaxis'], given the metricValues at all
    slicepoints [sums over non-visible axes].
    """
    def __init__(self):
        self.plotType = '2DBinnedData'
        self.objectPlotter = False
        self.defaultPlotDict = {'title': None, 'xlabel': None, 'ylable': None, 'units': None,
                                'logScale': False, 'clims': None, 'cmap': perceptual_rainbow,
                                'cbarFormat': None}

    def __call__(self, metricValues, slicer, userPlotDict, fignum=None):
        """
        Parameters
        ----------
        metricValue : numpy.ma.MaskedArray
        slicer : rubin_sim.maf.slicers.NDSlicer
        userPlotDict: dict
            Dictionary of plot parameters set by user (overrides default values).
            'xaxis' and 'yaxis' values define which axes of the nd data to plot along the x/y axes.
        fignum : int
            Matplotlib figure number to use (default = None, starts new figure).

        Returns
        -------
        int
           Matplotlib figure number used to create the plot.
        """
        if slicer.slicerName != 'NDSlicer':
            raise ValueError('TwoDSubsetData plots ndSlicer metric values')
        fig = plt.figure(fignum)
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)
        if 'xaxis' not in plotDict or 'yaxis' not in plotDict:
            raise ValueError('xaxis and yaxis must be specified in plotDict')
        xaxis = plotDict['xaxis']
        yaxis = plotDict['yaxis']
        # Reshape the metric data so we can isolate the values to plot
        # (just new view of data, not copy).
        newshape = []
        for b in slicer.bins:
            newshape.append(len(b) - 1)
        newshape.reverse()
        md = metricValues.reshape(newshape)
        # Sum over other dimensions. Note that masked values are not included in sum.
        sumaxes = list(range(slicer.nD))
        sumaxes.remove(xaxis)
        sumaxes.remove(yaxis)
        sumaxes = tuple(sumaxes)
        md = md.sum(sumaxes)
        # Plot the histogrammed data.
        # Plot data.
        x, y = np.meshgrid(slicer.bins[xaxis][:-1], slicer.bins[yaxis][:-1])
        if plotDict['logScale']:
            norm = colors.LogNorm()
        else:
            norm = None
        if plotDict['clims'] is None:
            im = plt.contourf(x, y, md, 250, norm=norm, extend='both', cmap=plotDict['cmap'])
        else:
            im = plt.contourf(x, y, md, 250, norm=norm, extend='both', cmap=plotDict['cmap'],
                              vmin=plotDict['clims'][0], vmax=plotDict['clims'][1])
        xlabel = plotDict['xlabel']
        if xlabel is None:
            xlabel = slicer.sliceColList[xaxis]
        plt.xlabel(xlabel)
        ylabel = plotDict['ylabel']
        if ylabel is None:
            ylabel = slicer.sliceColList[yaxis]
        plt.ylabel(ylabel)
        cb = plt.colorbar(im, aspect=25, extend='both',
                          orientation='horizontal', format=plotDict['cbarFormat'])
        cb.set_label(plotDict['units'])
        plt.title(plotDict['title'])
        return fig.number


class OneDSubsetData(BasePlotter):
    """
    Plot a single axes from the sliceColList, identified by plotDict['axis'],
    given the metricValues at all slicepoints [sums over non-visible axes].
    """
    def __init__(self):
        self.plotType = '1DBinnedData'
        self.objectPlotter = False
        self.defaultPlotDict = {'title': None, 'xlabel': None, 'ylabel': None, 'label': None, 'units': None,
                                'logScale': False, 'histRange': None, 'filled': False, 'alpha': 0.5,
                                'cmap': perceptual_rainbow, 'cbarFormat': None}

    def plotBinnedData1D(self, metricValues, slicer, userPlotDict, fignum=None):
        """
        Parameters
        ----------
        metricValue : numpy.ma.MaskedArray
        slicer : rubin_sim.maf.slicers.NDSlicer
        userPlotDict: dict
            Dictionary of plot parameters set by user (overrides default values).
            'axis' keyword identifies which axis to show in the plot (along xaxis of plot).
        fignum : int
            Matplotlib figure number to use (default = None, starts new figure).

        Returns
        -------
        int
           Matplotlib figure number used to create the plot.
        """
        if slicer.slicerName != 'NDSlicer':
            raise ValueError('TwoDSubsetData plots ndSlicer metric values')
        fig = plt.figure(fignum)
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)
        if 'axis' not in plotDict:
            raise ValueError('axis for 1-d plot must be specified in plotDict')
        # Reshape the metric data so we can isolate the values to plot
        # (just new view of data, not copy).
        axis = plotDict['axis']
        newshape = []
        for b in slicer.bins:
            newshape.append(len(b) - 1)
        newshape.reverse()
        md = metricValues.reshape(newshape)
        # Sum over other dimensions. Note that masked values are not included in sum.
        sumaxes = list(range(slicer.nD))
        sumaxes.remove(axis)
        sumaxes = tuple(sumaxes)
        md = md.sum(sumaxes)
        # Plot the histogrammed data.
        leftedge = slicer.bins[axis][:-1]
        width = np.diff(slicer.bins[axis])
        if plotDict['filled']:
            plt.bar(leftedge, md, width, label=plotDict['label'],
                    linewidth=0, alpha=plotDict['alpha'], log=plotDict['logScale'])
        else:
            x = np.ravel(list(zip(leftedge, leftedge + width)))
            y = np.ravel(list(zip(md, md)))
            if plotDict['logScale']:
                plt.semilogy(x, y, label=plotDict['label'])
            else:
                plt.plot(x, y, label=plotDict['label'])
        plt.ylabel(plotDict['ylabel'])
        xlabel = plotDict['xlabel']
        if xlabel is None:
            xlabel = slicer.sliceColName[axis]
            if plotDict['units'] != None:
                xlabel += ' (' + plotDict['units'] + ')'
        plt.xlabel(xlabel)
        if (plotDict['histRange'] != None):
            plt.xlim(plotDict['histRange'])
        plt.title(plotDict['title'])
        return fig.number
