from builtins import zip
import numbers
import copy
import numpy as np
import warnings
import healpy as hp
from matplotlib import colors
from matplotlib import ticker
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection

from rubin_sim.maf.utils import optimalBins, percentileClipping
from .plotHandler import BasePlotter, applyZPNorm

from rubin_sim.utils import _equatorialFromGalactic, _healbin
from .perceptual_rainbow import makePRCmap
perceptual_rainbow = makePRCmap()
import numpy.ma as ma

__all__ = ['setColorLims', 'setColorMap', 'HealpixSkyMap', 'HealpixPowerSpectrum',
           'HealpixHistogram', 'OpsimHistogram', 'BaseHistogram',
           'BaseSkyMap', 'HealpixSDSSSkyMap', 'LambertSkyMap']

baseDefaultPlotDict = {'title': None, 'xlabel': None, 'label': None,
                       'logScale': False, 'percentileClip': None, 'normVal': None, 'zp': None,
                       'cbarFormat': None, 'cmap': perceptual_rainbow, 'cbar_edge': True, 'nTicks': 10,
                       'colorMin': None, 'colorMax': None,
                       'xMin': None, 'xMax': None, 'yMin': None, 'yMax': None,
                       'labelsize': None, 'fontsize': None, 'figsize': None, 'subplot': 111,
                       'maskBelow': None}


def setColorLims(metricValue, plotDict):
    """Set up color bar limits."""
    # Use plot dict if these values are set.
    colorMin = plotDict['colorMin']
    colorMax = plotDict['colorMax']
    # If not, try to use percentile clipping.
    if (plotDict['percentileClip'] is not None) & (np.size(metricValue.compressed()) > 0):
        pcMin, pcMax = percentileClipping(metricValue.compressed(), percentile=plotDict['percentileClip'])
        if colorMin is None:
            colorMin = pcMin
        if colorMax is None:
            colorMax = pcMax
    # If not, just use the data limits.
    if colorMin is None:
        colorMin = metricValue.compressed().min()
    if colorMax is None:
        colorMax = metricValue.compressed().max()
    # But make sure there is some range on the colorbar
    if colorMin == colorMax:
        colorMin = colorMin - 0.5
        colorMax = colorMax + 0.5
    return np.sort([colorMin, colorMax])


def setColorMap(plotDict):
    cmap = plotDict['cmap']
    if cmap is None:
        cmap = 'perceptual_rainbow'
    if type(cmap) == str:
        cmap = getattr(cm, cmap)
    # Set background and masked pixel colors default healpy white and gray.
    cmap = copy.copy(cmap)
    cmap.set_over(cmap(1.0))
    cmap.set_under('w')
    cmap.set_bad('gray')
    return cmap


class HealpixSkyMap(BasePlotter):
    """
    Generate a sky map of healpix metric values using healpy's mollweide view.
    """
    def __init__(self):
        super(HealpixSkyMap, self).__init__()
        # Set the plotType
        self.plotType = 'SkyMap'
        self.objectPlotter = False
        # Set up the default plotting parameters.
        self.defaultPlotDict = {}
        self.defaultPlotDict.update(baseDefaultPlotDict)
        self.defaultPlotDict.update({'rot': (0, 0, 0), 'flip': 'astro', 'coord': 'C',
                                     'nside': 8, 'reduceFunc': np.mean,
                                     'visufunc': hp.mollview})
        # Note: for alt/az sky maps using the healpix plotter, you can use
        # {'rot': (90, 90, 90), 'flip': 'geo'}
        self.healpy_visufunc_params = {}
        self.ax = None
        self.im = None

    def __call__(self, metricValueIn, slicer, userPlotDict, fignum=None):
        """
        Parameters
        ----------
        metricValue : numpy.ma.MaskedArray
        slicer : rubin_sim.maf.slicers.HealpixSlicer
        userPlotDict: dict
            Dictionary of plot parameters set by user (overrides default values).
        fignum : int
            Matplotlib figure number to use (default = None, starts new figure).

        Returns
        -------
        int
           Matplotlib figure number used to create the plot.
        """
        # Override the default plotting parameters with user specified values.
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)

        self.healpy_visufunc = plotDict['visufunc']

        # Check if we have a valid HEALpix slicer
        if 'Heal' in slicer.slicerName:
            # Update the metric data with zeropoint or normalization.
            metricValue = applyZPNorm(metricValueIn, plotDict)
        else:
            # Bin the values up on a healpix grid.
            metricValue = _healbin(slicer.slicePoints['ra'], slicer.slicePoints['dec'],
                                   metricValueIn.filled(slicer.badval), nside=plotDict['nside'],
                                   reduceFunc=plotDict['reduceFunc'], fillVal=slicer.badval)
            mask = np.zeros(metricValue.size)
            mask[np.where(metricValue == slicer.badval)] = 1
            metricValue = ma.array(metricValue, mask=mask)
            metricValue = applyZPNorm(metricValue, plotDict)

        if plotDict['maskBelow'] is not None:
            toMask = np.where(metricValue <= plotDict['maskBelow'])[0]
            metricValue.mask[toMask] = True
            badval = hp.UNSEEN
        else:
            badval = slicer.badval

        # Generate a full-sky plot.
        fig = plt.figure(fignum, figsize=plotDict['figsize'])
        # Set up color bar limits.
        clims = setColorLims(metricValue, plotDict)
        cmap = setColorMap(plotDict)
        # Set log scale?
        norm = None
        if plotDict['logScale']:
            norm = 'log'
        # Avoid trying to log scale when zero is in the range.
        if (norm == 'log') & ((clims[0] <= 0 <= clims[1]) or (clims[0] >= 0 >= clims[1])):
            # Try something simple
            above = metricValue[np.where(metricValue > 0)]
            if len(above) > 0:
                clims[0] = above.max()
            # If still bad, give up and turn off norm
            if ((clims[0] <= 0 <= clims[1]) or (clims[0] >= 0 >= clims[1])):
                norm = None
            warnings.warn("Using norm was set to log, but color limits pass through 0. "
                          "Adjusting so plotting doesn't fail")
        if plotDict['coord'] == 'C':
            notext = True
        else:
            notext = False

        visufunc_params = {'title': plotDict['title'],
                           'cbar': False,
                           'min': clims[0],
                           'max': clims[1],
                           'rot': plotDict['rot'],
                           'flip': plotDict['flip'],
                           'coord': plotDict['coord'],
                           'cmap': cmap,
                           'norm': norm,
                           'sub': plotDict['subplot'],
                           'fig':fig.number,
                           'notext': notext}
        # Keys to specify only if present in plotDict
        for key in ('reso', 'lamb', 'reuse_axes', 'alpha', 'badcolor', 'bgcolor'):
            if key in plotDict:
                visufunc_params[key] = plotDict[key]
        
        visufunc_params.update(self.healpy_visufunc_params)
        self.healpy_visufunc(metricValue.filled(badval), **visufunc_params)

        # Add a graticule (grid) over the globe.
        hp.graticule(dpar=30, dmer=30)
        # Add colorbar (not using healpy default colorbar because we want more tickmarks).
        self.ax = plt.gca()
        im = self.ax.get_images()[0]
        # Add label.
        if plotDict['label'] is not None:
            plt.figtext(0.8, 0.8, '%s' % (plotDict['label']))
        # Make a color bar. Supress silly colorbar warnings.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cb = plt.colorbar(im, shrink=0.75, aspect=25, pad=0.1, orientation='horizontal',
                              format=plotDict['cbarFormat'], extendrect=True)
            cb.set_label(plotDict['xlabel'], fontsize=plotDict['fontsize'])
            if plotDict['labelsize'] is not None:
                cb.ax.tick_params(labelsize=plotDict['labelsize'])
            if norm == 'log':
                tick_locator = ticker.LogLocator(numticks=plotDict['nTicks'])
                cb.locator = tick_locator
                cb.update_ticks()
            if (plotDict['nTicks'] is not None) & (norm != 'log'):
                tick_locator = ticker.MaxNLocator(nbins=plotDict['nTicks'])
                cb.locator = tick_locator
                cb.update_ticks()
        # If outputing to PDF, this fixes the colorbar white stripes
        if plotDict['cbar_edge']:
            cb.solids.set_edgecolor("face")
        return fig.number

    
class HealpixPowerSpectrum(BasePlotter):
    def __init__(self):
        self.plotType = 'PowerSpectrum'
        self.objectPlotter = False
        self.defaultPlotDict = {}
        self.defaultPlotDict.update(baseDefaultPlotDict)
        self.defaultPlotDict.update({'maxl': None, 'removeDipole': True, 'linestyle': '-'})

    def __call__(self, metricValue, slicer, userPlotDict, fignum=None):
        """
        Generate and plot the power spectrum of metricValue (calculated on a healpix grid).
        """
        if 'Healpix' not in slicer.slicerName:
            raise ValueError('HealpixPowerSpectrum for use with healpix metricBundles.')
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)

        fig = plt.figure(fignum, figsize=plotDict['figsize'])
        if plotDict['subplot'] != '111':
            ax = fig.add_subplot(plotDict['subplot'])
        # If the mask is True everywhere (no data), just plot zeros
        if False not in metricValue.mask:
            return None
        if plotDict['removeDipole']:
            cl = hp.anafast(hp.remove_dipole(metricValue.filled(slicer.badval)), lmax=plotDict['maxl'])
        else:
            cl = hp.anafast(metricValue.filled(slicer.badval), lmax=plotDict['maxl'])
        ell = np.arange(np.size(cl))
        if plotDict['removeDipole']:
            condition = (ell > 1)
        else:
            condition = (ell > 0)
        ell = ell[condition]
        cl = cl[condition]
        # Plot the results.
        plt.plot(ell, (cl * ell * (ell + 1)) / 2.0 / np.pi,
                 color=plotDict['color'], linestyle=plotDict['linestyle'], label=plotDict['label'])
        if cl.max() > 0 and plotDict['logScale']:
            plt.yscale('log')
        plt.xlabel(r'$l$', fontsize=plotDict['fontsize'])
        plt.ylabel(r'$l(l+1)C_l/(2\pi)$', fontsize=plotDict['fontsize'])
        if plotDict['labelsize'] is not None:
            plt.tick_params(axis='x', labelsize=plotDict['labelsize'])
            plt.tick_params(axis='y', labelsize=plotDict['labelsize'])
        if plotDict['title'] is not None:
            plt.title(plotDict['title'])
        # Return figure number (so we can reuse/add onto/save this figure if desired).
        return fig.number


class HealpixHistogram(BasePlotter):
    def __init__(self):
        self.plotType = 'Histogram'
        self.objectPlotter = False
        self.defaultPlotDict = {}
        self.defaultPlotDict.update(baseDefaultPlotDict)
        self.defaultPlotDict.update({'ylabel': 'Area (1000s of square degrees)',
                                     'bins': None, 'binsize': None, 'cumulative': False,
                                     'scale': None, 'linestyle': '-'})
        self.baseHist = BaseHistogram()

    def __call__(self, metricValue, slicer, userPlotDict, fignum=None):
        """
        Histogram metricValue for all healpix points.
        """
        if 'Healpix' not in slicer.slicerName:
            raise ValueError('HealpixHistogram is for use with healpix slicer.')
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)
        if plotDict['scale'] is None:
            plotDict['scale'] = (hp.nside2pixarea(slicer.nside, degrees=True) / 1000.0)
        fignum = self.baseHist(metricValue, slicer, plotDict, fignum=fignum)
        return fignum


class OpsimHistogram(BasePlotter):
    def __init__(self):
        self.plotType = 'Histogram'
        self.objectPlotter = False
        self.defaultPlotDict = {}
        self.defaultPlotDict.update(baseDefaultPlotDict)
        self.defaultPlotDict.update({'ylabel': 'Number of Fields', 'yaxisformat': '%d',
                                     'bins': None, 'binsize': None, 'cumulative': False,
                                     'scale': 1.0, 'linestyle': '-'})
        self.baseHist = BaseHistogram()

    def __call__(self, metricValue, slicer, userPlotDict, fignum=None):
        """
        Histogram metricValue for all healpix points.
        """
        if slicer.slicerName != 'OpsimFieldSlicer':
            raise ValueError('OpsimHistogram is for use with OpsimFieldSlicer.')
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)
        fignum = self.baseHist(metricValue, slicer, plotDict, fignum=fignum)
        return fignum


class BaseHistogram(BasePlotter):
    def __init__(self):
        self.plotType = 'Histogram'
        self.objectPlotter = False
        self.defaultPlotDict = {}
        self.defaultPlotDict.update(baseDefaultPlotDict)
        self.defaultPlotDict.update({'ylabel': 'Count', 'bins': None, 'binsize': None, 'cumulative': False,
                                     'scale': 1.0, 'yaxisformat': '%.3f', 'linestyle': '-'})

    def __call__(self, metricValueIn, slicer, userPlotDict, fignum=None):
        """
        Plot a histogram of metricValues (such as would come from a spatial slicer).
        """
        # Adjust metric values by zeropoint or normVal, and use 'compressed' version of masked array.
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)
        metricValue = applyZPNorm(metricValueIn, plotDict)
        metricValue = metricValue.compressed()
        # Toss any NaNs or infs
        metricValue = metricValue[np.isfinite(metricValue)]
        # Determine percentile clipped X range, if set. (and xmin/max not set).
        if plotDict['xMin'] is None and plotDict['xMax'] is None:
            if plotDict['percentileClip']:
                plotDict['xMin'], plotDict['xMax'] = percentileClipping(metricValue,
                                                                        percentile=plotDict['percentileClip'])
        # Set the histogram range values, to avoid cases of trying to histogram single-valued data.
        # First we try to use the range specified by a user, if there is one. Then use the data if not.
        # all of this only works if plotDict is not cumulative.
        histRange = [plotDict['xMin'], plotDict['xMax']]
        if histRange[0] is None:
            histRange[0] = metricValue.min()
        if histRange[1] is None:
            histRange[1] = metricValue.max()
        # Need to have some range of values on the histogram, or it will fail.
        if histRange[0] == histRange[1]:
            warnings.warn('Histogram range was single-valued; expanding default range.')
            histRange[1] = histRange[0] + 1.0
        # Set up the bins for the histogram. User specified 'bins' overrides 'binsize'.
        # Note that 'bins' could be a single number or an array, simply passed to plt.histogram.
        if plotDict['bins'] is not None:
            bins = plotDict['bins']
        elif plotDict['binsize'] is not None:
            #  If generating a cumulative histogram, want to use full range of data (but with given binsize).
            #    .. but if user set histRange to be wider than full range of data, then
            #       extend bins to cover this range, so we can make prettier plots.
            if plotDict['cumulative']:
                if plotDict['xMin'] is not None:
                    # Potentially, expand the range for the cumulative histogram.
                    bmin = np.min([metricValue.min(), plotDict['xMin']])
                else:
                    bmin = metricValue.min()
                if plotDict['xMax'] is not None:
                    bmax = np.max([metricValue.max(), plotDict['xMax']])
                else:
                    bmax = metricValue.max()
                bins = np.arange(bmin, bmax + plotDict['binsize'] / 2.0, plotDict['binsize'])
            #  Otherwise, not cumulative so just use metric values, without potential expansion.
            else:
                bins = np.arange(histRange[0], histRange[1] + plotDict['binsize'] / 2.0, plotDict['binsize'])
            # Catch edge-case where there is only 1 bin value
            if bins.size < 2:
                bins = np.arange(bins.min() - plotDict['binsize'] * 2.0,
                                 bins.max() + plotDict['binsize'] * 2.0, plotDict['binsize'])
        else:
            # If user did not specify bins or binsize, then we try to figure out a good number of bins.
            bins = optimalBins(metricValue)
        # Generate plots.
        fig = plt.figure(fignum, figsize=plotDict['figsize'])
        if plotDict['subplot'] != 111 and plotDict['subplot'] != (1,1,1) and plotDict['subplot'] is not None:
            ax = fig.add_subplot(plotDict['subplot'])
        else:
            ax = plt.gca()
        # Check if any data falls within histRange, because otherwise histogram generation will fail.
        if isinstance(bins, np.ndarray):
            condition = ((metricValue >= bins.min()) & (metricValue <= bins.max()))
        else:
            condition = ((metricValue >= histRange[0]) & (metricValue <= histRange[1]))
        plotValue = metricValue[condition]
        if len(plotValue) == 0:
            # No data is within histRange/bins. So let's just make a simple histogram anyway.
            n, b, p = plt.hist(metricValue, bins=50, histtype='step', cumulative=plotDict['cumulative'],
                               log=plotDict['logScale'], label=plotDict['label'],
                               color=plotDict['color'])
        else:
            # There is data to plot, and we've already ensured histRange/bins are more than single value.
            n, b, p = plt.hist(metricValue, bins=bins, range=histRange,
                               histtype='step', log=plotDict['logScale'],
                               cumulative=plotDict['cumulative'],
                               label=plotDict['label'], color=plotDict['color'])
        hist_ylims = plt.ylim()
        if n.max() > hist_ylims[1]:
            plt.ylim(top = n.max())
        if n.min() < hist_ylims[0] and not plotDict['logScale']:
            plt.ylim(bottom = n.min())
        # Fill in axes labels and limits.
        # Option to use 'scale' to turn y axis into area or other value.

        def mjrFormatter(y, pos):
            if not isinstance(plotDict['scale'], numbers.Number):
                raise ValueError('plotDict["scale"] must be a number to scale the y axis.')
            return plotDict['yaxisformat'] % (y * plotDict['scale'])

        ax.yaxis.set_major_formatter(FuncFormatter(mjrFormatter))
        # Set optional x, y limits.
        if 'xMin' in plotDict:
            plt.xlim(left=plotDict['xMin'])
        if 'xMax' in plotDict:
            plt.xlim(right=plotDict['xMax'])
        if 'yMin' in plotDict:
            plt.ylim(bottom=plotDict['yMin'])
        if 'yMax' in plotDict:
            plt.ylim(top=plotDict['yMax'])
        # Set/Add various labels.
        plt.xlabel(plotDict['xlabel'], fontsize=plotDict['fontsize'])
        plt.ylabel(plotDict['ylabel'], fontsize=plotDict['fontsize'])
        plt.title(plotDict['title'])
        if plotDict['labelsize'] is not None:
            plt.tick_params(axis='x', labelsize=plotDict['labelsize'])
            plt.tick_params(axis='y', labelsize=plotDict['labelsize'])
        # Return figure number
        return fig.number


class BaseSkyMap(BasePlotter):
    def __init__(self):
        self.plotType = 'SkyMap'
        self.objectPlotter = False  # unless 'metricIsColor' is true..
        self.defaultPlotDict = {}
        self.defaultPlotDict.update(baseDefaultPlotDict)
        self.defaultPlotDict.update({'projection': 'aitoff', 'radius': np.radians(1.75), 'alpha': 1.0,
                                     'plotMask': False, 'metricIsColor': False, 'cbar': True,
                                     'raCen': 0.0, 'mwZone': True, 'bgcolor': 'gray'})

    def _plot_tissot_ellipse(self, lon, lat, radius, ax=None, **kwargs):
        """Plot Tissot Ellipse/Tissot Indicatrix

        Parameters
        ----------
        lon : float or array_like
        longitude-like of ellipse centers (radians)
        lat : float or array_like
        latitude-like of ellipse centers (radians)
        radius : float or array_like
        radius of ellipses (radians)
        ax : Axes object (optional)
        matplotlib axes instance on which to draw ellipses.

        Other Parameters
        ----------------
        other keyword arguments will be passed to matplotlib.patches.Ellipse.

        # The code in this method adapted from astroML, which is BSD-licensed.
        # See http: //github.com/astroML/astroML for details.
        """
        # Code adapted from astroML, which is BSD-licensed.
        # See http: //github.com/astroML/astroML for details.
        ellipses = []
        if ax is None:
            ax = plt.gca()
        for l, b, diam in np.broadcast(lon, lat, radius * 2.0):
            el = Ellipse((l, b), diam / np.cos(b), diam, **kwargs)
            ellipses.append(el)
        return ellipses

    def _plot_ecliptic(self, raCen=0, ax=None):
        """
        Plot a red line at location of ecliptic.
        """
        if ax is None:
            ax = plt.gca()
        ecinc = 23.439291 * (np.pi / 180.0)
        ra_ec = np.arange(0, np.pi * 2., (np.pi * 2. / 360.))
        dec_ec = np.sin(ra_ec) * ecinc
        lon = -(ra_ec - raCen - np.pi) % (np.pi * 2) - np.pi
        ax.plot(lon, dec_ec, 'r.', markersize=1.8, alpha=0.4)

    def _plot_mwZone(self, raCen=0, peakWidth=np.radians(10.), taperLength=np.radians(80.), ax=None):
        """
        Plot blue lines to mark the milky way galactic exclusion zone.
        """
        if ax is None:
            ax = plt.gca()
        # Calculate galactic coordinates for mw location.
        step = 0.02
        galL = np.arange(-np.pi, np.pi + step / 2., step)
        val = peakWidth * np.cos(galL / taperLength * np.pi / 2.)
        galB1 = np.where(np.abs(galL) <= taperLength, val, 0)
        galB2 = np.where(np.abs(galL) <= taperLength, -val, 0)
        # Convert to ra/dec.
        # Convert to lon/lat and plot.
        ra, dec = _equatorialFromGalactic(galL, galB1)
        lon = -(ra - raCen - np.pi) % (np.pi * 2) - np.pi
        ax.plot(lon, dec, 'b.', markersize=1.8, alpha=0.4)
        ra, dec = _equatorialFromGalactic(galL, galB2)
        lon = -(ra - raCen - np.pi) % (np.pi * 2) - np.pi
        ax.plot(lon, dec, 'b.', markersize=1.8, alpha=0.4)

    def __call__(self, metricValueIn, slicer, userPlotDict, fignum=None):
        """
        Plot the sky map of metricValue for a generic spatial slicer.
        """
        if 'ra' not in slicer.slicePoints or 'dec' not in slicer.slicePoints:
            errMessage = 'SpatialSlicer must contain "ra" and "dec" in slicePoints metadata.'
            errMessage += ' SlicePoints only contains keys %s.' % (slicer.slicePoints.keys())
            raise ValueError(errMessage)
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)
        metricValue = applyZPNorm(metricValueIn, plotDict)

        fig = plt.figure(fignum, figsize=plotDict['figsize'])
        # other projections available include
        # ['aitoff', 'hammer', 'lambert', 'mollweide', 'polar', 'rectilinear']
        ax = fig.add_subplot(plotDict['subplot'], projection=plotDict['projection'])
        # Set up valid datapoints and colormin/max values.
        if plotDict['plotMask']:
            # Plot all data points.
            mask = np.ones(len(metricValue), dtype='bool')
        else:
            # Only plot points which are not masked. Flip numpy ma mask where 'False' == 'good'.
            good = ~metricValue.mask

        # Add ellipses at RA/Dec locations - but don't add colors yet.
        lon = -(slicer.slicePoints['ra'][good] - plotDict['raCen'] - np.pi) % (np.pi * 2) - np.pi
        ellipses = self._plot_tissot_ellipse(lon, slicer.slicePoints['dec'][good],
                                             plotDict['radius'], rasterized=True, ax=ax)
        if plotDict['metricIsColor']:
            current = None
            for ellipse, mVal in zip(ellipses, metricValue.data[good]):
                if mVal[3] > 1:
                    ellipse.set_alpha(1.0)
                    ellipse.set_facecolor((mVal[0], mVal[1], mVal[2]))
                    ellipse.set_edgecolor('k')
                    current = ellipse
                else:
                    ellipse.set_alpha(mVal[3])
                    ellipse.set_color((mVal[0], mVal[1], mVal[2]))
                ax.add_patch(ellipse)
            if current:
                ax.add_patch(current)
        else:
            # Determine color min/max values. metricValue.compressed = non-masked points.
            clims = setColorLims(metricValue, plotDict)
            # Determine whether or not to use auto-log scale.
            if plotDict['logScale'] == 'auto':
                if clims[0] > 0:
                    if np.log10(clims[1]) - np.log10(clims[0]) > 3:
                        plotDict['logScale'] = True
                    else:
                        plotDict['logScale'] = False
                else:
                    plotDict['logScale'] = False
            if plotDict['logScale']:
                # Move min/max values to things that can be marked on the colorbar.
                #clims[0] = 10 ** (int(np.log10(clims[0])))
                #clims[1] = 10 ** (int(np.log10(clims[1])))
                norml = colors.LogNorm()
                p = PatchCollection(ellipses, cmap=plotDict['cmap'], alpha=plotDict['alpha'],
                                    linewidth=0, edgecolor=None, norm=norml, rasterized=True)
            else:
                p = PatchCollection(ellipses, cmap=plotDict['cmap'], alpha=plotDict['alpha'],
                                    linewidth=0, edgecolor=None, rasterized=True)
            p.set_array(metricValue.data[good])
            p.set_clim(clims)
            ax.add_collection(p)
            # Add color bar (with optional setting of limits)
            if plotDict['cbar']:
                cb = plt.colorbar(p, aspect=25, extendrect=True, orientation='horizontal',
                                  format=plotDict['cbarFormat'])
                # If outputing to PDF, this fixes the colorbar white stripes
                if plotDict['cbar_edge']:
                    cb.solids.set_edgecolor("face")
                cb.set_label(plotDict['xlabel'], fontsize=plotDict['fontsize'])
                cb.ax.tick_params(labelsize=plotDict['labelsize'])
                tick_locator = ticker.MaxNLocator(nbins=plotDict['nTicks'])
                cb.locator = tick_locator
                cb.update_ticks()
        # Add ecliptic
        self._plot_ecliptic(plotDict['raCen'], ax=ax)
        if plotDict['mwZone']:
            self._plot_mwZone(plotDict['raCen'], ax=ax)
        ax.grid(True, zorder=1)
        ax.xaxis.set_ticklabels([])
        if plotDict['bgcolor'] is not None:
            ax.set_facecolor(plotDict['bgcolor'])
        # Add label.
        if plotDict['label'] is not None:
            plt.figtext(0.75, 0.9, '%s' % plotDict['label'], fontsize=plotDict['fontsize'])
        if plotDict['title'] is not None:
            plt.text(0.5, 1.09, plotDict['title'], horizontalalignment='center',
                    transform=ax.transAxes, fontsize=plotDict['fontsize'])
        return fig.number


class HealpixSDSSSkyMap(BasePlotter):
    def __init__(self):
        self.plotType = 'SkyMap'
        self.objectPlotter = False
        self.defaultPlotDict = {}
        self.defaultPlotDict.update(baseDefaultPlotDict)
        self.defaultPlotDict.update({'cbarFormat': '%.2f',
                                     'raMin': -90, 'raMax': 90, 'raLen': 45,
                                     'decMin': -2., 'decMax': 2.})

    def __call__(self, metricValueIn, slicer, userPlotDict, fignum=None):

        """
        Plot the sky map of metricValue using healpy cartview plots in thin strips.
        raMin: Minimum RA to plot (deg)
        raMax: Max RA to plot (deg).  Note raMin/raMax define the centers that will be plotted.
        raLen:  Length of the plotted strips in degrees
        decMin: minimum dec value to plot
        decMax: max dec value to plot
        metricValueIn: metric values
        """
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)
        metricValue = applyZPNorm(metricValueIn, plotDict)
        norm = None
        if plotDict['logScale']:
            norm = 'log'
        clims = setColorLims(metricValue, plotDict)
        cmap = setColorMap(plotDict)
        racenters = np.arange(plotDict['raMin'], plotDict['raMax'], plotDict['raLen'])
        nframes = racenters.size
        fig = plt.figure(fignum)
        # Do not specify or use plotDict['subplot'] because this is done in each call to hp.cartview.
        for i, racenter in enumerate(racenters):
            if i == 0:
                useTitle = plotDict['title'] + ' /n' + '%i < RA < %i' % (racenter - plotDict['raLen'],
                                                                         racenter + plotDict['raLen'])
            else:
                useTitle = '%i < RA < %i' % (racenter - plotDict['raLen'], racenter + plotDict['raLen'])
            hp.cartview(metricValue.filled(slicer.badval), title=useTitle, cbar=False,
                        min=clims[0], max=clims[1], flip='astro', rot=(racenter, 0, 0),
                        cmap=cmap, norm=norm, lonra=[-plotDict['raLen'], plotDict['raLen']],
                        latra=[plotDict['decMin'], plotDict['decMax']], sub=(nframes + 1, 1, i + 1), fig=fig)
            hp.graticule(dpar=20, dmer=20, verbose=False)
        # Add colorbar (not using healpy default colorbar because want more tickmarks).
        ax = fig.add_axes([0.1, .15, .8, .075])  # left, bottom, width, height
        # Add label.
        if plotDict['label'] is not None:
            plt.figtext(0.8, 0.9, '%s' % plotDict['label'])
        # Make the colorbar as a seperate figure,
        # from http: //matplotlib.org/examples/api/colorbar_only.html
        cnorm = colors.Normalize(vmin=clims[0], vmax=clims[1])
        # supress silly colorbar warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=cnorm,
                                           orientation='horizontal', format=plotDict['cbarFormat'])
            cb.set_label(plotDict['xlabel'])
            cb.ax.tick_params(labelsize=plotDict['labelsize'])
            if norm == 'log':
                tick_locator = ticker.LogLocator(numticks=plotDict['nTicks'])
                cb.locator = tick_locator
                cb.update_ticks()
            if (plotDict['nTicks'] is not None) & (norm != 'log'):
                tick_locator = ticker.MaxNLocator(nbins=plotDict['nTicks'])
                cb.locator = tick_locator
                cb.update_ticks()
        # If outputing to PDF, this fixes the colorbar white stripes
        if plotDict['cbar_edge']:
            cb.solids.set_edgecolor("face")
        fig = plt.gcf()
        return fig.number


def project_lambert(longitude, latitude):
    """Project from RA,dec to plane
    https://en.wikipedia.org/wiki/Lambert_azimuthal_equal-area_projection
    """

    # flipping the sign on latitude goes north pole or south pole centered
    r_polar = 2*np.cos((np.pi/2+latitude)/2.)
    # Add pi/2 so north is up
    theta_polar = longitude + np.pi/2

    x = r_polar * np.cos(theta_polar)
    y = r_polar * np.sin(theta_polar)
    return x, y


def draw_grat(ax):
    """Draw some graticule lines on an axis
    """
    decs = np.radians(90.-np.array([20, 40, 60, 80]))
    ra = np.radians(np.arange(0, 361, 1))
    for dec in decs:
        temp_dec = ra*0+dec
        x, y = project_lambert(ra, temp_dec)
        ax.plot(x, y, 'k--', alpha=0.5)

    ras = np.radians(np.arange(0, 360+45, 45))
    dec = np.radians(90.-np.arange(0, 81, 1))
    for ra in ras:
        temp_ra = dec*0 + ra
        x, y = project_lambert(temp_ra, dec)
        ax.plot(x, y, 'k--', alpha=0.5)

    for dec in decs:
        x, y = project_lambert(np.radians(45.), dec)
        ax.text(x, y, '%i' % np.round(np.degrees(dec)))

    return ax

class LambertSkyMap(BasePlotter):
    """
    Use basemap and contour to make a Lambertian projection.
    Note that the plotDict can include a 'basemap' key with a dictionary of
    arbitrary kwargs to use with the call to Basemap.
    """

    def __init__(self):
        self.plotType = 'SkyMap'
        self.objectPlotter = False
        self.defaultPlotDict = {}
        self.defaultPlotDict.update(baseDefaultPlotDict)
        self.defaultPlotDict.update({'basemap': {'projection': 'nplaea', 'boundinglat': 1, 'lon_0': 180,
                                                 'resolution': None, 'celestial': False, 'round': False},
                                    'levels': 200, 'cbarFormat': '%i', 'norm': None, 'title': ''})

    def __call__(self, metricValueIn, slicer, userPlotDict, fignum=None):

        if 'ra' not in slicer.slicePoints or 'dec' not in slicer.slicePoints:
            errMessage = 'SpatialSlicer must contain "ra" and "dec" in slicePoints metadata.'
            errMessage += ' SlicePoints only contains keys %s.' % (slicer.slicePoints.keys())
            raise ValueError(errMessage)

        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)

        metricValue = applyZPNorm(metricValueIn, plotDict)
        clims = setColorLims(metricValue, plotDict)
        # Calculate the levels to use for the contour
        if np.size(plotDict['levels']) > 1:
            levels = plotDict['levels']
        else:
            step = (clims[1] - clims[0]) / plotDict['levels']
            levels = np.arange(clims[0], clims[1] + step, step)

        fig = plt.figure(fignum, figsize=plotDict['figsize'])
        ax = fig.add_subplot(plotDict['subplot'])

        x, y = project_lambert(slicer.slicePoints['ra'], slicer.slicePoints['dec'])
        # Contour the plot first to remove any anti-aliasing artifacts.  Doesn't seem to work though. See:
        # http: //stackoverflow.com/questions/15822159/aliasing-when-saving-matplotlib\
        # -filled-contour-plot-to-pdf-or-eps
        # tmpContour = m.contour(np.degrees(slicer.slicePoints['ra']),
        #                        np.degrees(slicer.slicePoints['dec']),
        #                        metricValue.filled(np.min(clims)-1), levels, tri=True,
        #                        cmap=plotDict['cmap'], ax=ax, latlon=True,
        #                        lw=1)

        # Set masked values to be below the lowest contour level.
        if plotDict['norm'] == 'log':
            z_val = metricValue.filled(np.min(clims)-0.9)
            norm = colors.LogNorm(vmin=z_val.min(), vmax=z_val.max())
        else:
            norm = plotDict['norm']
        tcf = ax.tricontourf(x, y, metricValue.filled(np.min(clims)-0.9), levels,
                             cmap=plotDict['cmap'], norm=norm)

        ax = draw_grat(ax)

        ax.set_xticks([])
        ax.set_yticks([])
        alt_limit = 10.
        x, y = project_lambert(0, np.radians(alt_limit))
        max_val = np.max(np.abs([x, y]))
        ax.set_xlim([-max_val, max_val])
        ax.set_ylim([-max_val, max_val])

        # Try to fix the ugly pdf contour problem
        for c in tcf.collections:
            c.set_edgecolor("face")

        cb = plt.colorbar(tcf, format=plotDict['cbarFormat'])
        cb.set_label(plotDict['xlabel'])
        if plotDict['labelsize'] is not None:
            cb.ax.tick_params(labelsize=plotDict['labelsize'])
        # Pop in an extra line to raise the title a bit
        ax.set_title(plotDict['title']+'\n ')
        # If outputing to PDF, this fixes the colorbar white stripes
        if plotDict['cbar_edge']:
            cb.solids.set_edgecolor("face")
        return fig.number
