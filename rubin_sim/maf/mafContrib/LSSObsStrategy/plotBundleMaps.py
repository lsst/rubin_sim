########################################################################################################################
# Purpose: plots for the data in a metricBundle object without using MAF routines. Includes
# functionality to plot skymaps, cartviews, and power spectra.
#
# Humna Awan: humna.awan@rutgers.edu
#
########################################################################################################################
import matplotlib.pyplot as plt
import numpy as np
import os
import healpy as hp

__all__ = ['plotBundleMaps']

def plotBundleMaps(path, outDir, bundle, dataLabel, filterBand,
                   dataName=None,
                   skymap=True, powerSpectrum=True,
                   cartview=False, raRange=[-180,180], decRange=[-70,10],
                   showPlots=True,
                   saveFigs=False, outDirNameForSavedFigs='',
                   lmax=500,
                   nTicks=5, numFormat='%.2f',
                   colorMin=None, colorMax=None):
    """

    Plot maps for the data in a metricBundle object without using MAF routines.

    Required Parameters
    -------------------
      * path: str: path to the main directory where output directory is saved
      * outDir: str: name of the main output directory
      * bundle: metricBundle object.
      * dataLabel: str: data type, e.g. 'counts', 'NumGal'. Will be the label for the colorbar in
                        in skymaps/cartview plots.
      * filterBand: str: filter to consider, e.g. 'r'
    
    Optional Parameters
    -------------------
      * dataName: str: dataLabel analog for filename. e.g. say for datalabel='c$_l$', a good dataName is 'cl'
                       Default: None 
      * skymap: boolean: set to True if want to plot skymaps. Default: True
      * powerSpectrum: boolean: set to True if want to plot powerspectra; dipole is removed. Default: True

      * cartview: boolean: set to True if want to plot cartview plots. Default: Fase
      * raRange: float array: range of right ascention (in degrees) to consider in cartview plot; only useful when 
                              cartview=True. Default: [-180,180]
      * decRange: float array: range of declination (in degrees) to consider in cartview plot; only useful when 
                               cartview=True. Default: [-70,10]

      * showPlots: boolean: set to True if want to show figures. Default: True
      * saveFigs: boolean: set to True if want to save figures. Default: False
      * outDirNameForSavedFigs: str: name for the output directory if saveFigs=True. Default: ''
      * lmax: int: upper limit on the multipole. Default: 500
      * nTicks: int: (number of ticks - 1) on the skymap colorbar. Default: 5
      * numFormat: str: number format for the labels on the colorbar. Default: '%.2f'
      * colorMin: float: lower limit on the colorscale for skymaps. Default: None
      * colorMax: float: upper limit on the colorscale for skymaps. Default: None

    """
    if dataName is None:
        dataName = dataLabel

    if saveFigs:
        # set up the subdirectory
        outDirNew = outDirNameForSavedFigs
        if not os.path.exists('%s%s/%s'%(path, outDir, outDirNew)):
            os.makedirs('%s%s/%s'%(path, outDir, outDirNew))
    
    if powerSpectrum:
        # plot out the power spectrum
        for dither in bundle:
            plt.clf()
            cl = hp.anafast(hp.remove_dipole(bundle[dither].metricValues.filled(bundle[dither].slicer.badval)),
                            lmax=lmax)
            ell = np.arange(len(cl))
            plt.plot(ell, (cl*ell*(ell+1))/2.0/np.pi)
            plt.title('%s: %s'%(dataLabel, dither))
            plt.xlabel(r'$\ell$')
            plt.ylabel(r'$\ell(\ell+1)C_\ell/(2\pi)$')
            plt.xlim(0, lmax)

            if saveFigs:
                # save power spectrum
                filename = '%s_powerSpectrum_%s.png'%(dataName, dither)
                plt.savefig('%s%s/%s/%s'%(path, outDir, outDirNew, filename),  bbox_inches='tight', format='png')
            if showPlots:
                plt.show()
            else:
                plt.close('all')
    if skymap:
        # plot out the skymaps
        for dither in bundle:
            inSurveyIndex = np.where(bundle[dither].metricValues.mask == False)[0]
            median = np.median(bundle[dither].metricValues.data[inSurveyIndex])
            stddev = np.std(bundle[dither].metricValues.data[inSurveyIndex])
        
            if (colorMin == None):
                colorMin = median-1.5*stddev
            if (colorMax == None):
                colorMax = median+1.5*stddev
        
            increment = (colorMax-colorMin)/float(nTicks)
            ticks = np.arange(colorMin+increment, colorMax, increment)
            
            hp.mollview(bundle[dither].metricValues.filled(bundle[dither].slicer.badval), 
                        flip='astro', rot=(0,0,0) ,
                        min=colorMin, max=colorMax, title='', cbar=False)
            hp.graticule(dpar=20, dmer=20, verbose=False)
            plt.title(dither)
            ax = plt.gca()
            im = ax.get_images()[0]
            fig= plt.gcf()
            cbaxes = fig.add_axes([0.1, 0.03, 0.8, 0.04]) # [left, bottom, width, height]
            cb = plt.colorbar(im, orientation='horizontal',
                              ticks=ticks, format=numFormat, cax=cbaxes)
            cb.set_label(dataLabel)
            
            if saveFigs:
                # save skymap
                filename = '%s_skymap_%s.png'%(dataName, dither)
                plt.savefig('%s%s/%s/%s'%(path, outDir, outDirNew, filename),  bbox_inches='tight', format='png')
            if showPlots:
                plt.show()
            else:
                plt.close('all')

    if cartview:
        # plot out the cartview plots
        for dither in bundle:
            inSurveyIndex = np.where(bundle[dither].metricValues.mask == False)[0]
            median = np.median(bundle[dither].metricValues.data[inSurveyIndex])
            stddev = np.std(bundle[dither].metricValues.data[inSurveyIndex])
        
            if (colorMin == None):
                colorMin = median-1.5*stddev
            if (colorMax == None):
                colorMax = median+1.5*stddev
        
            increment = (colorMax-colorMin)/float(nTicks)
            ticks = np.arange(colorMin+increment, colorMax, increment)

            hp.cartview(bundle[dither].metricValues.filled(bundle[dither].slicer.badval), 
                        flip='astro', rot=(0,0,0),
                        lonra=raRange, latra=decRange,
                        min=colorMin, max=colorMax, title='', cbar=False)
            hp.graticule(dpar=20, dmer=20, verbose=False)
            plt.title(dither)
            ax = plt.gca()
            im = ax.get_images()[0]
            fig = plt.gcf()
            cbaxes = fig.add_axes([0.1, 0.25, 0.8, 0.04]) # [left, bottom, width, height]
            cb = plt.colorbar(im, orientation='horizontal',
                              ticks=ticks, format=numFormat, cax=cbaxes)
            cb.set_label(dataLabel)
            
            if saveFigs:
                # save cartview plot
                filename = '%s_cartview_%s.png'%(dataName, dither)
                plt.savefig('%s%s/%s/%s'%(path, outDir, outDirNew, filename),  bbox_inches='tight', format='png')
            if showPlots:
                plt.show()
            else:
                plt.close('all')
