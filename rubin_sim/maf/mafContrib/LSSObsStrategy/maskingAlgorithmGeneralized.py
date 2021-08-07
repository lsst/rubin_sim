#####################################################################################################
# Purpose: change the values/mask of a metricBundle in the pixels with a certain value/mask.
# Example applicaton: mask the outermost/shallow edge of skymaps. 
#
# Humna Awan: humna.awan@rutgers.edu
#####################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import copy
import rubin_sim.maf.plots as plots
import matplotlib.cm as cm

__all__ = ['maskingAlgorithmGeneralized']

def maskingAlgorithmGeneralized(myBundles, plotHandler, dataLabel, nside=128,
                                findValue='unmasked', relation='=',
                                newValue='masked',
                                pixelRadius=6, returnBorderIndices=False,
                                printIntermediateInfo=False, plotIntermediatePlots=True,
                                printFinalInfo=True,
                                plotFinalPlots=True,
                                skyMapColorMin=None, skyMapColorMax=None):
    """
    Assign newValue to all pixels in a skymap within pixelRadius of pixels with value <, >, or = findValue.

    Parameters
    --------------------
    myBundles   : `dict` {`rubin_sim.maf.MetricBundles`}
        a dictionary for metricBundles.
    plotHandler :   `rubin_sim.maf.plots.plotHandler.PlotHandler`
    dataLabel : `str`
        description of the data, i.e. 'numGal'
    nside : `int`
        HEALpix resolution parameter. Default: 128
    findValue : `str`
        if related to mask, must be either 'masked' or 'unmasked'. otherwise, must be a number.
        Default: 'unmasked'
    relation : `str`
        must be '>','=','<'. Default: '='
    newValue : `str`
        if related to mask, must be either 'masked' or 'unmasked'; otherwise, must be a number.
        Default: 'masked'
    pixelRadius : `int`
        number of pixels to consider around a given pixel. Default: 6
    returnBorderIndices : `bool`
        set to True to return the array of indices of the pixels whose values/mask are changed. Default: False
    printIntermediateInfo : `bool`
        set to False if do not want to print intermediate info. Default: True
    plotIntermediatePlots : `bool`
        set to False if do not want to plot intermediate plots. Default: True
    printFinalInfo : `bool`
        set to False if do not want to print final info, i.e. total pixels changed. Default: True
    plotFinalPlots : `bool`
        set to False if do not want to plot the final plots. Default: True
    skyMapColorMin : float
        colorMin label value for skymap plotDict label. Default: None
    skyMapColorMax : float
        colorMax label value for skymap plotDict label. Default: None
    """
    # find pixels such that (pixelValue (relation) findValue) AND their neighbors dont have that (relation) findValue.
    # then assign newValue to all these pixels.
    # relation must be '>','=','<'
    # data indices are the pixels numbers ..
    # ------------------------------------------------------------------------
    # check whether need to mask anything at all
    if (pixelRadius == 0):
        print('No masking/changing of the original data.')
        if returnBorderIndices:
            borders = {}
            for dither in myBundles:
                borders[dither] = []
                
            return [myBundles, borders]
        else:
            return myBundles
    # ------------------------------------------------------------------------
    # make sure that relation is compatible with findValue
    if ((findValue == 'masked') | (findValue == 'unmasked')):
        if (relation != '='):
            print('ERROR: must have relation== "=" if findValue is related to mask.')
            print('Setting:  relation= "="\n')
            relation = '='
    # ------------------------------------------------------------------------
    # translate findValue into what has to be assigned
    findValueToConsider = findValue
    if findValue.__contains__('mask'):
        if (findValue == 'masked'):
            findValueToConsider = True
        if (findValue == 'unmasked'):
            findValueToConsider = False

    # translate newValue into what has to be assigned
    newValueToAssign= newValue   
    if newValue.__contains__('mask'):
        if (newValue == 'masked'):
            newValueToAssign = True
        if (newValue == 'unmasked'):
            newValueToAssign = False

    # ------------------------------------------------------------------------
    borders = {}
    for dither in myBundles:
        totalBorderPixel = []
        if printIntermediateInfo:
            print('Survey strategy: %s'%dither)
            
        # find the array to look at.
        if (findValue).__contains__('mask'):
            origArray = myBundles[dither].metricValues.mask.copy()    # mask array
        else:
            origArray = myBundles[dither].metricValues.data.copy()    # data array

        for r in range(0, pixelRadius):
            borderPixel = []
            tempCopy = copy.deepcopy(myBundles)
            # ignore the pixels whose neighbors formed the border in previous run
            if (r != 0):
                origArray[totalBorderPixel] = newValueToAssign

            # find the pixels that satisfy the relation with findValue and whose neighbors dont
            for i in range(0, len(origArray)):
                neighborsPixels = hp.get_all_neighbours(nside,i)   # i is the pixel number
                for j in neighborsPixels:
                    condition= None
                    if (relation == '<'):
                        condition = ((origArray[i] < findValueToConsider) & (origArray[j] >= findValueToConsider))
                    if (relation == '='):
                        condition = ((origArray[i] == findValueToConsider) & (origArray[j] != findValueToConsider))
                    if (relation == '>'):
                        condition = ((origArray[i] > findValueToConsider) & (origArray[j] <= findValueToConsider))
                    if (condition == None):
                        raise ValueError('ERROR: invalid relation: %s'%relation)
                        
                    if condition:
                        if (j != -1):                            # -1 entries correspond to inexistent neighbors
                            borderPixel.append(i)
        
            borderPixel = np.unique(borderPixel)
            totalBorderPixel.extend(borderPixel)

            if printIntermediateInfo:     
                print('Border pixels from run %s: %s'%(r+1, len(borderPixel)))
                print('Total pixels so far: %s\n'%len(totalBorderPixel))
      
            # plot found pixels
            if plotIntermediatePlots:
                if newValue.__contains__('mask'):
                    tempCopy[dither].metricValues.mask[:] = newValueToAssign
                    tempCopy[dither].metricValues.mask[totalBorderPixel] = not(newValueToAssign)
                    tempCopy[dither].metricValues.data[totalBorderPixel] = -500
                    plotDict = {'xlabel': dataLabel, 'title':'%s: %s Round # %s' %(dither, dataLabel, r+1), 
                                'logScale': False, 'labelsize': 9,'colorMin':-550, 'colorMax': 550, 'cmap':cm.jet}
                else:
                    tempCopy[dither].metricValues.mask[:] = True
                    tempCopy[dither].metricValues.mask[totalBorderPixel] = False
                    tempCopy[dither].metricValues.data[totalBorderPixel] = newValueToAssign
                    plotDict = {'xlabel': dataLabel, 'title':'%s %s Round # %s' %(dither, dataLabel, r+1), 
                                'logScale': False, 'labelsize': 9, 'maxl': 500, 'cmap':cm.jet}
                tempCopy[dither].setPlotDict(plotDict)
                tempCopy[dither].setPlotFuncs([plots.HealpixSkyMap(), plots.HealpixPowerSpectrum()])
                tempCopy[dither].plot(plotHandler=plotHandler)
                plt.show()
            # save the found pixels with the appropriate key
            borders[dither] = totalBorderPixel

    # ------------------------------------------------------------------------
    # change the original map/array now.
    for dither in myBundles:
        totalBorderPixel = borders[dither]

        if printFinalInfo:
            print('Survey strategy: %s'%dither)
            print('Total pixels changed: %s\n'%len(totalBorderPixel))
            
        if newValue.__contains__('mask'):
            myBundles[dither].metricValues.mask[totalBorderPixel] = newValueToAssign
        else:
            myBundles[dither].metricValues.data[totalBorderPixel] = newValueToAssign
            
        if plotFinalPlots:
            # skymap
            plotDict = {'xlabel':dataLabel, 'title':'%s: %s MaskedMap; pixelRadius: %s ' %(dither, dataLabel, pixelRadius),
                        'logScale': False, 'labelsize': 8,'colorMin': skyMapColorMin, 'colorMax': skyMapColorMax, 'cmap':cm.jet}
            myBundles[dither].setPlotDict(plotDict)
            myBundles[dither].setPlotFuncs([plots.HealpixSkyMap()])
            myBundles[dither].plot(plotHandler=plotHandler)
            # power spectrum
            plotDict = {'xlabel':dataLabel, 'title':'%s: %s MaskedMap; pixelRadius: %s ' %(dither, dataLabel, pixelRadius),
                        'logScale': False, 'labelsize': 12, 'maxl': 500, 'cmap':cm.jet}
            myBundles[dither].setPlotDict(plotDict)
            myBundles[dither].setPlotFuncs([plots.HealpixPowerSpectrum()])
            myBundles[dither].plot(plotHandler=plotHandler)
            plt.show()

    if returnBorderIndices:
        return [myBundles, borders]
    else:
        return myBundles
