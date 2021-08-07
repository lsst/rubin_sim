#####################################################################################################
# Purpose: calculate the coadded 5-sigma depth from various survey strategies. Incudes functionality
# to consider various survey strategies, mask shallow borders, create/save/show relevant plots, do
# an alm analysis, and save data.

# Humna Awan: humna.awan@rutgers.edu
#####################################################################################################
import numpy as np
import os
import healpy as hp
import copy
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator

import matplotlib.pyplot as plt

import rubin_sim.maf.db as db
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.plots as plots
import rubin_sim.maf.metricBundles as metricBundles
import rubin_sim.maf.maps as maps
import rubin_sim.maf.stackers as mafStackers   # stackers in sims_maf

from rubin_sim.maf.mafContrib.LSSObsStrategy.maskingAlgorithmGeneralized import maskingAlgorithmGeneralized
from rubin_sim.maf.mafContrib.LSSObsStrategy.almPlots import almPlots
from rubin_sim.maf.mafContrib.LSSObsStrategy.saveBundleData_npzFormat import saveBundleData_npzFormat

from rubin_sim.maf.mafContrib.LSSObsStrategy.constantsForPipeline import plotColor

__all__ = ['coaddM5Analysis']

def coaddM5Analysis(path, dbfile, runName, slair=False,
                    WFDandDDFs=False,
                    noDithOnly=False, bestDithOnly=False, someDithOnly=False,
                    specifiedDith=None,
                    nside=128, filterBand='r',
                    includeDustExtinction=False,
                    saveunMaskedCoaddData=False,
                    pixelRadiusForMasking=5, cutOffYear=None,
                    plotSkymap=True,
                    plotCartview=True,
                    unmaskedColorMin=None, unmaskedColorMax=None,
                    maskedColorMin=None, maskedColorMax=None,
                    nTicks=5,
                    plotPowerSpectrum=True,
                    showPlots=True, saveFigs=True,
                    almAnalysis=True,
                    raRange=[-50,50], decRange=[-65,5],
                    saveMaskedCoaddData=True):

    """

    Analyze the artifacts induced in the coadded 5sigma depth due to imperfect observing strategy.
      - Creates an output directory for subdirectories containing the specified things to save.
      - Creates, shows, and saves comparison plots.
      - Returns the metricBundle object containing the calculated coadded depth, and the output directory name.

    Parameters
    -------------------
    path: str
        path to the main directory where output directory is to be saved.
    dbfile: str
        path to the OpSim output file, e.g. to a copy of enigma_1189
    runName: str
        run name tag to identify the output of specified OpSim output, e.g. 'enigma1189'
    slair: `bool`
        set to True if analysis on a SLAIR output.
        Default: False
    WFDandDDFs: `bool`
        set to True if want to consider both WFD survet and DDFs. Otherwise will only work
        with WFD. Default: False
    noDithOnly: `bool`
        set to True if only want to consider the undithered survey. Default: False
    bestDithOnly: `bool`
        set to True if only want to consider RandomDitherFieldPerVisit.
        Default: False
    someDithOnly: `bool`
        set to True if only want to consider undithered and a few dithered surveys.
        Default: False
    specifiedDith: str
        specific dither strategy to run.
        Default: None
    nside: int
        HEALpix resolution parameter. Default: 128
    filterBand: str
        any one of 'u', 'g', 'r', 'i', 'z', 'y'. Default: 'r'
    includeDustExtinction: `bool`
        set to include dust extinction. Default: False
    saveunMaskedCoaddData: `bool`
        set to True to save data before border masking. Default: False
    pixelRadiusForMasking: int
        number of pixels to mask along the shallow border. Default: 5
    cutOffYear: int
        year cut to restrict analysis to only a subset of the survey.
        Must range from 1 to 9, or None for the full survey analysis (10 yrs).
        Default: None
    plotSkymap: `bool`
        set to True if want to plot skymaps. Default: True
    plotCartview: `bool`
        set to True if want to plot cartview plots. Default: False
    unmaskedColorMin: float
        lower limit on the colorscale for unmasked skymaps. Default: None
    unmaskedColorMax: float
        upper limit on the colorscale for unmasked skymaps. Default: None
    maskedColorMin: float
        lower limit on the colorscale for border-masked skymaps. Default: None
    maskedColorMax: float
        upper limit on the colorscale for border-masked skymaps. Default: None
    nTicks: int
        (number of ticks - 1) on the skymap colorbar. Default: 5
    plotPowerSpectrum: `bool`
        set to True if want to plot powerspectra. Default: True
    showPlots: `bool`
        set to True if want to show figures. Default: True
    saveFigs: `bool`
        set to True if want to save figures. Default: True
    almAnalysis: `bool`
        set to True to perform the alm analysis. Default: True
    raRange: float array
        range of right ascention (in degrees) to consider in alm  cartview plot;
        applicable when almAnalysis=True. Default: [-50,50]
    decRange: float array
        range of declination (in degrees) to consider in alm cartview plot;
        applicable when almAnalysis=True. Default: [-65,5]
    saveMaskedCoaddData: `bool`
        set to True to save the coadded depth data after the border
        masking. Default: True
    """
    # ------------------------------------------------------------------------
    # read in the database
    if slair:
        # slair database
        opsdb = db.Database(dbfile, defaultTable='observations')
    else:
        # OpSim database
        opsdb = db.OpsimDatabase(dbfile)

    # ------------------------------------------------------------------------
    # set up the outDir
    zeropt_tag = ''
    if cutOffYear is not None: zeropt_tag = '%syearCut'%cutOffYear
    else: zeropt_tag = 'fullSurveyPeriod'

    if includeDustExtinction: dust_tag = 'withDustExtinction'
    else: dust_tag = 'noDustExtinction'

    regionType = ''
    if  WFDandDDFs: regionType = 'WFDandDDFs_'
        
    outDir = 'coaddM5Analysis_%snside%s_%s_%spixelRadiusForMasking_%sBand_%s_%s_directory'%(regionType, nside,dust_tag,
                                                                                            pixelRadiusForMasking,
                                                                                            filterBand, runName, zeropt_tag)
    print('# outDir: %s'%outDir)
    resultsDb = db.ResultsDb(outDir=outDir)

    # ------------------------------------------------------------------------
    # set up the sql constraint
    if  WFDandDDFs:
        if cutOffYear is not None:
            nightCutOff = (cutOffYear)*365.25
            sqlconstraint = 'night<=%s and filter=="%s"'%(nightCutOff, filterBand)
        else:
            sqlconstraint = 'filter=="%s"'%filterBand
    else:
        # set up the propID and units on the ra, dec
        if slair: # no prop ID; only WFD is simulated.
            wfdWhere = ''
            raDecInDeg = True
        else:
            propIds, propTags = opsdb.fetchPropInfo()
            wfdWhere = '%s and '%opsdb.createSQLWhere('WFD', propTags)
            raDecInDeg = opsdb.raDecInDeg
        # set up the year cutoff
        if cutOffYear is not None:
            nightCutOff = (cutOffYear)*365.25
            sqlconstraint = '%snight<=%s and filter=="%s"'%(wfdWhere, nightCutOff, filterBand)
        else:
            sqlconstraint = '%sfilter=="%s"'%(wfdWhere, filterBand)
    print('# sqlconstraint: %s'%sqlconstraint)

    # ------------------------------------------------------------------------
    # setup all the slicers
    slicer = {}
    stackerList = {}
    
    if specifiedDith is not None: # would like to add all the stackers first and then keep only the one that is specified
        bestDithOnly, noDithOnly = False, False

    if bestDithOnly:
        stackerList['RandomDitherFieldPerVisit'] = [mafStackers.RandomDitherFieldPerVisitStacker(degrees=raDecInDeg,
                                                                                                 randomSeed=1000)]
        slicer['RandomDitherFieldPerVisit'] = slicers.HealpixSlicer(lonCol='randomDitherFieldPerVisitRa',
                                                                   latCol='randomDitherFieldPerVisitDec',
                                                                   latLonDeg=raDecInDeg,
                                                                   nside=nside, useCache=False)
    else:
        if slair:
            slicer['NoDither'] = slicers.HealpixSlicer(lonCol='RA', latCol='dec', latLonDeg=raDecInDeg,
                                                      nside=nside, useCache=False)
        else:
            slicer['NoDither'] = slicers.HealpixSlicer(lonCol='fieldRA', latCol='fieldDec', latLonDeg=raDecInDeg,
                                                      nside=nside, useCache=False)
        if someDithOnly and not noDithOnly:
            #stackerList['RepulsiveRandomDitherFieldPerVisit'] = [myStackers.RepulsiveRandomDitherFieldPerVisitStacker(degrees=raDecInDeg,
            #                                                                                                          randomSeed=1000)]
            #slicer['RepulsiveRandomDitherFieldPerVisit'] = slicers.HealpixSlicer(lonCol='repulsiveRandomDitherFieldPerVisitRa',
            #                                                                    latCol='repulsiveRandomDitherFieldPerVisitDec',
            #                                                                    latLonDeg=raDecInDeg, nside=nside,
            #                                                                    useCache=False)
            slicer['SequentialHexDitherFieldPerNight'] =  slicers.HealpixSlicer(lonCol='hexDitherFieldPerNightRa',
                                                                               latCol='hexDitherFieldPerNightDec',
                                                                               latLonDeg=raDecInDeg,
                                                                               nside=nside, useCache=False)
            slicer['PentagonDitherPerSeason'] = slicers.HealpixSlicer(lonCol='pentagonDitherPerSeasonRa', latCol='pentagonDitherPerSeasonDec',
                                                                     latLonDeg=raDecInDeg,
                                                                     nside=nside, useCache=False)
        elif not noDithOnly:
            # random dithers on different timescales
            stackerList['RandomDitherPerNight'] = [mafStackers.RandomDitherPerNightStacker(degrees=raDecInDeg, randomSeed=1000)]
            stackerList['RandomDitherFieldPerNight'] = [mafStackers.RandomDitherFieldPerNightStacker(degrees=raDecInDeg, randomSeed=1000)]
            stackerList['RandomDitherFieldPerVisit'] = [mafStackers.RandomDitherFieldPerVisitStacker(degrees=raDecInDeg, randomSeed=1000)]
            
            # rep random dithers on different timescales
            #stackerList['RepulsiveRandomDitherPerNight'] = [myStackers.RepulsiveRandomDitherPerNightStacker(degrees=raDecInDeg,
            #                                                                                                randomSeed=1000)]
            #stackerList['RepulsiveRandomDitherFieldPerNight'] = [myStackers.RepulsiveRandomDitherFieldPerNightStacker(degrees=raDecInDeg,
            #                                                                                                          randomSeed=1000)]
            #stackerList['RepulsiveRandomDitherFieldPerVisit'] = [myStackers.RepulsiveRandomDitherFieldPerVisitStacker(degrees=raDecInDeg,
            #                                                                                                          randomSeed=1000)]
            # set up slicers for different dithers
            # random dithers on different timescales
            slicer['RandomDitherPerNight'] = slicers.HealpixSlicer(lonCol='randomDitherPerNightRa', latCol='randomDitherPerNightDec',
                                                                  latLonDeg=raDecInDeg, nside=nside, useCache=False)
            slicer['RandomDitherFieldPerNight'] = slicers.HealpixSlicer(lonCol='randomDitherFieldPerNightRa',
                                                                       latCol='randomDitherFieldPerNightDec',
                                                                       latLonDeg=raDecInDeg, nside=nside, useCache=False)
            slicer['RandomDitherFieldPerVisit'] = slicers.HealpixSlicer(lonCol='randomDitherFieldPerVisitRa',
                                                                       latCol='randomDitherFieldPerVisitDec',
                                                                       latLonDeg=raDecInDeg, nside=nside, useCache=False)
            # rep random dithers on different timescales
            #slicer['RepulsiveRandomDitherPerNight'] = slicers.HealpixSlicer(lonCol='repulsiveRandomDitherPerNightRa',
            #                                                               latCol='repulsiveRandomDitherPerNightDec',
            #                                                               latLonDeg=raDecInDeg, nside=nside, useCache=False)
            #slicer['RepulsiveRandomDitherFieldPerNight'] = slicers.HealpixSlicer(lonCol='repulsiveRandomDitherFieldPerNightRa',
            #                                                                    latCol='repulsiveRandomDitherFieldPerNightDec',
            #                                                                    latLonDeg=raDecInDeg, nside=nside,
            #                                                                    useCache=False)
            #slicer['RepulsiveRandomDitherFieldPerVisit'] = slicers.HealpixSlicer(lonCol='repulsiveRandomDitherFieldPerVisitRa',
            #                                                                    latCol='repulsiveRandomDitherFieldPerVisitDec',
            #                                                                    latLonDeg=raDecInDeg, nside=nside,
            #                                                                    useCache=False)
            # spiral dithers on different timescales
            slicer['FermatSpiralDitherPerNight'] = slicers.HealpixSlicer(lonCol='fermatSpiralDitherPerNightRa',
                                                                         latCol='fermatSpiralDitherPerNightDec',
                                                                         latLonDeg=raDecInDeg, nside=nside, useCache=False)
            slicer['FermatSpiralDitherFieldPerNight'] = slicers.HealpixSlicer(lonCol='fermatSpiralDitherFieldPerNightRa',
                                                                              latCol='fermatSpiralDitherFieldPerNightDec',
                                                                              latLonDeg=raDecInDeg, nside=nside,
                                                                              useCache=False)
            slicer['FermatSpiralDitherFieldPerVisit'] = slicers.HealpixSlicer(lonCol='fermatSpiralDitherFieldPerVisitRa',
                                                                              latCol='fermatSpiralDitherFieldPerVisitDec',
                                                                              latLonDeg=raDecInDeg, nside=nside,
                                                                              useCache=False)
            # hex dithers on different timescales
            slicer['SequentialHexDitherPerNight'] = slicers.HealpixSlicer(lonCol='hexDitherPerNightRa', latCol='hexDitherPerNightDec',
                                                                          latLonDeg=raDecInDeg, nside=nside, useCache=False)
            slicer['SequentialHexDitherFieldPerNight'] = slicers.HealpixSlicer(lonCol='hexDitherFieldPerNightRa',
                                                                               latCol='hexDitherFieldPerNightDec',
                                                                               latLonDeg=raDecInDeg, nside=nside, useCache=False)
            slicer['SequentialHexDitherFieldPerVisit'] = slicers.HealpixSlicer(lonCol='hexDitherFieldPerVisitRa',
                                                                               latCol='hexDitherFieldPerVisitDec',
                                                                               latLonDeg=raDecInDeg, nside=nside, useCache=False)
            # per season dithers
            slicer['PentagonDitherPerSeason'] = slicers.HealpixSlicer(lonCol='pentagonDitherPerSeasonRa', latCol='pentagonDitherPerSeasonDec',
                                                                     latLonDeg=raDecInDeg, nside=nside, useCache=False)
            slicer['PentagonDiamondDitherPerSeason'] = slicers.HealpixSlicer(lonCol='pentagonDiamondDitherPerSeasonRa',
                                                                            latCol='pentagonDiamondDitherPerSeasonDec',
                                                                            latLonDeg=raDecInDeg, nside=nside,
                                                                            useCache=False)
            slicer['SpiralDitherPerSeason'] = slicers.HealpixSlicer(lonCol='spiralDitherPerSeasonRa',
                                                                   latCol='spiralDitherPerSeasonDec',
                                                                   latLonDeg=raDecInDeg, nside=nside, useCache=False)
    if specifiedDith is not None:
        stackerList_, slicer_ = {}, {}
        if specifiedDith in slicer.keys():
            if specifiedDith.__contains__('Random'):   # only Random dithers have a stacker object for rand seed specification
                stackerList_[specifiedDith] = stackerList[specifiedDith]
            slicer_[specifiedDith] = slicer[specifiedDith]
        else:
            raise ValueError('Invalid value for specifiedDith: %s. Allowed values include one of the following:\n%s'%(specifiedDith,
                                                                                                                      slicer.keys()))
        stackerList, slicer = stackerList_, slicer_

    # ------------------------------------------------------------------------
    if slair:
        m5Col = 'fivesigmadepth'
    else:
        m5Col = 'fiveSigmaDepth'
    # set up the metric
    if includeDustExtinction:
        # include dust extinction when calculating the co-added depth
        coaddMetric = metrics.ExgalM5(m5Col=m5Col, lsstFilter=filterBand)
    else:
        coaddMetric = metrics.Coaddm5Metric(m5col=m5col)
    dustMap = maps.DustMap(interp=False, nside=nside)   # include dustMap; actual in/exclusion of dust is handled by the galaxyCountMetric

    # ------------------------------------------------------------------------
    # set up the bundle
    coaddBundle = {}
    for dither in slicer:
        if dither in stackerList:
            coaddBundle[dither] = metricBundles.MetricBundle(coaddMetric, slicer[dither], sqlconstraint,
                                                             stackerList=stackerList[dither],
                                                             runName=runName, metadata=dither, mapsList=[dustMap])
        else:
            coaddBundle[dither] = metricBundles.MetricBundle(coaddMetric, slicer[dither], sqlconstraint,
                                                             runName=runName, metadata=dither, mapsList=[dustMap])

    # ------------------------------------------------------------------------
    # run the analysis
    if includeDustExtinction: print('\n# Running coaddBundle with dust extinction ...')
    else: print('\n# Running coaddBundle without dust extinction ...')
    cGroup = metricBundles.MetricBundleGroup(coaddBundle, opsdb, outDir=outDir, resultsDb=resultsDb,saveEarly=False)
    cGroup.runAll()

    # ------------------------------------------------------------------------
    plotHandler = plots.PlotHandler(outDir=outDir, resultsDb=resultsDb, thumbnail=False, savefig=False)

    print('# Number of pixels in the survey region (before masking the border):')
    for dither in coaddBundle:
        print('  %s: %s'%(dither, len(np.where(coaddBundle[dither].metricValues.mask == False)[0])))

    # ------------------------------------------------------------------------
    # save the unmasked data?
    if saveunMaskedCoaddData:
        outDir_new = 'unmaskedCoaddData'
        if not os.path.exists('%s%s/%s'%(path, outDir, outDir_new)):
            os.makedirs('%s%s/%s'%(path, outDir, outDir_new))
        saveBundleData_npzFormat('%s%s/%s'%(path, outDir, outDir_new), coaddBundle, 'coaddM5Data_unmasked', filterBand)
    
    # ------------------------------------------------------------------------
    # mask the edges
    print('\n# Masking the edges for coadd ...')
    coaddBundle = maskingAlgorithmGeneralized(coaddBundle, plotHandler,
                                             dataLabel='$%s$-band Coadded Depth'%filterBand,
                                             nside=nside,
                                             pixelRadius=pixelRadiusForMasking,
                                             plotIntermediatePlots=False,
                                             plotFinalPlots=False, printFinalInfo=True)
    # ------------------------------------------------------------------------
    # Calculate total power
    summarymetric = metrics.TotalPowerMetric()
    for dither in coaddBundle:
        coaddBundle[dither].setSummaryMetrics(summarymetric)
        coaddBundle[dither].computeSummaryStats()
        print('# Total power for %s case is %f.' %(dither, coaddBundle[dither].summaryValues['TotalPower']))
    print('')
    
    # ------------------------------------------------------------------------
    # run the alm analysis
    if almAnalysis: almPlots(path, outDir, copy.deepcopy(coaddBundle),
                             nside=nside, filterband=filterBand,
                             raRange=raRange, decRange=decRange,
                             showPlots=showPlots)
    # ------------------------------------------------------------------------
    # save the masked data?
    if saveMaskedCoaddData and (pixelRadiusForMasking>0):
        outDir_new = 'maskedCoaddData'
        if not os.path.exists('%s%s/%s'%(path, outDir, outDir_new)):
            os.makedirs('%s%s/%s'%(path, outDir, outDir_new))
        saveBundleData_npzFormat('%s%s/%s'%(path, outDir, outDir_new), coaddBundle,
                                 'coaddM5Data_masked', filterBand)

    # ------------------------------------------------------------------------
    # plot comparison plots
    if len(coaddBundle.keys())>1:  # more than one key
        # set up the directory
        outDir_comp = 'coaddM5ComparisonPlots'
        if not os.path.exists('%s%s/%s'%(path, outDir, outDir_comp)):
            os.makedirs('%s%s/%s'%(path, outDir, outDir_comp))
        # ------------------------------------------------------------------------
        # plot for the power spectra
        cl = {}
        for dither in plotColor:
            if dither in coaddBundle:
                cl[dither] = hp.anafast(hp.remove_dipole(coaddBundle[dither].metricValues.filled(coaddBundle[dither].slicer.badval)),
                                        lmax=500)
                ell = np.arange(np.size(cl[dither]))
                plt.plot(ell, (cl[dither]*ell*(ell+1))/2.0/np.pi, color=plotColor[dither], linestyle='-', label=dither)
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$\ell(\ell+1)C_\ell/(2\pi)$')
        plt.xlim(0,500)
        fig = plt.gcf()
        fig.set_size_inches(12.5, 10.5)
        leg = plt.legend(labelspacing=0.001)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(4.0)
        filename = 'powerspectrum_comparison_all.png'
        plt.savefig('%s%s/%s/%s'%(path, outDir, outDir_comp, filename), bbox_inches='tight', format='png')
        plt.show()

        # create the histogram
        scale = hp.nside2pixarea(nside, degrees=True)
        def tickFormatter(y, pos): return '%d'%(y * scale)    # convert pixel count to area
        binsize = 0.01
        for dither in plotColor:
            if dither in coaddBundle:
                ind = np.where(coaddBundle[dither].metricValues.mask == False)[0]
                binAll = int((max(coaddBundle[dither].metricValues.data[ind])-min(coaddBundle[dither].metricValues.data[ind]))/binsize)
                plt.hist(coaddBundle[dither].metricValues.data[ind],
                         bins=binAll, label=dither, histtype='step', color=plotColor[dither])
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        nYticks = 10.
        wantedYMax = ymax*scale
        wantedYMax = 10.*np.ceil(float(wantedYMax)/10.)
        increment = 5.*np.ceil(float(wantedYMax/nYticks)/5.)
        wantedArray= np.arange(0, wantedYMax, increment)
        ax.yaxis.set_ticks(wantedArray/scale)
        ax.yaxis.set_major_formatter(FuncFormatter(tickFormatter))
        plt.xlabel('$%s$-band Coadded Depth'%filterBand)
        plt.ylabel('Area (deg$^2$)')
        fig = plt.gcf()
        fig.set_size_inches(12.5, 10.5)
        leg = plt.legend(labelspacing=0.001, loc=2)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(2.0)
        filename = 'histogram_comparison.png'
        plt.savefig('%s%s/%s/%s'%(path, outDir, outDir_comp, filename), bbox_inches='tight', format='png')
        plt.show()
        # ------------------------------------------------------------------------
        # plot power spectra for the separte panel
        totKeys = len(list(coaddBundle.keys()))
        if (totKeys>1):
            plt.clf()
            nCols = 2
            nRows = int(np.ceil(float(totKeys)/nCols))
            fig, ax = plt.subplots(nRows,nCols)
            plotRow = 0
            plotCol = 0
            for dither in list(plotColor.keys()):
                if dither in list(coaddBundle.keys()):
                    ell = np.arange(np.size(cl[dither]))
                    ax[plotRow, plotCol].plot(ell, (cl[dither]*ell*(ell+1))/2.0/np.pi,
                                              color=plotColor[dither], label=dither)
                    if (plotRow==nRows-1):
                        ax[plotRow, plotCol].set_xlabel(r'$\ell$')
                    ax[plotRow, plotCol].set_ylabel(r'$\ell(\ell+1)C_\ell/(2\pi)$')
                    ax[plotRow, plotCol].yaxis.set_major_locator(MaxNLocator(3))
                    if (dither != 'NoDither'):
                        ax[plotRow, plotCol].set_ylim(0,0.0035)
                    ax[plotRow, plotCol].set_xlim(0,500)
                    plotRow += 1
                    if (plotRow > nRows-1):
                        plotRow = 0
                        plotCol += 1
            fig.set_size_inches(20,int(nRows*30/7.))
            filename = 'powerspectrum_sepPanels.png'
            plt.savefig('%s%s/%s/%s'%(path, outDir, outDir_comp, filename), bbox_inches='tight', format='png')
            plt.show()
    return coaddBundle, outDir
