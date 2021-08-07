#####################################################################################################
# Purpose: calculate artificial structure, i.e. fluctuations in galaxy counts, resulting from
# imperfect observing strategy (OS). Includes the functionality to account for dust extinction,
# photometric calibration errors (simple ansatz implemented here), individual redshift bins (see
# GalaxyCountsMetric_extended for details), as well as poisson noise in the galaxy counts.
#
# Basic workflow, assuming all the functionalities are used:
#       1. HEALpix slicers are set up for survey strategies.
#       2. Using GalaxyCountMetric_extended, which handles dust extinction and calculates galaxy counts
#          based on redshift-bin-specific powerlaws, galaxy counts are found for each HEALpix pixel.
#       3. The shallow borders are masked (based on user-specified 'pixel radius').
#       4. Photometric calibration errors are calculated.
#       5. The galaxy counts in each pixel are recalculated using GalaxyCounts_withPixelCalibration
#          since the calibration errors modify the upper limit on the integral used to calculate
#          galaxy counts. GalaxyCounts_withPixelCalibration takes in each pixel's modified integration
#          limit individually.
#       6. Poisson noise is added to the galaxy counts.
#       7. Fluctuations in the galaxy counts are calculated.
#
# For each pixel i, the photometric calibration errors are modeled as del_i= k*z_i/sqrt(nObs_i),
# where z_i is the average seeing the pixel minus avgSeeing across map, nObs is the number of observations,
# and k is a constant such that var(del_i)= (0.01)^2 -- 0.01 in accordance with LSST goal for relative
# photometric calibration.
#
# Most of the functionalities can be turned on/off, and plots and data can be saved at various points.
# Bordering masking adds significant run time as does the incorporation of photometric calibration
# errors. See the method descrpition for further details.
#
# Humna Awan: humna.awan@rutgers.edu
#####################################################################################################
import matplotlib.pyplot as plt
import numpy as np
import os
import healpy as hp
try:
    from sympy.solvers import solve
    from sympy import Symbol
except ImportError:
    pass
import copy
import time
from matplotlib.ticker import FuncFormatter
import datetime

import rubin_sim.maf
import rubin_sim.maf.db as db
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.stackers as mafStackers   # stackers in sims_maf
import rubin_sim.maf.plots as plots
import rubin_sim.maf.metricBundles as metricBundles
import rubin_sim.maf.maps as maps

from rubin_sim.maf.mafContrib.LSSObsStrategy.galaxyCountsMetric_extended import GalaxyCountsMetric_extended \
    as GalaxyCountsMetric
from rubin_sim.maf.mafContrib.LSSObsStrategy.galaxyCounts_withPixelCalibration import GalaxyCounts_withPixelCalibration \
    as GalaxyCounts_0ptErrors
from rubin_sim.maf.mafContrib.LSSObsStrategy.maskingAlgorithmGeneralized import maskingAlgorithmGeneralized
from rubin_sim.maf.mafContrib.LSSObsStrategy.numObsMetric import NumObsMetric
from rubin_sim.maf.mafContrib.LSSObsStrategy.saveBundleData_npzFormat import saveBundleData_npzFormat

from rubin_sim.maf.mafContrib.LSSObsStrategy.constantsForPipeline import powerLawConst_a, plotColor

__all__ = ['artificialStructureCalculation']


def artificialStructureCalculation(path, upperMagLimit, dbfile, runName,
                                   noDithOnly=False,
                                   bestDithOnly=False,
                                   specifiedDith=None,
                                   nside=128, filterBand='i',
                                   cutOffYear=None, redshiftBin='all',
                                   CFHTLSCounts=False, normalizedMockCatalogCounts=True,
                                   includeDustExtinction=True, saveRawNumGalData=True,
                                   pixelRadiusForMasking=5, saveNumGalDataAfterMasking=False,
                                   include0ptErrors=True,
                                   print0ptInformation=True,
                                   plot0ptPlots=True, show0ptPlots=False, save0ptPlots=True,
                                   saveNumGalDataAfter0pt=False,
                                   addPoissonNoise=True, saveDeltaNByNData=True,
                                   saveClsForDeltaNByN=True,
                                   show_comp_plots=False, return_stuff=False):
    """

    Calculate artificial structure, i.e. fluctuations in galaxy counts dN/N, resulting due
    to imperfect observing strategy (OS).
    - Creates an output directory for subdirectories containing the specified things to save.
    - Prints out execution time at key steps (after border-masking, incorporating calibration errors, etc.)
    - Returns the metricBundle object containing the calculated dN/N, the output directory name,
    the resultsDb object, and (if include0ptErrors=True)  calibration errors for each survey strategy.

    Parameters
    -------------
    path: str
        path to the main directory where output directory is to be saved.
    upperMagLimit: float
        upper limit on magnitude when calculating the galaxy counts.
    dbfile: str
        path to the OpSim output file, e.g. to a copy of enigma_1189
    runName: str
        run name tag to identify the output of specified OpSim output.
        Since new OpSim outputs have different columns, the runName for enigma_1189 **must**
        be 'enigma1189'; can be anything for other outputs, e.g. 'minion1016'
    noDithOnly: `bool`
        set to True if only want to consider the undithered survey. Default: False
    bestDithOnly: `bool`
        set to True if only want to consider RandomDitherFieldPerVisit.
        Default: False
    specifiedDith: str
        specific dither strategy to run; could be a string or a list of strings.
        Default: None
    nside: int
        HEALpix resolution parameter. Default: 128
    filterBand: str
        any one of 'u', 'g', 'r', 'i', 'z', 'y'. Default: 'i'
    cutOffYear: int
        year cut to restrict analysis to only a subset of the survey.
        Must range from 1 to 9, or None for the full survey analysis (10 yrs).
        Default: None
    redshiftBin: str
        options include '0.<z<0.15', '0.15<z<0.37', '0.37<z<0.66, '0.66<z<1.0',
        '1.0<z<1.5', '1.5<z<2.0', '2.0<z<2.5', '2.5<z<3.0','3.0<z<3.5', '3.5<z<4.0',
        'all' for no redshift restriction (i.e. 0.<z<4.0)
        Default: 'all'
    CFHTLSCounts: `bool`
        set to True if want to calculate the total galaxy counts from CFHTLS
        powerlaw from LSST Science Book. Must be run with redshiftBin='all'
        Default: False
    normalizedMockCatalogCounts: `bool`
        set to False if  want the raw/un-normalized galaxy
        counts from mock catalogs. Default: True
    includeDustExtinction: `bool`:
        set to include dust extinction when calculating the coadded
        depth. Default: True
    saveRawNumGalData: `bool`
        set to True to save numGal data right away, i.e. before
        0pt error calibration, bordering masking, or poisson noise.
        Default: True
    pixelRadiusForMasking: int
        number of pixels to mask along the shallow border. Default: 5
    saveNumGalDataAfterMasking: `bool`
        set to True to save numGal data after border masking.
        Default: False
    include0ptErrors: `bool`
        set to True to include photometric calibration errors.
        Default: True
    print0ptInformation: `bool`
        set to True to print out some statistics (variance, the k-value, etc.)
        of the calibration errors of every dither strategy.
        Default: True
    plot0ptPlots : `bool`
        generate 0pt plots. Default True.
    saveNumGalDataAfter0pt: `bool`
        set to True to save numGal data after border masking and 0pt calibration. Default: False
    addPoissonNoise: `bool`
        set to True to add poisson noise to the galaxy counts after border masking
        and the incorporation of calibration errors. Default: True
    saveNumGalDataAfterPoisson:: `bool`
        set to True to save numGal data right away, after border masking,
        including the calibration errors, and the  poisson noise.
        Default: True
    showDeltaNByNPlots: `bool`
        set to True to show the plots related to the fluctuations in the galaxy
        counts. Will work only when plotDeltaNByN=True. Default: False
    saveDeltaNByNPlots: `bool`
        set to True to save the plots related to the fluctuations in the galaxy
        counts. Will work only when plotDeltaNByN=True. Default: True
    saveDeltaNByNData: `bool`
        set to True to save data for the the fluctuations in the galaxy counts.
        Default: True
    saveClsForDeltaNByN: `bool`
        set to True to save the power spectrum data for the the fluctuations in
        the galaxy counts. Default: True
    show_comp_plots: `bool`
        set to True if want to display the comparison plots (only valid if have more
        han one dither strategy); otherwise, the plots will be saved and not displayed.
        Default: False
    return_stuff: `bool`
        set to True to get the metricBundle object, the outDir, and resultsDb object.
        Default: False
    """
    startTime = time.time()
    # ------------------------------------------------------------------------
    # set up the metric
    galCountMetric = GalaxyCountsMetric(upperMagLimit=upperMagLimit,
                                        includeDustExtinction=includeDustExtinction,
                                        redshiftBin=redshiftBin,
                                        filterBand=filterBand,
                                        nside=nside,
                                        CFHTLSCounts=CFHTLSCounts,
                                        normalizedMockCatalogCounts=normalizedMockCatalogCounts)
    # OpSim database
    opsdb = db.OpsimDatabase(dbfile)

    # ------------------------------------------------------------------------
    # set up the outDir name
    zeropt_tag, dust_tag = '', ''
    if include0ptErrors: zeropt_tag = 'with0ptErrors'
    else: zeropt_tag = 'no0ptErrors'
        
    if includeDustExtinction: dust_tag = 'withDustExtinction'
    else: dust_tag = 'noDustExtinction'

    if cutOffYear is not None: survey_tag = '%syearCut'%(cutOffYear)
    else: survey_tag = 'fullSurveyPeriod'

    # check to make sure redshift bin is ok.
    allowedRedshiftBins = list(powerLawConst_a.keys()) + ['all']
    if redshiftBin not in allowedRedshiftBins:
        print('ERROR: Invalid redshift bin. Input bin can only be among %s.\n'%(allowedRedshiftBins))
        return
    zbin_tag = redshiftBin
    if (redshiftBin=='all'): zbin_tag = 'allRedshiftData'
        
    poisson_tag = ''
    if addPoissonNoise: poisson_tag = 'withPoissonNoise'
    else: poisson_tag = 'noPoissonNoise'
        
    counts_tag = ''
    if CFHTLSCounts:
        counts_tag = 'CFHTLSpowerLaw'
    elif normalizedMockCatalogCounts:
        counts_tag = 'normalizedGalaxyCounts'
    else:
        counts_tag = 'unnormalizedGalaxyCounts'

    outDir = f'artificialStructure_{poisson_tag}_nside{nside}'\
             f'_pixelRadiusFormasking_{pixelRadiusForMasking}_{zeropt_tag}_{dust_tag}_{filterBand}'\
             f'_{upperMagLimit}_{runName}_{survey_tag}_{zbin_tag}_{counts_tag}_directory'

    print('# outDir: %s\n'%outDir)
    if not os.path.exists('%s%s'%(path, outDir)):
        os.makedirs('%s%s'%(path, outDir))

    results_dbname = 'resultsDb_%s.db'%np.random.randint(100)
    resultsDb = db.ResultsDb(database=results_dbname, outDir='%s%s'%(path, outDir))

    # ------------------------------------------------------------------------
    # set up the sql constraint
    propIds, propTags = opsdb.fetchPropInfo()
    wfdWhere = opsdb.createSQLWhere('WFD', propTags)
    raDecInDeg = opsdb.raDecInDeg
    if cutOffYear is not None:
        nightCutOff = (cutOffYear)*365.25
        sqlconstraint = '%s and night<=%s and filter=="%s"'%(wfdWhere, nightCutOff, filterBand)
    else:
        sqlconstraint = '%s and filter=="%s"'%(wfdWhere, filterBand)
    print('# sqlconstraint: %s'%sqlconstraint)

    # ------------------------------------------------------------------------
    # create a ReadMe type file to put info in.
    update = '%s\n'%(datetime.datetime.now())
    update += '\nArtificial structure calculation with %s, %s, and %s '%(zeropt_tag, dust_tag, poisson_tag)
    update += 'for %s for %s for %s<%s. '%(survey_tag, zbin_tag, filterBand, upperMagLimit)
    update += '\nWith %s and PixelRadiusForMasking: %s.\n'%(counts_tag, pixelRadiusForMasking)
    update += '\nsqlconstraint: %s'%sqlconstraint
    update += '\nRunning with %s\n'%runName
    update += '\noutDir: %s\n'%outDir
    update += '\nMAF version: %s\n'%rubin_sim.maf.__version__

    # figure out the readme name
    readme_name = 'ReadMe'
    readmes = [f for f in os.listdir('%s%s'%(path, outDir)) if any([f.endswith('.txt')])]
    numFile = 0
    for f in readmes:
        if f.__contains__('%s_'%readme_name):
            temp = f.split('.txt')[0]
            numFile = max(numFile, int(temp.split('%s_'%readme_name)[1]))
        else:
            numFile = 1
    readme_name = 'ReadMe_%s.txt'%(numFile+1)
    readme = open('%s%s/%s'%(path, outDir, readme_name), 'w')
    readme.write(update)
    readme.close()

    # ------------------------------------------------------------------------
    # setup all the slicers. set up randomSeed for random/repRandom strategies through stackerList.
    slicer = {}
    stackerList = {}

    if specifiedDith is not None:
        # would like to add all the stackers first and then keep only the one that is specified
        bestDithOnly, noDithOnly = False, False

    if bestDithOnly:
        stackerList['RandomDitherFieldPerVisit'] = [mafStackers.RandomDitherFieldPerVisitStacker(degrees=raDecInDeg,
                                                                                                 randomSeed=1000)]
        slicer['RandomDitherFieldPerVisit'] = slicers.HealpixSlicer(lonCol='randomDitherFieldPerVisitRa',
                                                                   latCol='randomDitherFieldPerVisitDec',
                                                                   latLonDeg=raDecInDeg,
                                                                   nside=nside, useCache=False)
    else:
        slicer['NoDither'] = slicers.HealpixSlicer(lonCol='fieldRA', latCol='fieldDec', latLonDeg=raDecInDeg,
                                                      nside=nside, useCache=False)
        if not noDithOnly:
            # random dithers on different timescales
            stackerList['RandomDitherPerNight'] = [mafStackers.RandomDitherPerNightStacker(degrees=raDecInDeg,
                                                                                           randomSeed=1000)]
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
            slicer['RandomDitherPerNight'] = slicers.HealpixSlicer(lonCol='randomDitherPerNightRa',
                                                                   latCol='randomDitherPerNightDec',
                                                                  latLonDeg=raDecInDeg, nside=nside,
                                                                   useCache=False)
            slicer['RandomDitherFieldPerNight'] = slicers.HealpixSlicer(lonCol='randomDitherFieldPerNightRa',
                                                                       latCol='randomDitherFieldPerNightDec',
                                                                       latLonDeg=raDecInDeg, nside=nside,
                                                                        useCache=False)
            slicer['RandomDitherFieldPerVisit'] = slicers.HealpixSlicer(lonCol='randomDitherFieldPerVisitRa',
                                                                       latCol='randomDitherFieldPerVisitDec',
                                                                       latLonDeg=raDecInDeg, nside=nside,
                                                                        useCache=False)
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
                                                                         latLonDeg=raDecInDeg, nside=nside,
                                                                         useCache=False)
            slicer['FermatSpiralDitherFieldPerNight'] = slicers.HealpixSlicer(lonCol='fermatSpiralDitherFieldPerNightRa',
                                                                              latCol='fermatSpiralDitherFieldPerNightDec',
                                                                              latLonDeg=raDecInDeg, nside=nside,
                                                                              useCache=False)
            slicer['FermatSpiralDitherFieldPerVisit'] = slicers.HealpixSlicer(lonCol='fermatSpiralDitherFieldPerVisitRa',
                                                                              latCol='fermatSpiralDitherFieldPerVisitDec',
                                                                              latLonDeg=raDecInDeg, nside=nside,
                                                                              useCache=False)
            # hex dithers on different timescales
            slicer['SequentialHexDitherPerNight'] = slicers.HealpixSlicer(lonCol='hexDitherPerNightRa',
                                                                          latCol='hexDitherPerNightDec',
                                                                          latLonDeg=raDecInDeg, nside=nside,
                                                                          useCache=False)
            slicer['SequentialHexDitherFieldPerNight'] = slicers.HealpixSlicer(lonCol='hexDitherFieldPerNightRa',
                                                                               latCol='hexDitherFieldPerNightDec',
                                                                               latLonDeg=raDecInDeg, nside=nside,
                                                                               useCache=False)
            slicer['SequentialHexDitherFieldPerVisit'] = slicers.HealpixSlicer(lonCol='hexDitherFieldPerVisitRa',
                                                                               latCol='hexDitherFieldPerVisitDec',
                                                                               latLonDeg=raDecInDeg, nside=nside,
                                                                               useCache=False)
            # per season dithers
            slicer['PentagonDitherPerSeason'] = slicers.HealpixSlicer(lonCol='pentagonDitherPerSeasonRa',
                                                                      latCol='pentagonDitherPerSeasonDec',
                                                                     latLonDeg=raDecInDeg, nside=nside,
                                                                      useCache=False)
            slicer['PentagonDiamondDitherPerSeason'] = slicers.HealpixSlicer(lonCol='pentagonDiamondDitherPerSeasonRa',
                                                                            latCol='pentagonDiamondDitherPerSeasonDec',
                                                                            latLonDeg=raDecInDeg, nside=nside,
                                                                            useCache=False)
            slicer['SpiralDitherPerSeason'] = slicers.HealpixSlicer(lonCol='spiralDitherPerSeasonRa',
                                                                   latCol='spiralDitherPerSeasonDec',
                                                                   latLonDeg=raDecInDeg, nside=nside,
                                                                    useCache=False)
    # ------------------------------------------------------------------------
    if specifiedDith is not None:
        stackerList_, slicer_ = {}, {}
        if isinstance(specifiedDith, str):
            if specifiedDith in slicer.keys():
                if specifiedDith.__contains__('Random'):
                    # only Random dithers have a stacker object for rand seed specification
                    stackerList_[specifiedDith] = stackerList[specifiedDith]
                slicer_[specifiedDith] = slicer[specifiedDith]
        elif isinstance(specifiedDith, list):
            for specific in specifiedDith:
                if specific in slicer.keys():
                    if specific.__contains__('Random'):
                        # only Random dithers have a stacker object for rand seed specification
                        stackerList_[specific] = stackerList[specific]
                    slicer_[specific] = slicer[specific]
        else:
            err = 'Invalid value for specifiedDith: %s.'%specifiedDith
            err += 'Allowed values include one of the following:\n%s'%(slicer.keys())
            raise ValueError(err)
        stackerList, slicer = stackerList_, slicer_

    print('\nRunning the analysis for %s'%slicer.keys())
    # ------------------------------------------------------------------------
    readme = open('%s%s/%s'%(path, outDir, readme_name), 'a')
    readme.write('\nObserving strategies considered: %s\n'%(list(slicer.keys())))
    readme.close()
    # ------------------------------------------------------------------------
    # set up bundle for numGal (and later deltaN/N)
    myBundles = {}
    dustMap = maps.DustMap(interp=False, nside=nside)   # include dustMap; actual in/exclusion of dust is handled by the galaxyCountMetric
    for dither in slicer:
        if dither in stackerList:
            myBundles[dither] = metricBundles.MetricBundle(galCountMetric, slicer[dither], sqlconstraint,
                                                           stackerList=stackerList[dither],
                                                           runName=runName, metadata=dither, mapsList=[dustMap])
        else:
            myBundles[dither] = metricBundles.MetricBundle(galCountMetric, slicer[dither], sqlconstraint,
                                                           runName=runName, metadata=dither, mapsList=[dustMap])
    # ------------------------------------------------------------------------
    # run the metric/slicer combination for galaxy counts (numGal)
    print('\n# Running myBundles ...')
    bGroup = metricBundles.MetricBundleGroup(myBundles, opsdb, outDir='%s%s'%(path, outDir),
                                             resultsDb=resultsDb, saveEarly=False)
    bGroup.runAll()
    # ------------------------------------------------------------------------

     # save the raw numGal data.
    if saveRawNumGalData:
        outDir_new = 'numGalData_beforeMasking_before0pt'
        if not os.path.exists('%s%s/%s'%(path, outDir, outDir_new)):
            os.makedirs('%s%s/%s'%(path, outDir, outDir_new))
        saveBundleData_npzFormat('%s%s/%s'%(path, outDir, outDir_new), myBundles, 'numGalData_unmasked_no0pt', filterBand)
    # ------------------------------------------------------------------------
    # print out tot(numGal) associated with each strategy
    # write to the readme as well
    update = '\n# Before any border masking or photometric error calibration: '
    print(update)
    for dither in myBundles:
        ind = np.where(myBundles[dither].metricValues.mask[:] == False)[0]
        printOut = 'Total Galaxies for %s: %.9e' %(dither, sum(myBundles[dither].metricValues.data[ind]))
        update += '\n %s'%printOut
        print(printOut)
    update += '\n'
    readme = open('%s%s/%s'%(path, outDir, readme_name), 'a')
    readme.write(update)
    readme.close()
    print('\n## Time since the start of the calculation: %.2f hrs'%((time.time()-startTime)/3600.))

    # ------------------------------------------------------------------------
    # mask the edges: the data in the masked pixels is not changed
    plotHandler = plots.PlotHandler(outDir='%s%s'%(path, outDir), resultsDb=resultsDb, thumbnail=False, savefig=False)
    print('\n# Masking the edges ...')
    myBundles, borderPixelsMasked = maskingAlgorithmGeneralized(myBundles, plotHandler, 'Number of Galaxies',
                                                                nside=nside,
                                                               pixelRadius=pixelRadiusForMasking,
                                                                plotIntermediatePlots=False,
                                                               plotFinalPlots=False, printFinalInfo=True,
                                                                returnBorderIndices=True)
    # ------------------------------------------------------------------------

    # save the numGal data.
    if saveNumGalDataAfterMasking:
        outDir_new = 'numGalData_afterBorderMasking'
        if not os.path.exists('%s%s/%s'%(path, outDir, outDir_new)):
            os.makedirs('%s%s/%s'%(path, outDir, outDir_new))
        saveBundleData_npzFormat('%s%s/%s'%(path, outDir, outDir_new), myBundles, 'numGalData_masked', filterBand)
    # ------------------------------------------------------------------------
    # print out tot(numGal) associated with each strategy
    # write to the readme as well
    if (pixelRadiusForMasking!=0):
        update = '\n# After border masking: '
        print(update)
        for dither in myBundles:
            ind = np.where(myBundles[dither].metricValues.mask[:] == False)[0]
            printOut = 'Total Galaxies for %s: %.9e' %(dither, sum(myBundles[dither].metricValues.data[ind]))
            print(printOut)
            update += '\n %s'%printOut
        update += '\n'

        readme = open('%s%s/%s'%(path, outDir, readme_name), 'a')
        readme.write(update)
        readme.close()
    print('\n## Time since the start of the calculation: %.2f hrs'%((time.time()-startTime)/3600.))
    
    ################################################################################################################
    # If include 0pt errors
    # Ansatz: for each pixel i, del_i= k*z_i/sqrt(nObs_i),
    # where z_i is the average seeing the pixel minus avgSeeing across map, nObs is the number of observations,
    # and k is a constant such that var(del_i)= (0.01)^2. 0.01 for the 1% LSST goal.
    # k-constraint equation becomes: k^2*var(z_i/sqrt(nObs_i))= (0.01)^2    --- equation 1
    if include0ptErrors:
        tablename = 'SummaryAllProps'
        if tablename in opsdb.tableNames:
            colname = 'seeingFwhmEff'
            if colname not in opsdb.columnNames[tablename]:
                raise ValueError('Unclear which seeing column to use.')
        elif 'Summary' in opsdb.tableNames:
            tablename = 'Summary'
            colname = 'finSeeing'
            if colname not in opsdb.columnNames[tablename]:
                colname = 'FWHMeff'
                if colname not in opsdb.columnNames[tablename]:
                    raise ValueError('Unclear which seeing column to use.')

        meanMetric = metrics.MeanMetric(col=colname)   # for avgSeeing per HEALpix pixel
        
        nObsMetric = NumObsMetric(nside=nside)   # for numObs per HEALpix pixel
        if includeDustExtinction: coaddMetric = metrics.ExgalM5(lsstFilter=filterBand)
        else: coaddMetric = metrics.Coaddm5Metric()

        avgSeeingBundle = {}
        nObsBundle = {}
        coaddBundle = {}
        
        # can pass dustMap to metricBundle regardless of whether to include dust extinction or not. 
        # the metric choice (coadd vs. exGal) takes care of whether to use the dustMap or not.
        dustMap = maps.DustMap(interp=False, nside=nside)
        for dither in slicer:
            if dither in stackerList:
                avgSeeingBundle[dither] = metricBundles.MetricBundle(meanMetric, slicer[dither], sqlconstraint,
                                                                     stackerList=stackerList[dither],
                                                                     runName=runName, metadata=dither)
                nObsBundle[dither] = metricBundles.MetricBundle(nObsMetric, slicer[dither], sqlconstraint,
                                                                stackerList=stackerList[dither],
                                                                runName=runName, metadata=dither)
                coaddBundle[dither] = metricBundles.MetricBundle(coaddMetric, slicer[dither], sqlconstraint,
                                                                 stackerList=stackerList[dither],
                                                                 runName=runName, metadata=dither,
                                                                 mapsList=[dustMap])
            else:
                avgSeeingBundle[dither] = metricBundles.MetricBundle(meanMetric, slicer[dither], sqlconstraint,
                                                                     runName=runName, metadata=dither)
                nObsBundle[dither] = metricBundles.MetricBundle(nObsMetric, slicer[dither], sqlconstraint,
                                                                runName=runName, metadata=dither)
                coaddBundle[dither] = metricBundles.MetricBundle(coaddMetric, slicer[dither], sqlconstraint,
                                                                 runName=runName, metadata=dither,
                                                                 mapsList=[dustMap])
        print('\n# Running avgSeeingBundle ...')
        aGroup = metricBundles.MetricBundleGroup(avgSeeingBundle, opsdb, outDir='%s%s'%(path, outDir),
                                                 resultsDb=resultsDb, saveEarly=False)
        aGroup.runAll()

        print('\n# Running nObsBundle ...')
        nGroup = metricBundles.MetricBundleGroup(nObsBundle, opsdb, outDir='%s%s'%(path, outDir),
                                                 resultsDb=resultsDb, saveEarly=False)
        nGroup.runAll()

        print('\n# Running coaddBundle ...')
        cGroup = metricBundles.MetricBundleGroup(coaddBundle, opsdb, outDir='%s%s'%(path, outDir),
                                                 resultsDb=resultsDb, saveEarly=False)
        cGroup.runAll()

        # ------------------------------------------------------------------------
        # mask the border pixels
        for dither in slicer:
            avgSeeingBundle[dither].metricValues.mask[borderPixelsMasked[dither]] = True
            nObsBundle[dither].metricValues.mask[borderPixelsMasked[dither]] = True
            coaddBundle[dither].metricValues.mask[borderPixelsMasked[dither]] = True


        # ------------------------------------------------------------------------
        # calculate averageSeeing over the entrie map
        bundle = {}
        bundle['avgSeeingAcrossMap'] = metricBundles.MetricBundle(meanMetric, slicers.UniSlicer(),
                                                                  sqlconstraint,runName=runName,
                                                                  metadata='avgSeeingAcrossMap')
        bundleGroup = metricBundles.MetricBundleGroup(bundle, opsdb, outDir='%s%s'%(path, outDir),
                                                      resultsDb=resultsDb, saveEarly=False)
        bundleGroup.runAll()
        avgSeeingAcrossMap = bundle['avgSeeingAcrossMap'].metricValues.data[0]
        printOut = '\n# Average seeing across map: %s' %(avgSeeingAcrossMap)
        print(printOut)
        
        # add to the readme
        readme = open('%s%s/%s'%(path, outDir, readme_name), 'a')
        readme.write(printOut)
        readme.close()
        
        # find the zero point uncertainties: for each pixel i, del_i=k*z_i/sqrt(nObs_i),
        # where z_i is the average seeing the pixel minus avgSeeing across map, nObs is the number of observations,
        # and k is a constant such that var(del_i)=(0.01)^2.
        # k-constraint equation becomes: k^2*var(z_i/sqrt(nObs_i))=(0.01)^2    --- equation 1
        k = Symbol('k')
        zeroPtError = {}
        kValue = {}

        print('\n# 0pt calculation ansatz: \delta_i=k*z_i/sqrt{nObs_i}, where k is s.t. var(\delta_i)=(0.01)^$')
        
        if save0ptPlots:
            outDir_new = '0pt_plots'
            if not os.path.exists('%s%s/%s'%(path, outDir, outDir_new)):
                os.makedirs('%s%s/%s'%(path, outDir, outDir_new))

        # ------------------------------------------------------------------------
        # add to the readme
        readme = open('%s%s/%s'%(path, outDir, readme_name), 'a')
        readme.write('\n\n0pt Information: ')
        readme.close()
            
        for dither in avgSeeingBundle:
            z_i = avgSeeingBundle[dither].metricValues.data[:]-avgSeeingAcrossMap
            nObs_i = nObsBundle[dither].metricValues.data[:]
            ind = np.where((nObsBundle[dither].metricValues.mask == False) & \
                          (nObs_i != 0.0))[0]  # make sure the uncertainty is valid; no division by 0
            temp = np.var(z_i[ind]/np.sqrt(nObs_i[ind]))  # see equation 1
            kValue[dither] = solve(k**2*temp-0.01**2,k)[1]

            err = np.empty(len(z_i))
            err.fill(-500)   # initiate
            err[ind] = (kValue[dither]*z_i[ind])/np.sqrt(nObs_i[ind])
            zeroPtError[dither] = err

            # add to the readme
            readme = open('%s%s/%s'%(path, outDir, readme_name), 'a')
            readme.write('\nDith strategy: %s'%dither)
            readme.close()

            # ------------------------------------------------------------------------
            if print0ptInformation:
                update = '\n# %s'%dither
                ind = np.where(zeroPtError[dither] != -500)[0]
                goodError = zeroPtError[dither][ind]
                update += 'var(0pt): %s'%np.var(goodError)
                update += '\n0.01^2 - var(0pt) = %s'%((0.01)**2-np.var(goodError))
                update += '\nk-value: %s\n'%kValue[dither]
                print(update)
                # add to the readme
                readme = open('%s%s/%s'%(path, outDir, readme_name), 'a')
                readme.write(update)
                readme.close()
            # ------------------------------------------------------------------------
            if plot0ptPlots:
                # since not saving the bundle for 0pt errors, must plot out stuff without the plotBundle routine.
                ind = np.where(zeroPtError[dither] != -500)[0]
                goodError = zeroPtError[dither][ind]

                for i in range(len(goodError)):
                    goodError[i] = float(goodError[i])

                update = '\n# %s'%dither
                update += '\nMin error: %s'%min(goodError)
                update += '\nMax error: %s'%max(goodError)
                update += '\nMean error: %s'%np.mean(goodError)
                update += '\nStd of error: %s\n'%np.std(goodError)
                print(update)
                                 
                # add to the readme
                readme = open('%s%s/%s'%(path, outDir, readme_name), 'a')
                readme.write(update)
                readme.close()
                        
                # plot histogram
                binsize = 0.005
                bins = np.arange(min(goodError)-5*binsize, max(goodError)+5*binsize, binsize)
                plt.clf()
                plt.hist(goodError, bins=bins)
                plt.xlabel('Zeropoint Uncertainty')
                plt.ylabel('Counts')

                plt.title('0pt error histogram; binSize = %s; upperMagLimit = %s'%(binsize, upperMagLimit))
                if save0ptPlots:
                    filename = '0ptHistogram_%s_%s.png'%(filterBand, dither)
                    plt.savefig('%s%s/%s/%s'%(path, outDir, outDir_new, filename), format='png')
                if show0ptPlots:
                    plt.show()
                else:
                    plt.close()

                # plot skymap
                temp = copy.deepcopy(coaddBundle[dither])
                temp.metricValues.data[ind] = goodError
                temp.metricValues.mask[:] = True
                temp.metricValues.mask[ind] = False

                inSurveyIndex = np.where(temp.metricValues.mask == False)[0]
                median = np.median(temp.metricValues.data[inSurveyIndex])
                stddev = np.std(temp.metricValues.data[inSurveyIndex])

                colorMin = -0.010 #median-1.5*stddev
                colorMax = 0.010 #median+1.5*stddev
                nTicks = 5
                increment = (colorMax-colorMin)/float(nTicks)
                ticks = np.arange(colorMin+increment, colorMax, increment)

                plt.clf()
                hp.mollview(temp.metricValues.filled(temp.slicer.badval), 
                            flip='astro', rot=(0,0,0) ,
                            min=colorMin, max=colorMax, title='',cbar=False)
                hp.graticule(dpar=20, dmer=20, verbose=False)
                plt.title(dither)
                ax = plt.gca()
                im = ax.get_images()[0]
                fig= plt.gcf()
                cbaxes = fig.add_axes([0.1, 0.03, 0.8, 0.04]) # [left, bottom, width, height]
                cb = plt.colorbar(im, orientation='horizontal',
                                  ticks=ticks, format='%.3f', cax=cbaxes)
                cb.set_label('Photometric Calibration Error')
                          
                if save0ptPlots:
                    filename = '0ptSkymap_%s.png'%(dither)
                    plt.savefig('%s%s/%s/%s'%(path, outDir, outDir_new, filename),
                                bbox_inches='tight', format='png')
                    
                if show0ptPlots: plt.show()
                else: plt.close()

                # plot power spectrum
                plt.clf()
                spec = hp.anafast(temp.metricValues.filled(temp.slicer.badval), lmax=500)            
                ell = np.arange(np.size(spec))
                condition = (ell > 1)
                plt.plot(ell, (spec*ell*(ell+1))/2.0/np.pi)
                plt.title( 'Photometric Calibration Error: %s'%dither)
                plt.xlabel(r'$\ell$')
                plt.ylabel(r'$\ell(\ell+1)C_\ell/(2\pi)$')
                plt.xlim(0,500)
                
                if save0ptPlots:
                    # save power spectrum
                    filename = '0ptPowerSpectrum_%s.png'%(dither)
                    plt.savefig('%s%s/%s/%s'%(path, outDir, outDir_new, filename),
                                bbox_inches='tight', format='png')

                if show0ptPlots: plt.show()
                else: plt.close()

        print('\n## Time since the start of the calculation: %.2f hrs'%((time.time()-startTime)/3600.))

        # ------------------------------------------------------------------------
        # Now recalculate the numGal with the fluctuations in depth due to calibation uncertainties.
        print('\n# Recalculating numGal including 0pt errors on the upper mag limit .. ')
        for dither in myBundles:
            zeroPtErr = zeroPtError[dither].copy()
            inSurvey =  np.where(myBundles[dither].metricValues.mask == False)[0]   # 04/27: only look at inSurvey region
            for i in inSurvey:   # 4/27 
                if (zeroPtErr[i] != -500):   # run only when zeroPt was calculated
                    myBundles[dither].metricValues.data[i] = GalaxyCounts_0ptErrors(coaddBundle[dither].metricValues.data[i],
                                                                                    upperMagLimit+zeroPtErr[i],
                                                                                    redshiftBin=redshiftBin,
                                                                                    filterBand=filterBand, nside=nside,
                                                                                    CFHTLSCounts=CFHTLSCounts,
                                                                                    normalizedMockCatalogCounts=normalizedMockCatalogCounts)
        # ------------------------------------------------------------------------


        # save the raw numGal data.
        if saveNumGalDataAfter0pt:
            outDir_new = 'numGalData_afterBorderMasking_after0pt'
            if not os.path.exists('%s%s/%s'%(path, outDir, outDir_new)):
                os.makedirs('%s%s/%s'%(path, outDir, outDir_new))
            saveBundleData_npzFormat('%s%s/%s'%(path, outDir, outDir_new), myBundles, 'numGalData_masked_with0pt', filterBand)
        # ------------------------------------------------------------------------
        # print out tot(numGal) associated with each strategy
        # add to the read me as well
        update = '\n# After 0pt error calculation and border masking: '
        print(update)
        for dither in myBundles:
            ind = np.where(myBundles[dither].metricValues.mask[:] == False)[0]
            printOut = 'Total Galaxies for %s: %.9e' %(dither, sum(myBundles[dither].metricValues.data[ind]))
            update += '\n %s'%printOut
            print(printOut)
        update += '\n'
        readme = open('%s%s/%s'%(path, outDir, readme_name), 'a')
        readme.write(update)
        readme.close()

    print('\n## Time since the start of the calculation: %.2f hrs'%((time.time()-startTime)/3600.))

    #########################################################################################################
    # add poisson noise?
    if addPoissonNoise:
        print('\n# adding poisson noise to numGal ... ')
        for dither in myBundles:
            # make sure the values are valid; sometimes metric leaves negative numbers or nan values.
            outOfSurvey = np.where(myBundles[dither].metricValues.mask == True)[0]
            myBundles[dither].metricValues.data[outOfSurvey] = 0.0
        
            inSurvey = np.where(myBundles[dither].metricValues.mask == False)[0]
            j = np.where(myBundles[dither].metricValues.data[inSurvey] < 1.)[0]
            myBundles[dither].metricValues.data[inSurvey][j] = 0.0

            noisyNumGal = np.random.poisson(lam = myBundles[dither].metricValues.data, size=None)
            myBundles[dither].metricValues.data[:] = noisyNumGal
        # ------------------------------------------------------------------------

        # save the numGal data.
        if saveNumGalDataAfterPoisson:
            outDir_new = 'numGalData_afterBorderMasking_after0pt_afterPoisson'
            if not os.path.exists('%s%s/%s'%(path, outDir, outDir_new)):
                os.makedirs('%s%s/%s'%(path, outDir, outDir_new))
            saveBundleData_npzFormat('%s%s/%s'%(path, outDir, outDir_new),
                                     myBundles, 'numGalData_masked_with0pt_withPoisson', filterBand)
        # ------------------------------------------------------------------------
        # print out tot(numGal) associated with each strategy
        # add to the read me as well
        update = '\n# After adding poisson noise: '
        print(update)
        for dither in myBundles:
            ind = np.where(myBundles[dither].metricValues.mask[:] == False)[0]
            printOut = 'Total Galaxies for %s: %.9e' %(dither, sum(myBundles[dither].metricValues.data[ind]))
            update += '\n %s'%printOut
            print(printOut)
        update += '\n'
        readme = open('%s%s/%s'%(path, outDir, readme_name), 'a')
        readme.write(update)
        readme.close()

    print('\n## Time since the start of the calculation: %.2f hrs'%((time.time()-startTime)/3600.))
    #########################################################################################################
    plotHandler = plots.PlotHandler(outDir='%s%s'%(path, outDir), resultsDb=resultsDb, thumbnail=False, savefig=False)
    print('\n# Calculating fluctuations in the galaxy counts ...')
    # Change numGal metric data to deltaN/N
    numGal= {}
    # add to readme too
    update = '\n'
    for dither in myBundles:
        # zero out small/nan entries --- problem: should really be zeroed out by the metric ***
        j = np.where(np.isnan(myBundles[dither].metricValues.data)==True)[0]
        myBundles[dither].metricValues.data[j] = 0.0
        j = np.where(myBundles[dither].metricValues.data < 1.)[0]
        myBundles[dither].metricValues.data[j] = 0.0
        # calculate the fluctuations
        numGal[dither] = myBundles[dither].metricValues.data.copy()   # keep track of numGal for plotting purposes
        validPixel = np.where(myBundles[dither].metricValues.mask == False)[0]
        galaxyAverage = sum(numGal[dither][validPixel])/len(validPixel)

        # in place calculation of the fluctuations
        myBundles[dither].metricValues.data[:] = 0.0
        myBundles[dither].metricValues.data[validPixel] = (numGal[dither][validPixel]-galaxyAverage)/galaxyAverage
        printOut = '# Galaxy Average for %s: %s'%(dither, galaxyAverage)
        print(printOut)       
        update += '%s\n'%printOut
    
    readme = open('%s%s/%s'%(path, outDir, readme_name), 'a')
    readme.write(update)
    readme.close()

    # ------------------------------------------------------------------------
    # save the deltaN/N data
    if saveDeltaNByNData:
        outDir_new = 'deltaNByNData'
        if not os.path.exists('%s%s/%s'%(path, outDir, outDir_new)):
            os.makedirs('%s%s/%s'%(path, outDir, outDir_new))
        saveBundleData_npzFormat('%s%s/%s'%(path, outDir, outDir_new), myBundles, 'deltaNByNData_masked', filterBand)
    # ------------------------------------------------------------------------
    # Calculate total power
    # add to the read me as well
    summarymetric = metrics.TotalPowerMetric()
    update = ''
    for dither in myBundles:
        myBundles[dither].setSummaryMetrics(summarymetric)
        myBundles[dither].computeSummaryStats()
        printOut = '# Total power for %s case is %f.' %(dither, myBundles[dither].summaryValues['TotalPower'])
        print(printOut)
        update += '\n%s'%(printOut)
    update += '\n'

    readme = open('%s%s/%s'%(path, outDir, readme_name), 'a')
    readme.write(update)
    readme.close()
    # ------------------------------------------------------------------------
    # calculate the power spectra
    cl = {}
    for dither in myBundles:
        cl[dither] = hp.anafast(myBundles[dither].metricValues.filled(myBundles[dither].slicer.badval),
                                lmax=500)
    # save deltaN/N spectra?
    if saveClsForDeltaNByN:
        outDir_new = 'cls_DeltaByN'
        if not os.path.exists('%s%s/%s'%(path, outDir, outDir_new)):
            os.makedirs('%s%s/%s'%(path, outDir, outDir_new))
        
        for dither in myBundles:
            filename = 'cls_deltaNByN_%s_%s'%(filterBand, dither)
            np.save('%s%s/%s/%s'%(path, outDir, outDir_new, filename), cl[dither])
    
    ##########################################################################################################
    # Plots for the fluctuations: power spectra, histogram
    if len(list(myBundles.keys()))>1:
        outDir_new = 'artificialFluctuationsComparisonPlots'
        if not os.path.exists('%s%s/%s'%(path, outDir, outDir_new)):
            os.makedirs('%s%s/%s'%(path, outDir, outDir_new))
        # ------------------------------------------------------------------------
        # power spectra
        for dither in myBundles:
            ell = np.arange(np.size(cl[dither]))
            condition = (ell > 1)
            plt.plot(ell, (cl[dither]*ell*(ell+1))/2.0/np.pi,
                     color=plotColor[dither], linestyle='-', label=dither)
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$\ell(\ell+1)C_\ell/(2\pi)$')
        plt.xlim(0,500)
        leg = plt.legend(labelspacing=0.001)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(2.0)
        plt.savefig('%s%s/%s/powerspectrum_comparison.png'%(path, outDir, outDir_new), format='png')
        if show_comp_plots:
            plt.show()
        else:
            plt.close('all')
        # ------------------------------------------------------------------------
        # create the histogram
        scale = hp.nside2pixarea(nside, degrees=True)
        def tickFormatter(y, pos):
            return '%d'%(y * scale)    # convert pixel count to area

        for dither in myBundles:
            ind = np.where(myBundles[dither].metricValues.mask == False)[0]
            binsize = 0.01
            binAll = int((max(myBundles[dither].metricValues.data[ind])-
                          min(myBundles[dither].metricValues.data[ind]))/binsize)
            plt.hist(myBundles[dither].metricValues.data[ind], bins=binAll, label=dither,
                     histtype='step', color=plotColor[dither])
        #plt.xlim(-0.6,1.2)
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        nYticks= 10.
        wantedYMax = ymax*scale
        wantedYMax = 10.*np.ceil(float(wantedYMax)/10.)
        increment = 5.*np.ceil(float(wantedYMax/nYticks)/5.)
        wantedArray = np.arange(0, wantedYMax, increment)
        ax.yaxis.set_ticks(wantedArray/scale)
        ax.yaxis.set_major_formatter(FuncFormatter(tickFormatter))
        plt.xlabel(r'$\mathrm{\Delta N/\overline{N}}$')
        plt.ylabel('Area (deg$^2$)')
        leg = plt.legend(labelspacing=0.001, bbox_to_anchor=(1, 1))
        for legobj in leg.legendHandles:
            legobj.set_linewidth(2.0)
        plt.savefig('%s%s/%s/histogram_comparison.png'%(path, outDir, outDir_new),
                    bbox_inches='tight', format='png')
        if show_comp_plots:
            plt.show()
        else:
            plt.close('all')

    # now remove the results db object -- useless
    os.remove('%s%s/%s'%(path, outDir, results_dbname))
    print('Removed %s from outDir'%(results_dbname))

    # all done. final update.
    update = '\n## All done. Time since the start of the calculation: %.2f hrs'%((time.time()-startTime)/3600.)
    print(update)
    readme = open('%s%s/%s'%(path, outDir, readme_name), 'a')
    readme.write(update)
    readme.close()

    if return_stuff:
        if include0ptErrors:
            return myBundles, outDir, resultsDb, zeroPtError
        else:
            return myBundles, outDir, resultsDb
