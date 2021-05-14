import numpy as np
import healpy as hp
from rubin_sim.utils import hpid2RaDec, angularSeparation
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.plots as plots
import rubin_sim.maf.maps as maps
import rubin_sim.maf.metricBundles as mb
from .common import standardSummary, filterList, combineMetadata
from .colMapDict import ColMapDict
from .srdBatch import fOBatch, astrometryBatch, rapidRevisitBatch
from .descWFDBatch import descWFDBatch


__all__ = ['scienceRadarBatch']


def scienceRadarBatch(colmap=None, runName='opsim', extraSql=None, extraMetadata=None, nside=64,
                      benchmarkArea=18000, benchmarkNvisits=825, DDF=True):
    """A batch of metrics for looking at survey performance relative to the SRD and the main
    science drivers of LSST.

    Parameters
    ----------

    """
    # Hide dependencies
    from mafContrib.LSSObsStrategy.galaxyCountsMetric_extended import GalaxyCountsMetric_extended
    from mafContrib import (Plasticc_metric, plasticc_slicer, load_plasticc_lc,
                            TdePopMetric, generateTdePopSlicer,
                            generateMicrolensingSlicer, MicrolensingMetric)

    if colmap is None:
        colmap = ColMapDict('fbs')

    if extraSql is None:
        extraSql = ''
    if extraSql == '':
        joiner = ''
    else:
        joiner = ' and '

    bundleList = []
    # Get some standard per-filter coloring and sql constraints
    filterlist, colors, filterorders, filtersqls, filtermetadata = filterList(all=False,
                                                                              extraSql=extraSql,
                                                                              extraMetadata=extraMetadata)

    standardStats = standardSummary(withCount=False)

    healslicer = slicers.HealpixSlicer(nside=nside)
    subsetPlots = [plots.HealpixSkyMap(), plots.HealpixHistogram()]

    # Load up the plastic light curves - SNIa-normal are loaded in descWFDBatch
    models = ['SNIa-normal', 'KN']
    plasticc_models_dict = {}
    for model in models:
        plasticc_models_dict[model] = list(load_plasticc_lc(model=model).values())

    #########################
    # SRD, DM, etc
    #########################
    fOb = fOBatch(runName=runName, colmap=colmap, extraSql=extraSql, extraMetadata=extraMetadata,
                  benchmarkArea=benchmarkArea, benchmarkNvisits=benchmarkNvisits)
    astromb = astrometryBatch(runName=runName, colmap=colmap, extraSql=extraSql, extraMetadata=extraMetadata)
    rapidb = rapidRevisitBatch(runName=runName, colmap=colmap, extraSql=extraSql, extraMetadata=extraMetadata)

    # loop through and modify the display dicts - set SRD as group and their previous 'group' as the subgroup
    temp_list = []
    for key in fOb:
        temp_list.append(fOb[key])
    for key in astromb:
        temp_list.append(astromb[key])
    for key in rapidb:
        temp_list.append(rapidb[key])
    for metricb in temp_list:
        metricb.displayDict['subgroup'] = metricb.displayDict['group'].replace('SRD', '').lstrip(' ')
        metricb.displayDict['group'] = 'SRD'
    bundleList.extend(temp_list)

    displayDict = {'group': 'SRD', 'subgroup': 'Year Coverage', 'order': 0,
                   'caption': 'Number of years with observations.'}
    slicer = slicers.HealpixSlicer(nside=nside)
    metric = metrics.YearCoverageMetric()
    for f in filterlist:
        plotDict = {'colorMin': 7, 'colorMax': 10, 'color': colors[f]}
        summary = [metrics.AreaSummaryMetric(area=18000, reduce_func=np.mean, decreasing=True,
                                             metricName='N Seasons (18k) %s' % f)]
        bundleList.append(mb.MetricBundle(metric, slicer, filtersqls[f],
                                          plotDict=plotDict, metadata=filtermetadata[f],
                                          displayDict=displayDict, summaryMetrics=summary))

    #########################
    # Solar System
    #########################
    # Generally, we need to run Solar System metrics separately; they're a multi-step process.


    #########################
    # Galaxies
    #########################

    displayDict = {'group': 'Galaxies', 'subgroup': 'Galaxy Counts', 'order': 0, 'caption': None}
    plotDict = {'percentileClip': 95., 'nTicks': 5}
    sql = extraSql + joiner + 'filter="i"'
    metadata = combineMetadata(extraMetadata, 'i band')
    metric = GalaxyCountsMetric_extended(filterBand='i', redshiftBin='all', nside=nside)
    summary = [metrics.AreaSummaryMetric(area=18000, reduce_func=np.sum, decreasing=True,
                                         metricName='N Galaxies (18k)')]
    summary.append(metrics.SumMetric(metricName='N Galaxies (all)'))
    # make sure slicer has cache off
    slicer = slicers.HealpixSlicer(nside=nside, useCache=False)
    displayDict['caption'] = 'Number of galaxies across the sky, in i band. Generally, full survey footprint.'
    bundle = mb.MetricBundle(metric, slicer, sql, plotDict=plotDict,
                             metadata=metadata,
                             displayDict=displayDict, summaryMetrics=summary,
                             plotFuncs=subsetPlots)
    bundleList.append(bundle)
    displayDict['order'] += 1


    #########################
    # Cosmology
    #########################

    # note the desc batch does not currently take the extraSql or extraMetadata arguments.
    descBundleDict = descWFDBatch(colmap=colmap, runName=runName, nside=nside)
    for d in descBundleDict:
        bundleList.append(descBundleDict[d])

    #########################
    # Variables and Transients
    #########################
    displayDict = {'group': 'Variables/Transients',
                   'subgroup': 'Periodic Stars',
                   'order': 0, 'caption': None}
    for period in [0.5, 1, 2,]:
        for magnitude in [21., 24.]:
            amplitudes = [0.05, 0.1, 1.0]
            periods = [period] * len(amplitudes)
            starMags = [magnitude] * len(amplitudes)

            plotDict = {'nTicks': 3, 'colorMin': 0, 'colorMax': 3, 'xMin': 0, 'xMax': 3}
            metadata = combineMetadata('P_%.1f_Mag_%.0f_Amp_0.05-0.1-1' % (period, magnitude),
                                       extraMetadata)
            sql = None
            displayDict['caption'] = 'Metric evaluates if a periodic signal of period %.1f days could ' \
                                     'be detected for an r=%i star. A variety of amplitudes of periodicity ' \
                                     'are tested: [1, 0.1, and 0.05] mag amplitudes, which correspond to ' \
                                     'metric values of [1, 2, or 3]. ' % (period, magnitude)
            metric = metrics.PeriodicDetectMetric(periods=periods, starMags=starMags,
                                                  amplitudes=amplitudes,
                                                  metricName='PeriodDetection')
            bundle = mb.MetricBundle(metric, healslicer, sql, metadata=metadata,
                                     displayDict=displayDict, plotDict=plotDict,
                                     plotFuncs=subsetPlots, summaryMetrics=standardStats)
            bundleList.append(bundle)
            displayDict['order'] += 1

    # XXX add some PLASTICC metrics for kilovnova and tidal disruption events.
    displayDict['subgroup'] = 'KN'
    displayDict['caption'] = 'Fraction of Kilonova (from PLASTICC)'
    displayDict['order'] = 0
    slicer = plasticc_slicer(plcs=plasticc_models_dict['KN'], seed=43, badval=0)
    metric = Plasticc_metric(metricName='KN')
    plotFuncs = [plots.HealpixSkyMap()]
    summary_stats = [metrics.MeanMetric(maskVal=0)]
    bundle = mb.MetricBundle(metric, slicer, extraSql, runName=runName, summaryMetrics=summary_stats,
                             plotFuncs=plotFuncs, metadata=extraMetadata,
                             displayDict=displayDict)
    bundleList.append(bundle)

    # Tidal Disruption Events
    displayDict['subgroup'] = 'TDE'
    displayDict['caption'] = 'TDE lightcurves that could be identified'

    metric = TdePopMetric()
    slicer = generateTdePopSlicer()
    sql = ''
    plotDict = {'reduceFunc': np.sum, 'nside': 128}
    plotFuncs = [plots.HealpixSkyMap()]
    bundle = mb.MetricBundle(metric, slicer, sql, runName=runName,
                             plotDict=plotDict, plotFuncs=plotFuncs,
                             summaryMetrics=[metrics.MeanMetric(maskVal=0)],
                             displayDict=displayDict)
    bundleList.append(bundle)


    # Microlensing events
    displayDict['subgroup'] = 'Microlensing'
    displayDict['caption'] = 'Fast microlensing events'

    plotDict = {'nside': 128}
    sql = ''
    slicer = generateMicrolensingSlicer(min_crossing_time=1, max_crossing_time=10)
    metric = MicrolensingMetric(metricName='Fast Microlensing')
    bundle = mb.MetricBundle(metric, slicer, sql, runName=runName,
                             summaryMetrics=[metrics.MeanMetric(maskVal=0)],
                             plotFuncs=[plots.HealpixSkyMap()], metadata=extraMetadata,
                             displayDict=displayDict, plotDict=plotDict)
    bundleList.append(bundle)

    displayDict['caption'] = 'Slow microlensing events'
    slicer = generateMicrolensingSlicer(min_crossing_time=100, max_crossing_time=1500)
    metric = MicrolensingMetric(metricName='Slow Microlensing')
    bundle = mb.MetricBundle(metric, slicer, sql, runName=runName,
                             summaryMetrics=[metrics.MeanMetric(maskVal=0)],
                             plotFuncs=[plots.HealpixSkyMap()], metadata=extraMetadata,
                             displayDict=displayDict, plotDict=plotDict)
    bundleList.append(bundle)

    #########################
    # Milky Way
    #########################

    displayDict = {'group': 'Milky Way', 'subgroup': ''}

    displayDict['subgroup'] = 'N stars'
    slicer = slicers.HealpixSlicer(nside=nside, useCache=False)
    sum_stats = [metrics.SumMetric(metricName='Total N Stars, crowding')]
    for f in filterlist:
        stellar_map = maps.StellarDensityMap(filtername=f)
        displayDict['order'] = filterorders[f]
        displayDict['caption'] = 'Number of stars in %s band with an measurement error due to crowding ' \
                                 'of less than 0.2 mag' % f
        # Configure the NstarsMetric - note 'filtername' refers to the filter in which to evaluate crowding
        metric = metrics.NstarsMetric(crowding_error=0.2, filtername=f, ignore_crowding=False,
                                      seeingCol=colmap['seeingGeom'], m5Col=colmap['fiveSigmaDepth'],
                                      maps=[])
        plotDict = {'nTicks': 5, 'logScale': True, 'colorMin': 100}
        bundle = mb.MetricBundle(metric, slicer, filtersqls[f], runName=runName,
                                 summaryMetrics=sum_stats,
                                 plotFuncs=subsetPlots, plotDict=plotDict,
                                 displayDict=displayDict, mapsList=[stellar_map])
        bundleList.append(bundle)


    slicer = slicers.HealpixSlicer(nside=nside, useCache=False)
    sum_stats = [metrics.SumMetric(metricName='Total N Stars, no crowding')]
    for f in filterlist:
        stellar_map = maps.StellarDensityMap(filtername=f)
        displayDict['order'] = filterorders[f]
        displayDict['caption'] = 'Number of stars in %s band with an measurement error ' \
                                 'of less than 0.2 mag, not considering crowding' % f
        # Configure the NstarsMetric - note 'filtername' refers to the filter in which to evaluate crowding
        metric = metrics.NstarsMetric(crowding_error=0.2, filtername=f, ignore_crowding=True,
                                      seeingCol=colmap['seeingGeom'], m5Col=colmap['fiveSigmaDepth'],
                                      metricName='Nstars_no_crowding', maps=[])
        plotDict = {'nTicks': 5, 'logScale': True, 'colorMin': 100}
        bundle = mb.MetricBundle(metric, slicer, filtersqls[f], runName=runName,
                                 summaryMetrics=sum_stats,
                                 plotFuncs=subsetPlots, plotDict=plotDict,
                                 displayDict=displayDict, mapsList=[stellar_map])
        bundleList.append(bundle)


    #########################
    # DDF
    #########################
    if DDF:
        # Hide this import to avoid adding a dependency.
        from rubin_sim.featureScheduler.surveys import generate_dd_surveys, Deep_drilling_survey
        ddf_surveys = generate_dd_surveys()

        # Add on the Euclid fields
        # XXX--to update. Should have a spot where all the DDF locations are stored.
        ddf_surveys.append(Deep_drilling_survey([], 58.97, -49.28, survey_name='DD:EDFSa'))
        ddf_surveys.append(Deep_drilling_survey([], 63.6, -47.60, survey_name='DD:EDFSb'))

        # For doing a high-res sampling of the DDF for co-adds
        ddf_radius = 1.8  # Degrees
        ddf_nside = 512

        ra, dec = hpid2RaDec(ddf_nside, np.arange(hp.nside2npix(ddf_nside)))

        displayDict = {'group': 'DDF depths', 'subgroup': None}

        for survey in ddf_surveys:
            displayDict['subgroup'] = survey.survey_name
            # Crop off the u-band only DDF
            if survey.survey_name[0:4] != 'DD:u':
                dist_to_ddf = angularSeparation(ra, dec, np.degrees(survey.ra), np.degrees(survey.dec))
                goodhp = np.where(dist_to_ddf <= ddf_radius)
                slicer = slicers.UserPointsSlicer(ra=ra[goodhp], dec=dec[goodhp], useCamera=False)
                for f in filterlist:
                    metric = metrics.Coaddm5Metric(metricName=survey.survey_name + ', ' + f)
                    summary = [metrics.MedianMetric(metricName='Median depth ' + survey.survey_name+', ' + f)]
                    plotDict = {'color': colors[f]}
                    sql = filtersqls[f]
                    displayDict['order'] = filterorders[f]
                    displayDict['caption'] = 'Coadded m5 depth in %s band.' % (f)
                    bundle = mb.MetricBundle(metric, slicer, sql, metadata=filtermetadata[f],
                                             displayDict=displayDict, summaryMetrics=summary,
                                             plotFuncs=[], plotDict=plotDict)
                    bundleList.append(bundle)

        displayDict = {'group': 'DDF Transients', 'subgroup': None}
        for survey in ddf_surveys:
            displayDict['subgroup'] = survey.survey_name
            if survey.survey_name[0:4] != 'DD:u':
                slicer = plasticc_slicer(plcs=plasticc_models_dict['SNIa-normal'], seed=42,
                                         ra_cen=survey.ra, dec_cen=survey.dec, radius=np.radians(3.),
                                         useCamera=False)
                metric = Plasticc_metric(metricName=survey.survey_name+' SNIa')
                sql = extraSql
                summary_stats = [metrics.MeanMetric(maskVal=0)]
                plotFuncs = [plots.HealpixSkyMap()]
                bundle = mb.MetricBundle(metric, slicer, sql, runName=runName,
                                         summaryMetrics=summary_stats,
                                         plotFuncs=plotFuncs, metadata=extraMetadata,
                                         displayDict=displayDict)
                bundleList.append(bundle)
                displayDict['order'] = 10

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    bundleDict = mb.makeBundlesDictFromList(bundleList)

    return bundleDict
