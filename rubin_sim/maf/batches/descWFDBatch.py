import numpy as np
import healpy as hp
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.plots as plots
import rubin_sim.maf.maps as maps
import rubin_sim.maf.metricBundles as mb
from .common import standardSummary, filterList
from .colMapDict import ColMapDict
from rubin_sim.maf.mafContrib.lssmetrics.depthLimitedNumGalMetric import DepthLimitedNumGalMetric
from rubin_sim.maf.mafContrib import StaticProbesFoMEmulatorMetric

__all__ = ['descWFDBatch', 'tdcBatch']


def descWFDBatch(colmap=None, runName='opsim', nside=64,
                 bandpass='i', nfilters_needed=6, lim_ebv=0.2,
                 mag_cuts=None):

    if mag_cuts is None:
        mag_cuts = {1: 24.75 - 0.1, 3: 25.35 - 0.1, 6: 25.72 - 0.1, 10: 26.0 - 0.1}

    # The options to add additional sql constraints are removed for now.
    if colmap is None:
        colmap = ColMapDict('fbs')

    # Calculate a subset of DESC WFD-related metrics.
    displayDict = {'group': 'Cosmology'}
    subgroupCount = 1

    standardStats = standardSummary(withCount=False)
    subsetPlots = [plots.HealpixSkyMap(), plots.HealpixHistogram()]

    if not isinstance(mag_cuts, dict):
        if isinstance(mag_cuts, float) or isinstance(mag_cuts, int):
            mag_cuts = {10: mag_cuts}
        else:
            raise TypeError()
    yrs = list(mag_cuts.keys())
    maxYr = max(yrs)

    # One of the primary concerns for DESC WFD metrics is to add dust extinction and coadded depth limits
    # as well as to get some coverage in all 6 bandpasses.
    # These cuts figure into many of the general metrics.

    displayDict['subgroup'] = f'{subgroupCount}: Static Science'
    ## Static Science
    # Calculate the static science metrics - effective survey area, mean/median coadded depth, stdev of
    # coadded depth and the 3x2ptFoM emulator.

    dustmap = maps.DustMap(nside=nside, interp=False)
    pix_area = hp.nside2pixarea(nside, degrees=True)
    summaryMetrics = [metrics.MeanMetric(), metrics.MedianMetric(), metrics.RmsMetric(),
                      metrics.CountRatioMetric(normVal=1/pix_area, metricName='Effective Area (deg)')]
    bundleList = []
    displayDict['order'] = 0
    for yr_cut in yrs:
        ptsrc_lim_mag_i_band = mag_cuts[yr_cut]
        sqlconstraint = 'night <= %s' % (yr_cut * 365.25)
        sqlconstraint += ' and note not like "DD%"'
        metadata = f'{bandpass} band non-DD year {yr_cut}'
        ThreebyTwoSummary_simple = metrics.StaticProbesFoMEmulatorMetricSimple(nside=nside,
                                                                               year=yr_cut,
                                                                               metricName='3x2ptFoM_simple')
        ThreebyTwoSummary = StaticProbesFoMEmulatorMetric(nside=nside, metricName='3x2ptFoM')

        m = metrics.ExgalM5_with_cuts(m5Col=colmap['fiveSigmaDepth'], filterCol=colmap['filter'],
                                      lsstFilter=bandpass, nFilters=nfilters_needed,
                                      extinction_cut=lim_ebv, depth_cut=ptsrc_lim_mag_i_band)
        s = slicers.HealpixSlicer(nside=nside, useCache=False)
        caption = f'Cosmology/Static Science metrics are based on evaluating the region of '
        caption += f'the sky that meets the requirements (in year {yr_cut} of coverage in '
        caption += f'all {nfilters_needed}, a lower E(B-V) value than {lim_ebv}, and at '
        caption += f'least a coadded depth of {ptsrc_lim_mag_i_band} in {bandpass}. '
        caption += f'From there the effective survey area, coadded depth, standard deviation of the depth, '
        caption += f'and a 3x2pt static science figure of merit emulator are calculated using the '
        caption += f'dust-extincted coadded depth map (over that reduced footprint).'
        displayDict['caption'] = caption
        bundle = mb.MetricBundle(m, s, sqlconstraint, mapsList=[dustmap], metadata=metadata,
                                 summaryMetrics=summaryMetrics + [ThreebyTwoSummary, ThreebyTwoSummary_simple],
                                 displayDict=displayDict)
        displayDict['order'] += 1
        bundleList.append(bundle)



    ## LSS Science
    # The only metric we have from LSS is the NGals metric - which is similar to the GalaxyCountsExtended
    # metric, but evaluated only on the depth/dust cuts footprint.
    subgroupCount += 1
    displayDict['subgroup'] = f'{subgroupCount}: LSS'
    displayDict['order'] = 0
    plotDict = {'nTicks': 5}
    # Have to include all filters in query, so that we check for all-band coverage.
    # Galaxy numbers calculated using 'bandpass' images only though.
    sqlconstraint = f'note not like "DD%"'
    metadata = f'{bandpass} band galaxies non-DD'
    metric = DepthLimitedNumGalMetric(m5Col=colmap['fiveSigmaDepth'], filterCol=colmap['filter'],
                                      nside=nside, filterBand=bandpass, redshiftBin='all',
                                      nfilters_needed=nfilters_needed,
                                      lim_mag_i_ptsrc=mag_cuts[maxYr], lim_ebv=lim_ebv)
    summary = [metrics.AreaSummaryMetric(area=18000, reduce_func=np.sum, decreasing=True,
                                         metricName='N Galaxies (18k)')]
    summary.append(metrics.SumMetric(metricName='N Galaxies (all)'))
    slicer = slicers.HealpixSlicer(nside=nside, useCache=False)
    bundle = mb.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                             metadata=metadata, mapsList=[dustmap],
                             displayDict=displayDict, summaryMetrics=summary,
                             plotFuncs=subsetPlots)
    bundleList.append(bundle)

    ## WL metrics
    # Calculates the number of visits per pointing, after removing parts of the footprint due to dust/depth
    subgroupCount += 1
    displayDict['subgroup'] = f'{subgroupCount}: WL'
    displayDict['order'] = 0
    sqlconstraint = f'note not like "DD%" and filter = "{bandpass}"'
    metadata = f'{bandpass} band non-DD'
    minExpTime = 15
    m = metrics.WeakLensingNvisits(m5Col=colmap['fiveSigmaDepth'], expTimeCol=colmap['exptime'],
                                   lsstFilter=bandpass, depthlim=mag_cuts[maxYr],
                                   ebvlim=lim_ebv, min_expTime=minExpTime, metricName='WeakLensingNvisits')
    s = slicers.HealpixSlicer(nside=nside, useCache=False)
    displayDict['caption'] = f'The number of visits per pointing, over the same reduced footprint as '
    displayDict['caption'] += f'described above. A cutoff of {minExpTime} removes very short visits.'
    displayDict['order'] = 1
    bundle = mb.MetricBundle(m, s, sqlconstraint, mapsList=[dustmap], metadata=metadata,
                             summaryMetrics=standardStats, displayDict=displayDict)
    bundleList.append(bundle)

    subgroupCount += 1
    displayDict['subgroup'] = f'{subgroupCount}: Camera Rotator'
    displayDict['caption'] = 'Kuiper statistic (0 is uniform, 1 is delta function) of the '
    slicer = slicers.HealpixSlicer(nside=nside)
    metric1 = metrics.KuiperMetric('rotSkyPos')
    metric2 = metrics.KuiperMetric('rotTelPos')
    filterlist, colors, filterorders, filtersqls, filtermetadata = filterList(all=False,
                                                                              extraSql=None,
                                                                              extraMetadata=None)
    for f in filterlist:
        for m in [metric1, metric2]:
            plotDict = {'color': colors[f]}
            displayDict['order'] = filterorders[f]
            displayDict['caption'] += f"{m.colname} for visits in {f} band."
            bundleList.append(mb.MetricBundle(m, slicer, filtersqls[f], plotDict=plotDict,
                                              displayDict=displayDict, summaryMetrics=standardStats,
                                              plotFuncs=subsetPlots))

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def tdcBatch(colmap=None, runName='opsim', nside=64,
             extraSql=None, extraMetadata=None):
    # The options to add additional sql constraints are removed for now.
    if colmap is None:
        colmap = ColMapDict('fbs')

    # Calculate a subset of DESC WFD-related metrics.
    displayDict = {'group': 'Strong Lensing'}
    displayDict['subgroup'] = 'Lens Time Delay'

    subsetPlots = [plots.HealpixSkyMap(), plots.HealpixHistogram()]

    summaryMetrics = [metrics.MeanMetric(), metrics.MedianMetric(), metrics.RmsMetric()]
    # Ideally need a way to do better on calculating the summary metrics for the high accuracy area.

    slicer = slicers.HealpixSlicer(nside=nside, useCache=False)
    tdcMetric = metrics.TdcMetric(metricName='TDC',  mjdCol=colmap['mjd'],
                                  nightCol=colmap['night'], filterCol=colmap['filter'],
                                  m5Col=colmap['fiveSigmaDepth'])
    dustmap = maps.DustMap(nside=nside, interp=False)
    bundle = mb.MetricBundle(tdcMetric, slicer, constraint=extraSql, metadata=extraMetadata,
                             displayDict=displayDict, plotFuncs=subsetPlots, mapsList=[dustmap],
                             summaryMetrics=summaryMetrics)

    bundleList = [bundle]

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)
