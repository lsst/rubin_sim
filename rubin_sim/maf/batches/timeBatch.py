"""Sets of metrics to look at time between visits/pairs, etc.
"""
import numpy as np
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.plots as plots
import rubin_sim.maf.metricBundles as mb
from .colMapDict import ColMapDict
from .common import standardSummary, extendedSummary, filterList, combineMetadata, radecCols

__all__ = ['intraNight', 'interNight', 'timeGaps', 'seasons']


def intraNight(colmap=None, runName='opsim', nside=64, extraSql=None, extraMetadata=None,
               slicer=None, display_group='IntraNight', subgroup='Pairs'):
    """Generate a set of statistics about the pair/triplet/etc. rate within a night.

    Parameters
    ----------
    colmap : dict or None, optional
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : str, optional
        The name of the simulated survey. Default is "opsim".
    nside : int, optional
        Nside for the healpix slicer. Default 64.
    extraSql : str or None, optional
        Additional sql constraint to apply to all metrics.
    extraMetadata : str or None, optional
        Additional metadata to apply to all results.
    slicer : slicer object (None)
        Optionally use something other than a HealpixSlicer

    Returns
    -------
    metricBundleDict
    """

    if colmap is None:
        colmap = ColMapDict('opsimV4')

    metadata = extraMetadata
    if extraSql is not None and len(extraSql) > 0:
        if metadata is None:
            metadata = extraSql

    raCol, decCol, degrees, ditherStacker, ditherMeta = radecCols(None, colmap, None)
    metadata = combineMetadata(metadata, ditherMeta)

    bundleList = []
    standardStats = standardSummary()
    subsetPlots = [plots.HealpixSkyMap(), plots.HealpixHistogram()]

    if slicer is None:
        slicer = slicers.HealpixSlicer(nside=nside, latCol=decCol, lonCol=raCol, latLonDeg=degrees)

    # Look for the fraction of visits in gri where there are pairs within dtMin/dtMax.
    displayDict = {'group': display_group, 'subgroup': subgroup, 'caption': None, 'order': 0}
    if extraSql is not None and len(extraSql) > 0:
        sql = '(%s) and (filter="g" or filter="r" or filter="i")' % extraSql
    else:
        sql = 'filter="g" or filter="r" or filter="i"'
    md = 'gri'
    if metadata is not None:
        md += ' ' + metadata
    dtMin = 10.0
    dtMax = 60.0
    metric = metrics.PairFractionMetric(mjdCol=colmap['mjd'], minGap=dtMin, maxGap=dtMax,
                                        metricName='Fraction of visits in pairs (%.0f-%.0f min)' % (dtMin,
                                                                                                    dtMax))
    displayDict['caption'] = 'Fraction of %s visits that have a paired visit' \
                             'between %.1f and %.1f minutes away. ' % (md, dtMin, dtMax)
    displayDict['caption'] += 'If all visits were in pairs, this fraction would be 1.'
    displayDict['order'] += 1
    bundle = mb.MetricBundle(metric, slicer, sql, metadata=md, summaryMetrics=standardStats,
                             plotFuncs=subsetPlots, displayDict=displayDict)
    bundleList.append(bundle)

    dtMin = 20.0
    dtMax = 90.0
    metric = metrics.PairFractionMetric(mjdCol=colmap['mjd'], minGap=dtMin, maxGap=dtMax,
                                        metricName='Fraction of visits in pairs (%.0f-%.0f min)' % (dtMin,
                                                                                                    dtMax))
    displayDict['caption'] = 'Fraction of %s visits that have a paired visit' \
                             'between %.1f and %.1f minutes away. ' % (md, dtMin, dtMax)
    displayDict['caption'] += 'If all visits were in pairs, this fraction would be 1.'
    displayDict['order'] += 1
    bundle = mb.MetricBundle(metric, slicer, sql, metadata=md, summaryMetrics=standardStats,
                             plotFuncs=subsetPlots, displayDict=displayDict)
    bundleList.append(bundle)

    # Look at the fraction of visits which have another visit within dtMax, gri.
    dtMax = 60.0
    metric = metrics.NRevisitsMetric(mjdCol=colmap['mjd'], dT=dtMax, normed=True,
                                     metricName='Fraction of visits with a revisit < %.0f min' % dtMax)
    displayDict['caption'] = 'Fraction of %s visits that have another visit ' \
                             'within %.1f min. ' % (md, dtMax)
    displayDict['caption'] += 'If all visits were in pairs (only), this fraction would be 0.5.'
    displayDict['order'] += 1
    bundle = mb.MetricBundle(metric, slicer, sql, metadata=md, summaryMetrics=standardStats,
                             plotFuncs=subsetPlots, displayDict=displayDict)
    bundleList.append(bundle)

    # Intranight gap map, all filters. Returns value in hours.
    metric = metrics.IntraNightGapsMetric(metricName='Median Intra-Night Gap', mjdCol=colmap['mjd'],
                                          reduceFunc=np.median)
    displayDict['caption'] = 'Median gap between consecutive visits within a night, all bands'
    if metadata is None or len(metadata) == 0:
        displayDict['caption'] += ', all proposals.'
    else:
        displayDict['caption'] += ', %s.' % metadata
    displayDict['order'] += 1
    plotDict = {'percentileClip': 95}
    bundle = mb.MetricBundle(metric, slicer, extraSql, metadata=metadata, displayDict=displayDict,
                             plotFuncs=subsetPlots, plotDict=plotDict,
                             summaryMetrics=standardStats)
    bundleList.append(bundle)

    # Histogram the number of visits per night.
    countbins = np.arange(0, 10, 1)
    metric = metrics.NVisitsPerNightMetric(nightCol=colmap['night'], bins=countbins,
                                           metricName="NVisitsPerNight")
    plotDict = {'bins': countbins, 'xlabel': 'Number of visits each night'}
    displayDict['caption'] = 'Histogram of the number of visits in each night, per point on the sky'
    if metadata is None or len(metadata) == 0:
        displayDict['caption'] += ', all proposals.'
    else:
        displayDict['caption'] += ', %s.' % metadata
    displayDict['order'] = 0
    plotFunc = plots.SummaryHistogram()
    bundle = mb.MetricBundle(metric, slicer, extraSql, plotDict=plotDict,
                             displayDict=displayDict, metadata=metadata, plotFuncs=[plotFunc])
    bundleList.append(bundle)

    # Histogram of the time between revisits (all filters) within two hours.
    binMin = 0
    binMax = 120.
    binsize = 5.
    bins_metric = np.arange(binMin / 60.0 / 24.0, (binMax + binsize) / 60. / 24., binsize / 60. / 24.)
    bins_plot = bins_metric * 24.0 * 60.0
    metric = metrics.TgapsMetric(bins=bins_metric, timesCol=colmap['mjd'], metricName='DeltaT Histogram')
    plotDict = {'bins': bins_plot, 'xlabel': 'dT (minutes)'}
    displayDict['caption'] = 'Histogram of the time between consecutive visits to a given point ' \
                             'on the sky, considering visits between %.1f and %.1f minutes' % (binMin,
                                                                                               binMax)
    if metadata is None or len(metadata) == 0:
        displayDict['caption'] += ', all proposals.'
    else:
        displayDict['caption'] += ', %s.' % metadata
    displayDict['order'] += 1
    plotFunc = plots.SummaryHistogram()
    bundle = mb.MetricBundle(metric, slicer, extraSql, plotDict=plotDict,
                             displayDict=displayDict, metadata=metadata, plotFuncs=[plotFunc])
    bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def interNight(colmap=None, runName='opsim', nside=64, extraSql=None, extraMetadata=None,
               slicer=None, display_group='InterNight', subgroup='Night gaps'):
    """Generate a set of statistics about the spacing between nights with observations.

    Parameters
    ----------
    colmap : dict or None, optional
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : str, optional
        The name of the simulated survey. Default is "opsim".
    nside : int, optional
        Nside for the healpix slicer. Default 64.
    extraSql : str or None, optional
        Additional sql constraint to apply to all metrics.
    extraMetadata : str or None, optional
        Additional metadata to use for all outputs.
    slicer : slicer object (None)
        Optionally use something other than a HealpixSlicer

    Returns
    -------
    metricBundleDict
    """

    if colmap is None:
        colmap = ColMapDict('opsimV4')

    bundleList = []

    # Set up basic all and per filter sql constraints.
    raCol, decCol, degrees, ditherStacker, ditherMeta = radecCols(None, colmap, None)
    metadata = combineMetadata(extraMetadata, ditherMeta)
    filterlist, colors, orders, sqls, metadata = filterList(all=True,
                                                            extraSql=extraSql,
                                                            extraMetadata=metadata)

    if slicer is None:
        slicer = slicers.HealpixSlicer(nside=nside, latCol=decCol, lonCol=raCol, latLonDeg=degrees)

    displayDict = {'group': display_group, 'subgroup': subgroup, 'caption': None, 'order': 0}

    # Histogram of the number of nights between visits.
    bins = np.arange(1, 20.5, 1)
    metric = metrics.NightgapsMetric(bins=bins, nightCol=colmap['night'], metricName='DeltaNight Histogram')
    plotDict = {'bins': bins, 'xlabel': 'dT (nights)'}
    displayDict['caption'] = 'Histogram of the number of nights between consecutive visits to a ' \
                             'given point on the sky, considering separations between %d and %d' \
                             % (bins.min(), bins.max())
    if metadata['all'] is None or len(metadata['all']) == 0:
        displayDict['caption'] += ', all proposals.'
    else:
        displayDict['caption'] += ', %s.' % metadata['all']
    plotFunc = plots.SummaryHistogram()
    bundle = mb.MetricBundle(metric, slicer, sqls['all'], plotDict=plotDict,
                             displayDict=displayDict, metadata=metadata['all'], plotFuncs=[plotFunc])
    bundleList.append(bundle)

    standardStats = standardSummary()
    subsetPlots = [plots.HealpixSkyMap(), plots.HealpixHistogram()]

    # Median inter-night gap (each and all filters)
    metric = metrics.InterNightGapsMetric(metricName='Median Inter-Night Gap', mjdCol=colmap['mjd'],
                                          reduceFunc=np.median)
    for f in filterlist:
        displayDict['caption'] = 'Median gap between nights with observations, %s.' % metadata[f]
        displayDict['order'] = orders[f]
        plotDict = {'color': colors[f], 'percentileClip': 95.}
        bundle = mb.MetricBundle(metric, slicer, sqls[f], metadata=metadata[f],
                                 displayDict=displayDict,
                                 plotFuncs=subsetPlots, plotDict=plotDict,
                                 summaryMetrics=standardStats)
        bundleList.append(bundle)

    # Maximum inter-night gap (in each and all filters).
    metric = metrics.InterNightGapsMetric(metricName='Max Inter-Night Gap', mjdCol=colmap['mjd'],
                                          reduceFunc=np.max)
    for f in filterlist:
        displayDict['caption'] = 'Maximum gap between nights with observations, %s.' % metadata[f]
        displayDict['order'] = orders[f]
        plotDict = {'color': colors[f], 'percentileClip': 95., 'binsize': 5}
        bundle = mb.MetricBundle(metric, slicer, sqls[f], metadata=metadata[f], displayDict=displayDict,
                                 plotFuncs=subsetPlots, plotDict=plotDict, summaryMetrics=standardStats)
        bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def timeGaps(colmap=None, runName='opsim', nside=64,
          extraSql=None, extraMetadata=None, slicer=None,
          display_group='TimeGaps', subgroup='Time'):
    """Generate a set of statistics about the spacing between nights with observations.

     Parameters
     ----------
     colmap : dict or None, optional
         A dictionary with a mapping of column names. Default will use OpsimV4 column names.
     runName : str, optional
         The name of the simulated survey. Default is "opsim".
     nside : int, optional
         Nside for the healpix slicer. Default 64.
     extraSql : str or None, optional
         Additional sql constraint to apply to all metrics.
     extraMetadata : str or None, optional
         Additional metadata to use for all outputs.
     slicer : slicer object (None)
         Optionally use something other than a HealpixSlicer

     Returns
     -------
     metricBundleDict
     """

    if colmap is None:
        colmap = ColMapDict('opsimV4')

    bundleList = []

    raCol = colmap['ra']
    decCol = colmap['dec']
    degrees = colmap['raDecDeg']
    filterlist, colors, orders, sqls, metadata = filterList(all=True, extraSql=extraSql,
                                                            extraMetadata=extraMetadata)

    if slicer is None:
        slicer = slicers.HealpixSlicer(nside=nside, latCol=decCol, lonCol=raCol, latLonDeg=degrees)

    displayDict = {'group': display_group, 'subgroup': subgroup, 'caption': None, 'order': 0}

    # Logarithmically spaced gaps from 30s to 5 years
    tMin = 30 / 60 / 60 / 24.  # 30s
    tMax = 5 * 365.25  # 5 years
    tgaps = np.logspace(np.log10(tMin), np.log10(tMax), 100)

    for f in filterlist:
        m1 = metrics.TgapsMetric(bins=tgaps, allGaps=False)
        plotDict = {'bins': tgaps, 'xscale': 'log', 'yMin': 0, 'figsize': (8, 6),
                    'ylabel': 'Number of observation pairs',
                    'xlabel': 'Time gap between pairs of visits (days)',
                    'color': colors[f]}
        plotFuncs = [plots.SummaryHistogram()]
        displayDict['caption'] = f'Summed Histogram of time between visits at each point in the sky, ' \
                                 f'in {f} band(s).'
        displayDict['order'] = orders[f]
        bundleList.append(mb.MetricBundle(m1, slicer, constraint=sqls[f], metadata=metadata[f],
                                          runName=runName, plotDict=plotDict, plotFuncs=plotFuncs,
                                          displayDict=displayDict))

        m2 = metrics.TgapsPercentMetric(minTime=2 / 24., maxTime=14 / 24., allGaps=False,
                                        metricName='TgapsPercent_2-14hrs')
        plotFuncs = [plots.HealpixSkyMap(), plots.HealpixHistogram()]
        plotDict = {'colorMin': 0, 'color': colors[f]}
        summaryMetrics = extendedSummary()
        displayDict['caption'] = f'Percent of the total time gaps which fall into the interval' \
                                 f' between 2-14 hours, in {f} band(s).'
        displayDict['order'] = orders[f]
        bundleList.append(mb.MetricBundle(m2, slicer, constraint=sqls[f], metadata=metadata[f],
                                          runName=runName, summaryMetrics=summaryMetrics,
                                          plotDict=plotDict, plotFuncs=plotFuncs, displayDict=displayDict))

        m3 = metrics.TgapsPercentMetric(minTime=14. / 24., maxTime=(14. / 24 + 1.), allGaps=False,
                                        metricName='TgapsPercent_1day')
        displayDict['caption'] = f'Percent of the total time gaps which fall into the interval around 1 day,' \
                                 f' in {f} band(s).'
        displayDict['order'] = orders[f]
        bundleList.append(mb.MetricBundle(m3, slicer, constraint=sqls[f], metadata=metadata[f],
                                          runName=runName, summaryMetrics=summaryMetrics,
                                          plotDict=plotDict, plotFuncs=plotFuncs, displayDict=displayDict))
    return mb.makeBundlesDictFromList(bundleList)


def seasons(colmap=None, runName='opsim', nside=64, extraSql=None, extraMetadata=None):
    """Generate a set of statistics about the length and number of seasons.

    Parameters
    ----------
    colmap : dict or None, optional
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : str, optional
        The name of the simulated survey. Default is "opsim".
    nside : int, optional
        Nside for the healpix slicer. Default 64.
    extraSql : str or None, optional
        Additional sql constraint to apply to all metrics.
    extraMetadata : str or None, optional
        Additional metadata to use for all outputs.

    Returns
    -------
    metricBundleDict
    """

    if colmap is None:
        colmap = ColMapDict('opsimV4')

    bundleList = []

    # Set up basic all and per filter sql constraints.
    raCol, decCol, degrees, ditherStacker, ditherMeta = radecCols(None, colmap, None)
    metadata = combineMetadata(extraMetadata, ditherMeta)
    filterlist, colors, orders, sqls, metadata = filterList(all=True,
                                                            extraSql=extraSql,
                                                            extraMetadata=metadata)

    slicer = slicers.HealpixSlicer(nside=nside, latCol=decCol, lonCol=raCol, latLonDeg=degrees)

    displayDict = {'group': 'IntraSeason', 'subgroup': 'Season length', 'caption': None, 'order': 0}

    standardStats = standardSummary()
    subsetPlots = [plots.HealpixSkyMap(), plots.HealpixHistogram()]

    metric = metrics.SeasonLengthMetric(metricName='Median Season Length', mjdCol=colmap['mjd'],
                                        reduceFunc=np.median)
    for f in filterlist:
        displayDict['caption'] = 'Median season length, %s.' % metadata[f]
        displayDict['order'] = orders[f]
        maxS = 250
        if f == 'all':
            minS = 90
        else:
            minS = 30
        plotDict = {'color': colors[f], 'colorMin': minS, 'colorMax': maxS, 'xMin': minS, 'xMax': maxS}
        bundle = mb.MetricBundle(metric, slicer, sqls[f], metadata=metadata[f],
                                 displayDict=displayDict,
                                 plotFuncs=subsetPlots, plotDict=plotDict,
                                 summaryMetrics=standardStats)
        bundleList.append(bundle)

    # Number of seasons
    metric = metrics.CampaignLengthMetric(metricName='NSeasons', mjdCol=colmap['mjd'],
                                          expTimeCol=colmap['exptime'], minExpTime=15)
    displayDict['caption'] = 'Number of seasons, any filter.'
    displayDict['order'] = 0
    plotDict = {'color': 'k', 'colorMin': 0, 'colorMax': 11, 'xMin': 0, 'xMax': 11}
    bundle = mb.MetricBundle(metric, slicer, sqls['all'], metadata=metadata['all'],
                             displayDict=displayDict,
                             plotFuncs=subsetPlots, plotDict=plotDict,
                             summaryMetrics=standardStats)
    bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)
