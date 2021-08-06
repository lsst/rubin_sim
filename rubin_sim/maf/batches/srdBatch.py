"""Sets of metrics to look at the SRD metrics.
"""
import numpy as np
import healpy as hp
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.stackers as stackers
import rubin_sim.maf.plots as plots
import rubin_sim.maf.metricBundles as mb
from .colMapDict import ColMapDict
from .common import standardSummary, radecCols, combineMetadata

__all__ = ['fOBatch', 'astrometryBatch', 'rapidRevisitBatch']


def fOBatch(colmap=None, runName='opsim', extraSql=None, extraMetadata=None, nside=64,
            benchmarkArea=18000, benchmarkNvisits=825, minNvisits=750):
    """Metrics for calculating fO.

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

    Returns
    -------
    metricBundleDict
    """
    if colmap is None:
        colmap = ColMapDict('fbs')

    bundleList = []

    sql = ''
    metadata = 'All visits'
    # Add additional sql constraint (such as wfdWhere) and metadata, if provided.
    if (extraSql is not None) and (len(extraSql) > 0):
        sql = extraSql
        if extraMetadata is None:
            metadata = extraSql.replace('filter =', '').replace('filter=', '')
            metadata = metadata.replace('"', '').replace("'", '')
    if extraMetadata is not None:
        metadata = extraMetadata

    subgroup = metadata

    raCol, decCol, degrees, ditherStacker, ditherMeta = radecCols(None, colmap, None)
    # Don't want dither info in subgroup (too long), but do want it in bundle name.
    metadata = combineMetadata(metadata, ditherMeta)

    # Set up fO metric.
    slicer = slicers.HealpixSlicer(nside=nside, lonCol=raCol, latCol=decCol, latLonDeg=degrees)

    displayDict = {'group': 'SRD FO metrics', 'subgroup': subgroup, 'order': 0}

    # Configure the count metric which is what is used for f0 slicer.
    metric = metrics.CountExplimMetric(col=colmap['mjd'], metricName='fO', expCol=colmap['exptime'])
    plotDict = {'xlabel': 'Number of Visits', 'Asky': benchmarkArea,
                'Nvisit': benchmarkNvisits, 'xMin': 0, 'xMax': 1500}
    summaryMetrics = [metrics.fOArea(nside=nside, norm=False, metricName='fOArea',
                                     Asky=benchmarkArea, Nvisit=benchmarkNvisits),
                      metrics.fOArea(nside=nside, norm=True, metricName='fOArea/benchmark',
                                     Asky=benchmarkArea, Nvisit=benchmarkNvisits),
                      metrics.fONv(nside=nside, norm=False, metricName='fONv',
                                   Asky=benchmarkArea, Nvisit=benchmarkNvisits),
                      metrics.fONv(nside=nside, norm=True, metricName='fONv/benchmark',
                                   Asky=benchmarkArea, Nvisit=benchmarkNvisits),
                      metrics.fOArea(nside=nside, norm=False, metricName=f'fOArea_{minNvisits}',
                                     Asky=benchmarkArea, Nvisit=minNvisits)]
    caption = 'The FO metric evaluates the overall efficiency of observing. '
    caption += ('foNv: out of %.2f sq degrees, the area receives at least X and a median of Y visits '
                '(out of %d, if compared to benchmark). ' % (benchmarkArea, benchmarkNvisits))
    caption += ('fOArea: this many sq deg (out of %.2f sq deg if compared '
                'to benchmark) receives at least %d visits. ' % (benchmarkArea, benchmarkNvisits))
    displayDict['caption'] = caption
    bundle = mb.MetricBundle(metric, slicer, sql, plotDict=plotDict,
                             stackerList = [ditherStacker],
                             displayDict=displayDict, summaryMetrics=summaryMetrics,
                             plotFuncs=[plots.FOPlot()], metadata=metadata)
    bundleList.append(bundle)
    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def astrometryBatch(colmap=None, runName='opsim',
                    extraSql=None, extraMetadata=None,
                    nside=64, ditherStacker=None, ditherkwargs=None):
    """Metrics for evaluating proper motion and parallax.

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
    ditherStacker: str or rubin_sim.maf.stackers.BaseDitherStacker
        Optional dither stacker to use to define ra/dec columns.
    ditherkwargs: dict, optional
        Optional dictionary of kwargs for the dither stacker.

    Returns
    -------
    metricBundleDict
    """
    if colmap is None:
        colmap = ColMapDict('fbs')
    bundleList = []

    sql = ''
    metadata = 'All visits'
    # Add additional sql constraint (such as wfdWhere) and metadata, if provided.
    if (extraSql is not None) and (len(extraSql) > 0):
        sql = extraSql
        if extraMetadata is None:
            metadata = extraSql.replace('filter =', '').replace('filter=', '')
            metadata = metadata.replace('"', '').replace("'", '')
    if extraMetadata is not None:
        metadata = extraMetadata

    subgroup = metadata

    raCol, decCol, degrees, ditherStacker, ditherMeta = radecCols(ditherStacker, colmap, ditherkwargs)
    # Don't want dither info in subgroup (too long), but do want it in bundle name.
    metadata = combineMetadata(metadata, ditherMeta)

    rmags_para = [22.4, 24.0]
    rmags_pm = [20.5, 24.0]

    # Set up parallax/dcr stackers.
    parallaxStacker = stackers.ParallaxFactorStacker(raCol=raCol, decCol=decCol,
                                                     dateCol=colmap['mjd'], degrees=degrees)
    dcrStacker = stackers.DcrStacker(filterCol=colmap['filter'], altCol=colmap['alt'], degrees=degrees,
                                     raCol=raCol, decCol=decCol, lstCol=colmap['lst'],
                                     site='LSST', mjdCol=colmap['mjd'])

    # Set up parallax metrics.
    slicer = slicers.HealpixSlicer(nside=nside, lonCol=raCol, latCol=decCol, latLonDeg=degrees)
    subsetPlots = [plots.HealpixSkyMap(), plots.HealpixHistogram()]

    displayDict = {'group': 'SRD Parallax', 'subgroup': subgroup,
                   'order': 0, 'caption': None}
    # Expected error on parallax at 10 AU.
    plotmaxVals = (2.0, 15.0)
    summary = [metrics.AreaSummaryMetric(area=18000, reduce_func=np.median, decreasing=False,
                                         metricName='Median Parallax Error (18k)')]
    summary.append(metrics.PercentileMetric(percentile=95, metricName='95th Percentile Parallax Error'))
    summary.extend(standardSummary())
    for rmag, plotmax in zip(rmags_para, plotmaxVals):
        plotDict = {'xMin': 0, 'xMax': plotmax, 'colorMin': 0, 'colorMax': plotmax}
        metric = metrics.ParallaxMetric(metricName='Parallax Error @ %.1f' % (rmag), rmag=rmag,
                                        seeingCol=colmap['seeingGeom'], filterCol=colmap['filter'],
                                        m5Col=colmap['fiveSigmaDepth'], normalize=False)
        bundle = mb.MetricBundle(metric, slicer, sql, metadata=metadata,
                                 stackerList=[parallaxStacker, ditherStacker],
                                 displayDict=displayDict, plotDict=plotDict,
                                 summaryMetrics=summary,
                                 plotFuncs=subsetPlots)
        bundleList.append(bundle)
        displayDict['order'] += 1

    # Parallax normalized to 'best possible' if all visits separated by 6 months.
    # This separates the effect of cadence from depth.
    for rmag in rmags_para:
        metric = metrics.ParallaxMetric(metricName='Normalized Parallax @ %.1f' % (rmag), rmag=rmag,
                                        seeingCol=colmap['seeingGeom'], filterCol=colmap['filter'],
                                        m5Col=colmap['fiveSigmaDepth'], normalize=True)
        bundle = mb.MetricBundle(metric, slicer, sql, metadata=metadata,
                                 stackerList=[parallaxStacker, ditherStacker],
                                 displayDict=displayDict,
                                 summaryMetrics=standardSummary(),
                                 plotFuncs=subsetPlots)
        bundleList.append(bundle)
        displayDict['order'] += 1
    # Parallax factor coverage.
    for rmag in rmags_para:
        metric = metrics.ParallaxCoverageMetric(metricName='Parallax Coverage @ %.1f' % (rmag),
                                                rmag=rmag, m5Col=colmap['fiveSigmaDepth'],
                                                mjdCol=colmap['mjd'], filterCol=colmap['filter'],
                                                seeingCol=colmap['seeingGeom'])
        bundle = mb.MetricBundle(metric, slicer, sql, metadata=metadata,
                                 stackerList=[parallaxStacker, ditherStacker],
                                 displayDict=displayDict, summaryMetrics=standardSummary(),
                                 plotFuncs=subsetPlots)
        bundleList.append(bundle)
        displayDict['order'] += 1
    # Parallax problems can be caused by HA and DCR degeneracies. Check their correlation.
    for rmag in rmags_para:
        metric = metrics.ParallaxDcrDegenMetric(metricName='Parallax-DCR degeneracy @ %.1f' % (rmag),
                                                rmag=rmag, seeingCol=colmap['seeingEff'],
                                                filterCol=colmap['filter'], m5Col=colmap['fiveSigmaDepth'])
        caption = 'Correlation between parallax offset magnitude and hour angle for a r=%.1f star.' % (rmag)
        caption += ' (0 is good, near -1 or 1 is bad).'
        bundle = mb.MetricBundle(metric, slicer, sql, metadata=metadata,
                                 stackerList=[dcrStacker, parallaxStacker, ditherStacker],
                                 displayDict=displayDict, summaryMetrics=standardSummary(),
                                 plotFuncs=subsetPlots)
        bundleList.append(bundle)
        displayDict['order'] += 1

    # Proper Motion metrics.
    displayDict = {'group': 'SRD Proper Motion', 'subgroup': subgroup, 'order': 0, 'caption': None}
    # Proper motion errors.
    plotmaxVals = (1.0, 5.0)
    summary = [metrics.AreaSummaryMetric(area=18000, reduce_func=np.median, decreasing=False,
                                         metricName='Median Proper Motion Error (18k)')]
    summary.append(metrics.PercentileMetric(metricName='95th Percentile Proper Motion Error'))
    summary.extend(standardSummary())
    for rmag, plotmax in zip(rmags_pm, plotmaxVals):
        plotDict = {'xMin': 0, 'xMax': plotmax, 'colorMin': 0, 'colorMax': plotmax}
        metric = metrics.ProperMotionMetric(metricName='Proper Motion Error @ %.1f' % rmag,
                                            rmag=rmag, m5Col=colmap['fiveSigmaDepth'],
                                            mjdCol=colmap['mjd'], filterCol=colmap['filter'],
                                            seeingCol=colmap['seeingGeom'], normalize=False)
        bundle = mb.MetricBundle(metric, slicer, sql, metadata=metadata,
                                 stackerList=[ditherStacker],
                                 displayDict=displayDict, plotDict=plotDict,
                                 summaryMetrics=summary,
                                 plotFuncs=subsetPlots)
        bundleList.append(bundle)
        displayDict['order'] += 1
    # Normalized proper motion.
    for rmag in rmags_pm:
        metric = metrics.ProperMotionMetric(metricName='Normalized Proper Motion @ %.1f' % rmag,
                                            rmag=rmag, m5Col=colmap['fiveSigmaDepth'],
                                            mjdCol=colmap['mjd'], filterCol=colmap['filter'],
                                            seeingCol=colmap['seeingGeom'], normalize=True)
        bundle = mb.MetricBundle(metric, slicer, sql, metadata=metadata,
                                 stackerList=[ditherStacker],
                                 displayDict=displayDict, summaryMetrics=standardSummary(),
                                 plotFuncs=subsetPlots)
        bundleList.append(bundle)
        displayDict['order'] += 1

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def rapidRevisitBatch(colmap=None, runName='opsim',
                      extraSql=None, extraMetadata=None, nside=64,
                      ditherStacker=None, ditherkwargs=None):
    """Metrics for evaluating proper motion and parallax.

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
    ditherStacker: str or rubin_sim.maf.stackers.BaseDitherStacker
        Optional dither stacker to use to define ra/dec columns.
    ditherkwargs: dict, optional
        Optional dictionary of kwargs for the dither stacker.

    Returns
    -------
    metricBundleDict
    """
    if colmap is None:
        colmap = ColMapDict('fbs')
    bundleList = []

    sql = ''
    metadata = 'All visits'
    # Add additional sql constraint (such as wfdWhere) and metadata, if provided.
    if (extraSql is not None) and (len(extraSql) > 0):
        sql = extraSql
        if extraMetadata is None:
            metadata = extraSql.replace('filter =', '').replace('filter=', '')
            metadata = metadata.replace('"', '').replace("'", '')
    if extraMetadata is not None:
        metadata = extraMetadata

    subgroup = metadata

    raCol, decCol, degrees, ditherStacker, ditherMeta = radecCols(ditherStacker, colmap, ditherkwargs)
    # Don't want dither info in subgroup (too long), but do want it in bundle name.
    metadata = combineMetadata(metadata, ditherMeta)

    slicer = slicers.HealpixSlicer(nside=nside, lonCol=raCol, latCol=decCol, latLonDeg=degrees)
    subsetPlots = [plots.HealpixSkyMap(), plots.HealpixHistogram()]

    displayDict = {'group': 'SRD Rapid Revisits', 'subgroup': subgroup,
                   'order': 0, 'caption': None}

    # Calculate the actual number of revisits within 30 minutes.
    dTmax = 30   # time in minutes
    m2 = metrics.NRevisitsMetric(dT=dTmax, mjdCol=colmap['mjd'], normed=False,
                                 metricName='NumberOfQuickRevisits')
    plotDict = {'colorMin': 400, 'colorMax': 2000, 'xMin': 400, 'xMax': 2000}
    caption = 'Number of consecutive visits with return times faster than %.1f minutes, ' % (dTmax)
    caption += 'in any filter, all proposals. '
    displayDict['caption'] = caption
    bundle = mb.MetricBundle(m2, slicer, sql, plotDict=plotDict, plotFuncs=subsetPlots,
                             stackerList=[ditherStacker],
                             metadata=metadata, displayDict=displayDict,
                             summaryMetrics=standardSummary(withCount=False))
    bundleList.append(bundle)
    displayDict['order'] += 1

    # Better version of the rapid revisit requirements: require a minimum number of visits between
    # dtMin and dtMax, but also a minimum number of visits between dtMin and dtPair (the typical pair time).
    # 1 means the healpix met the requirements (0 means did not).
    dTmin = 40.0/60.0  # (minutes) 40s minumum for rapid revisit range
    dTpairs = 20.0  # minutes (time when pairs should start kicking in)
    dTmax = 30.0  # 30 minute maximum for rapid revisit range
    nOne = 82  # Number of revisits between 40s-30m required
    nTwo = 28  # Number of revisits between 40s - tPairs required.
    pixArea = float(hp.nside2pixarea(nside, degrees=True))
    scale = pixArea * hp.nside2npix(nside)
    m1 = metrics.RapidRevisitMetric(metricName='RapidRevisits', mjdCol=colmap['mjd'],
                                    dTmin=dTmin / 60.0 / 60.0 / 24.0, dTpairs = dTpairs / 60.0 / 24.0,
                                    dTmax=dTmax / 60.0 / 24.0, minN1=nOne, minN2=nTwo)
    plotDict = {'xMin': 0, 'xMax': 1, 'colorMin': 0, 'colorMax': 1, 'logScale': False}
    cutoff1 = 0.9
    summaryStats = [metrics.FracAboveMetric(cutoff=cutoff1, scale=scale, metricName='Area (sq deg)')]
    caption = 'Rapid Revisit: area that receives at least %d visits between %.3f and %.1f minutes, ' \
              % (nOne, dTmin, dTmax)
    caption += 'with at least %d of those visits falling between %.3f and %.1f minutes. ' \
               % (nTwo, dTmin, dTpairs)
    caption += 'Summary statistic "Area" indicates the area on the sky which meets this requirement.' \
               ' (SRD design specification is 2000 sq deg).'
    displayDict['caption'] = caption
    bundle = mb.MetricBundle(m1, slicer, sql, plotDict=plotDict, plotFuncs=subsetPlots,
                             stackerList=[ditherStacker],
                             metadata=metadata, displayDict=displayDict, summaryMetrics=summaryStats)
    bundleList.append(bundle)
    displayDict['order'] += 1

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)

