from __future__ import print_function
import warnings
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.stackers as stackers
import rubin_sim.maf.plots as plots
import rubin_sim.maf.metricBundles as metricBundles
from .colMapDict import ColMapDict
from .common import standardSummary
from .slewBatch import slewBasics
from .hourglassBatch import hourglassPlots

__all__ = ['glanceBatch']


def glanceBatch(colmap=None, runName='opsim',
                nside=64, filternames=('u', 'g', 'r', 'i', 'z', 'y'),
                nyears=10, pairnside=32, sqlConstraint=None, slicer_camera='LSST'):
    """Generate a handy set of metrics that give a quick overview of how well a survey performed.
    This is a meta-set of other batches, to some extent.

    Parameters
    ----------
    colmap : dict, optional
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    run_name : str, optional
        The name of the simulated survey. Default is "opsim".
    nside : int, optional
        The nside for the healpix slicers. Default 64.
    filternames : list of str, optional
        The list of individual filters to use when running metrics.
        Default is ('u', 'g', 'r', 'i', 'z', 'y').
        There is always an all-visits version of the metrics run as well.
    nyears : int (10)
        How many years to attempt to make hourglass plots for
    pairnside : int (32)
        nside to use for the pair fraction metric (it's slow, so nice to use lower resolution)
    sqlConstraint : str or None, optional
        Additional SQL constraint to apply to all metrics.
    slicer_camera : str ('LSST')
        Sets which spatial slicer to use. options are 'LSST' and 'ComCam'

    Returns
    -------
    metricBundleDict
    """
    if isinstance(colmap, str):
        raise ValueError('colmap must be a dictionary, not a string')

    if colmap is None:
        colmap = ColMapDict('opsimV4')

    bundleList = []

    if sqlConstraint is None:
        sqlC = ''
    else:
        sqlC = '(%s) and' % sqlConstraint

    if slicer_camera == 'LSST':
        spatial_slicer = slicers.HealpixSlicer
    elif slicer_camera == 'ComCam':
        spatial_slicer = slicers.HealpixComCamSlicer
    else:
        raise ValueError('Camera must be LSST or Comcam')

    sql_per_filt = ['%s %s="%s"' % (sqlC, colmap['filter'], filtername) for filtername in filternames]
    sql_per_and_all_filters = [sqlConstraint] + sql_per_filt

    standardStats = standardSummary()
    subsetPlots = [plots.HealpixSkyMap(), plots.HealpixHistogram()]

    # Super basic things
    displayDict = {'group': 'Basic Stats', 'order': 1}
    sql = sqlConstraint
    slicer = slicers.UniSlicer()
    # Length of Survey
    metric = metrics.FullRangeMetric(col=colmap['mjd'], metricName='Length of Survey (days)')
    bundle = metricBundles.MetricBundle(metric, slicer, sql, displayDict=displayDict)
    bundleList.append(bundle)

    # Total number of filter changes
    metric = metrics.NChangesMetric(col=colmap['filter'], orderBy=colmap['mjd'])
    bundle = metricBundles.MetricBundle(metric, slicer, sql, displayDict=displayDict)
    bundleList.append(bundle)

    # Total open shutter fraction
    metric = metrics.OpenShutterFractionMetric(slewTimeCol=colmap['slewtime'],
                                               expTimeCol=colmap['exptime'],
                                               visitTimeCol=colmap['visittime'])
    bundle = metricBundles.MetricBundle(metric, slicer, sql, displayDict=displayDict)
    bundleList.append(bundle)

    # Total effective exposure time
    metric = metrics.TeffMetric(m5Col=colmap['fiveSigmaDepth'],
                                filterCol=colmap['filter'], normed=True)
    for sql in sql_per_and_all_filters:
        bundle = metricBundles.MetricBundle(metric, slicer, sql, displayDict=displayDict)
        bundleList.append(bundle)

    # Number of observations, all and each filter
    metric = metrics.CountMetric(col=colmap['mjd'], metricName='Number of Exposures')
    for sql in sql_per_and_all_filters:
        bundle = metricBundles.MetricBundle(metric, slicer, sql, displayDict=displayDict)
        bundleList.append(bundle)

    # The alt/az plots of all the pointings
    slicer = spatial_slicer(nside=nside, latCol=colmap['alt'],
                            lonCol=colmap['az'], latLonDeg=colmap['raDecDeg'], useCache=False)
    metric = metrics.CountMetric(colmap['mjd'], metricName='Nvisits as function of Alt/Az')
    plotFuncs = [plots.LambertSkyMap()]

    plotDict = {'norm': 'log'}
    for sql in sql_per_and_all_filters:
        bundle = metricBundles.MetricBundle(metric, slicer, sql, plotFuncs=plotFuncs,
                                            displayDict=displayDict, plotDict=plotDict)
        bundleList.append(bundle)

    # Things to check per night
    # Open Shutter per night
    displayDict = {'group': 'Pointing Efficency', 'order': 2}
    slicer = slicers.OneDSlicer(sliceColName=colmap['night'], binsize=1)
    metric = metrics.OpenShutterFractionMetric(slewTimeCol=colmap['slewtime'],
                                               expTimeCol=colmap['exptime'],
                                               visitTimeCol=colmap['visittime'])
    sql = sqlConstraint
    bundle = metricBundles.MetricBundle(metric, slicer, sql,
                                        summaryMetrics=standardStats, displayDict=displayDict)
    bundleList.append(bundle)

    # Number of filter changes per night
    slicer = slicers.OneDSlicer(sliceColName=colmap['night'], binsize=1)
    metric = metrics.NChangesMetric(col=colmap['filter'], orderBy=colmap['mjd'],
                                    metricName='Filter Changes')
    bundle = metricBundles.MetricBundle(metric, slicer, sql,
                                        summaryMetrics=standardStats, displayDict=displayDict)
    bundleList.append(bundle)

    # A few basic maps
    # Number of observations, coadded depths
    extended_stats = standardStats.copy()
    extended_stats.append(metrics.AreaSummaryMetric(decreasing=True, metricName='top18k'))
    extended_stats.append(metrics.PercentileMetric(col='metricdata', percentile=10))
    displayDict = {'group': 'Basic Maps', 'order': 3}
    slicer = spatial_slicer(nside=nside, latCol=colmap['dec'], lonCol=colmap['ra'],
                            latLonDeg=colmap['raDecDeg'])
    metric = metrics.CountMetric(col=colmap['mjd'])
    plotDict = {'percentileClip': 95.}
    for sql in sql_per_and_all_filters:
        bundle = metricBundles.MetricBundle(metric, slicer, sql,
                                            summaryMetrics=extended_stats,
                                            displayDict=displayDict,
                                            plotDict=plotDict)
        bundleList.append(bundle)

    metric = metrics.Coaddm5Metric(m5Col=colmap['fiveSigmaDepth'])
    for sql in sql_per_and_all_filters:
        bundle = metricBundles.MetricBundle(metric, slicer, sql,
                                            summaryMetrics=extended_stats, displayDict=displayDict)
        bundleList.append(bundle)

    # Let's look at two years
    displayDict = {'group': 'Roll Check', 'order': 1}
    rolling_metrics = []
    rolling_metrics.append(metrics.CountMetric(col=colmap['mjd'], metricName='Year2.5Count'))
    rolling_metrics.append(metrics.CountMetric(col=colmap['mjd'], metricName='Year3.5Count'))
    rolling_sqls = []
    rolling_sqls.append('night > %f and night < %f' % (365.25*2.5, 365.25*3.5))
    rolling_sqls.append('night > %f and night < %f' % (365.25*3.5, 365.25*4.5))
    for metric, sql in zip(rolling_metrics, rolling_sqls):
        bundle = metricBundles.MetricBundle(metric, slicer, sql,
                                            summaryMetrics=extended_stats,
                                            plotDict=plotDict, displayDict=displayDict)
        bundleList.append(bundle)

    # Checking a few basic science things
    # Maybe check astrometry, observation pairs, SN
    plotDict = {'percentileClip': 95.}
    displayDict = {'group': 'Science', 'subgroup': 'Astrometry', 'order': 4}

    stackerList = []
    stacker = stackers.ParallaxFactorStacker(raCol=colmap['ra'],
                                             decCol=colmap['dec'],
                                             degrees=colmap['raDecDeg'],
                                             dateCol=colmap['mjd'])
    stackerList.append(stacker)

    astrom_stats = [metrics.AreaSummaryMetric(decreasing=False, metricName='best18k'),
                    metrics.PercentileMetric(col='metricdata', percentile=90)]
    # Maybe parallax and proper motion, fraction of visits in a good pair for SS
    displayDict['caption'] = r'Parallax precision of an $r=20$ flat SED star'
    metric = metrics.ParallaxMetric(m5Col=colmap['fiveSigmaDepth'],
                                    filterCol=colmap['filter'],
                                    seeingCol=colmap['seeingGeom'])
    sql = sqlConstraint
    bundle = metricBundles.MetricBundle(metric, slicer, sql, plotFuncs=subsetPlots,
                                        displayDict=displayDict, stackerList=stackerList,
                                        plotDict=plotDict,
                                        summaryMetrics=astrom_stats)
    bundleList.append(bundle)
    displayDict['caption'] = r'Proper motion precision of an $r=20$ flat SED star'
    metric = metrics.ProperMotionMetric(m5Col=colmap['fiveSigmaDepth'],
                                        mjdCol=colmap['mjd'],
                                        filterCol=colmap['filter'],
                                        seeingCol=colmap['seeingGeom'])
    bundle = metricBundles.MetricBundle(metric, slicer, sql, plotFuncs=subsetPlots,
                                        displayDict=displayDict, plotDict=plotDict,
                                        summaryMetrics=astrom_stats)
    bundleList.append(bundle)

    # Solar system stuff
    displayDict['caption'] = 'Fraction of observations that are in pairs'
    displayDict['subgroup'] = 'Solar System'

    sql = '%s (filter="g" or filter="r" or filter="i")' % sqlC
    pairSlicer = slicers.HealpixSlicer(nside=pairnside, latCol=colmap['dec'], lonCol=colmap['ra'],
                                       latLonDeg=colmap['raDecDeg'])
    metric = metrics.PairFractionMetric(mjdCol=colmap['mjd'])
    bundle = metricBundles.MetricBundle(metric, pairSlicer, sql, plotFuncs=subsetPlots,
                                        displayDict=displayDict)
    bundleList.append(bundle)

    # stats from the note column
    if 'note' in colmap.keys():
        displayDict = {'group': 'Basic Stats', 'subgroup': 'Percent stats'}
        metric = metrics.StringCountMetric(col=colmap['note'], percent=True, metricName='Percents')
        sql = ''
        slicer = slicers.UniSlicer()
        bundle = metricBundles.MetricBundle(metric, slicer, sql, displayDict=displayDict)
        bundleList.append(bundle)
        displayDict['subgroup'] = 'Count Stats'
        metric = metrics.StringCountMetric(col=colmap['note'], metricName='Counts')
        bundle = metricBundles.MetricBundle(metric, slicer, sql, displayDict=displayDict)
        bundleList.append(bundle)

    for b in bundleList:
        b.setRunName(runName)

    bd = metricBundles.makeBundlesDictFromList(bundleList)

    # Add hourglass plots.
    hrDict = hourglassPlots(colmap=colmap, runName=runName, nyears=nyears, extraSql=sqlConstraint)
    bd.update(hrDict)
    # Add basic slew stats.
    try:
        slewDict = slewBasics(colmap=colmap, runName=runName)
        bd.update(slewDict)
    except KeyError as e:
        warnings.warn('Could not add slew stats: missing required key %s from colmap' % (e))

    return bd

