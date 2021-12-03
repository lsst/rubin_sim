"""Evaluate some bulk properties of the sky coverage
"""
import numpy as np
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.metricBundles as mb
from .colMapDict import ColMapDict

__all__ = ['meanRADec', 'eastWestBias']


def meanRADec(colmap=None, runName='opsim', extraSql=None, extraMetadata=None):
    """Plot the range of RA/Dec as a function of night.

    Parameters
    ----------
    colmap : dict, optional
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : str, optional
        The name of the simulated survey. Default is "opsim".
    extraSql : str, optional
        Additional constraint to add to any sql constraints (e.g. 'night<365')
        Default None, for no additional constraints.
    extraMetadata : str, optional
        Additional metadata to add before any below (i.e. "WFD").  Default is None.
    """
    if colmap is None:
        colmap = ColMapDict('opsimV4')
    bundleList = []

    group = 'RA Dec coverage'

    subgroup = 'All visits'
    if extraMetadata is not None:
        subgroup = extraMetadata

    displayDict = {'group': group, 'subgroup': subgroup, 'order': 0}

    ra_metrics = [metrics.MeanAngleMetric(colmap['ra']), metrics.FullRangeAngleMetric(colmap['ra'])]
    dec_metrics = [metrics.MeanMetric(colmap['dec']), metrics.MinMetric(colmap['dec']),
                   metrics.MaxMetric(colmap['dec'])]
    for m in ra_metrics:
        slicer = slicers.OneDSlicer(sliceColName=colmap['night'], binsize=1)
        if not colmap['raDecDeg']:
            plotDict = {'yMin': np.radians(-5), 'yMax': np.radians(365)}
        else:
            plotDict = {'yMin': -5, 'yMax': 365}
        bundle = mb.MetricBundle(m, slicer, extraSql, metadata=extraMetadata,
                                 displayDict=displayDict, plotDict=plotDict)
        bundleList.append(bundle)

    for m in dec_metrics:
        slicer = slicers.OneDSlicer(sliceColName=colmap['night'], binsize=1)
        bundle = mb.MetricBundle(m, slicer, extraSql, metadata=extraMetadata,
                                 displayDict=displayDict)
        bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def eastWestBias(colmap=None, runName='opsim', extraSql=None, extraMetadata=None):
    """Plot the number of observations to the east vs to the west, per night.

    Parameters
    ----------
    colmap : dict, optional
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : str, optional
        The name of the simulated survey. Default is "opsim".
    extraSql : str, optional
        Additional constraint to add to any sql constraints (e.g. 'night<365')
        Default None, for no additional constraints.
    extraMetadata : str, optional
        Additional metadata to add before any below (i.e. "WFD").  Default is None.
    """
    if colmap is None:
        colmap = ColMapDict('opsimV4')
    bundleList = []

    group = 'East vs West'

    subgroup = 'All visits'
    if extraMetadata is not None:
        subgroup = extraMetadata

    displayDict = {'group': group, 'subgroup': subgroup, 'order': 0}

    eastvswest = 180
    if not colmap['raDecDeg']:
        eastvswest = np.radians(eastvswest)

    displayDict['caption'] = 'Number of visits per night that occur with azimuth <= 180.'
    if extraSql is not None:
        displayDict['caption'] += ' With additional sql constraint %s.' % extraSql
    metric = metrics.CountMetric(colmap['night'], metricName='Nvisits East')
    slicer = slicers.OneDSlicer(sliceColName=colmap['night'], binsize=1)
    sql = '%s <= %f' % (colmap['az'], eastvswest)
    if extraSql is not None:
        sql = '(%s) and (%s)' % (sql, extraSql)
    plotDict = {'color': 'orange', 'label': 'East'}
    bundle = mb.MetricBundle(metric, slicer, sql, metadata=extraMetadata,
                             displayDict=displayDict, plotDict=plotDict)
    bundleList.append(bundle)

    displayDict['caption'] = 'Number of visits per night that occur with azimuth > 180.'
    if extraSql is not None:
        displayDict['caption'] += ' With additional sql constraint %s.' % extraSql
    metric = metrics.CountMetric(colmap['night'], metricName='Nvisits West')
    slicer = slicers.OneDSlicer(sliceColName=colmap['night'], binsize=1)
    sql = '%s > %f' % (colmap['az'], eastvswest)
    if extraSql is not None:
        sql = '(%s) and (%s)' % (sql, extraSql)
    plotDict = {'color': 'blue', 'label': 'West'}
    bundle = mb.MetricBundle(metric, slicer, sql, metadata=extraMetadata,
                             displayDict=displayDict, plotDict=plotDict)
    bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)
