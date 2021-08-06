"""Evaluate the open shutter fraction.
"""
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.metricBundles as mb
from .colMapDict import ColMapDict
from .common import standardSummary

__all__ = ['openshutterFractions']


def openshutterFractions(colmap=None, runName='opsim', extraSql=None, extraMetadata=None):
    """Evaluate open shutter fraction over whole survey and per night.

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

    group = 'Open Shutter Fraction'

    subgroup = 'All visits'
    if extraMetadata is not None:
        subgroup = extraMetadata + ' ' + subgroup.lower()
    elif extraSql is not None and extraMetadata is None:
        subgroup = subgroup + ' ' + extraSql

    # Open Shutter fraction over whole survey.
    displayDict = {'group': group, 'subgroup': subgroup, 'order': 0}
    displayDict['caption'] = 'Total open shutter fraction over %s. ' % subgroup.lower()
    displayDict['caption'] += 'Does not include downtime due to weather.'
    metric = metrics.OpenShutterFractionMetric(slewTimeCol=colmap['slewtime'],
                                               expTimeCol=colmap['exptime'],
                                               visitTimeCol=colmap['visittime'])
    slicer = slicers.UniSlicer()
    bundle = mb.MetricBundle(metric, slicer, extraSql, metadata=subgroup,
                             displayDict=displayDict)
    bundleList.append(bundle)
    # Count the number of nights on-sky in the survey.
    displayDict['caption'] = 'Number of nights on the sky during the survey, %s.' % subgroup.lower()
    metric = metrics.CountUniqueMetric(colmap['night'])
    slicer = slicers.UniSlicer()
    bundle = mb.MetricBundle(metric, slicer, extraSql, metadata=subgroup,
                             displayDict=displayDict)
    bundleList.append(bundle)
    # Count the number of nights total in the survey (start to finish of observations).
    displayDict['caption'] = 'Number of nights from start to finish of survey, %s.' % subgroup.lower()
    metric = metrics.FullRangeMetric(colmap['night'])
    slicer = slicers.UniSlicer()
    bundle = mb.MetricBundle(metric, slicer, extraSql, metadata=subgroup,
                             displayDict=displayDict)
    bundleList.append(bundle)

    # Open shutter fraction per night.
    subgroup = 'Per night'
    if extraMetadata is not None:
        subgroup = extraMetadata + ' ' + subgroup.lower()
    elif extraSql is not None and extraMetadata is None:
        subgroup = subgroup + ' ' + extraSql
    displayDict = {'group': group, 'subgroup': subgroup, 'order': 0}
    displayDict['caption'] = 'Open shutter fraction %s.' % (subgroup.lower())
    displayDict['caption'] += ' This compares on-sky image time against on-sky time + slews + filter ' \
                              'changes + readout, but does not include downtime due to weather.'
    metric = metrics.OpenShutterFractionMetric(slewTimeCol=colmap['slewtime'],
                                               expTimeCol=colmap['exptime'],
                                               visitTimeCol=colmap['visittime'])
    slicer = slicers.OneDSlicer(sliceColName=colmap['night'], binsize=1)
    bundle = mb.MetricBundle(metric, slicer, extraSql, metadata=subgroup,
                             summaryMetrics=standardSummary(), displayDict=displayDict)
    bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)
