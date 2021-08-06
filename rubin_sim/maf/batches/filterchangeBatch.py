import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.metricBundles as mb
from .colMapDict import ColMapDict
from .common import standardSummary

__all__ = ['filtersPerNight', 'filtersWholeSurvey']


def setupMetrics(colmap, wholesurvey=False):
    metricList = []
    captionList = []
    # Number of filter changes (per slice - either whole survey or X nights)
    if wholesurvey:
        metricList.append(metrics.NChangesMetric(col=colmap['filter'], orderBy=colmap['mjd'],
                                                 metricName='Total Filter Changes'))
    else:
        metricList.append(metrics.NChangesMetric(col=colmap['filter'], orderBy=colmap['mjd'],
                                                 metricName='Filter Changes'))
    captionList.append('Total filter changes ')
    # Minimum time between filter changes
    metricList.append(metrics.MinTimeBetweenStatesMetric(changeCol=colmap['filter'], timeCol=colmap['mjd']))
    captionList.append('Minimum time between filter changes ')
    # Number of filter changes faster than 10 minutes
    metricList.append(metrics.NStateChangesFasterThanMetric(changeCol=colmap['filter'], timeCol=colmap['mjd'],
                                                            cutoff=10))
    captionList.append('Number of filter changes faster than 10 minutes ')
    # Number of filter changes faster than 20 minutes
    metricList.append(metrics.NStateChangesFasterThanMetric(changeCol=colmap['filter'], timeCol=colmap['mjd'],
                                                            cutoff=20))
    captionList.append('Number of filter changes faster than 20 minutes ')
    # Maximum number of filter changes faster than 10 minutes within slice
    metricList.append(metrics.MaxStateChangesWithinMetric(changeCol=colmap['filter'], timeCol=colmap['mjd'],
                                                          timespan=10))
    captionList.append('Max number of filter  changes within a window of 10 minutes ')
    # Maximum number of filter changes faster than 20 minutes within slice
    metricList.append(metrics.MaxStateChangesWithinMetric(changeCol=colmap['filter'], timeCol=colmap['mjd'],
                                                          timespan=20))
    captionList.append('Max number of filter changes within a window of 20 minutes ')
    return metricList, captionList


def filtersPerNight(colmap=None, runName='opsim', nights=1, extraSql=None, extraMetadata=None):
    """Generate a set of metrics measuring the number and rate of filter changes over a given span of nights.

    Parameters
    ----------
    colmap : dict, optional
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    run_name : str, optional
        The name of the simulated survey. Default is "opsim".
    nights : int, optional
        Size of night bin to use when calculating metrics.  Default is 1.
    extraSql : str, optional
        Additional constraint to add to any sql constraints (e.g. 'propId=1' or 'fieldID=522').
        Default None, for no additional constraints.
    extraMetadata : str, optional
        Additional metadata to add before any below (i.e. "WFD").  Default is None.

    Returns
    -------
    metricBundleDict
    """

    if colmap is None:
        colmap = ColMapDict('opsimV4')
    bundleList = []

    # Set up sql and metadata, if passed any additional information.
    sql = ''
    metadata = 'Per'
    if nights == 1:
        metadata += ' Night'
    else:
        metadata += ' %s Nights' % nights
    metacaption = metadata.lower()
    if (extraSql is not None) and (len(extraSql) > 0):
        sql = extraSql
        if extraMetadata is None:
            metadata += ' %s' % extraSql
            metacaption += ', with %s selection' % extraSql
    if extraMetadata is not None:
        metadata += ' %s' % extraMetadata
        metacaption += ', %s only' % extraMetadata
    metacaption += '.'

    displayDict = {'group': 'Filter Changes', 'subgroup': metadata}
    summaryStats = standardSummary()

    slicer = slicers.OneDSlicer(sliceColName=colmap['night'], binsize=nights)
    metricList, captionList = setupMetrics(colmap)
    for m, caption in zip(metricList, captionList):
        displayDict['caption'] = caption + metacaption
        bundle = mb.MetricBundle(m, slicer, sql, runName=runName, metadata=metadata,
                                 displayDict=displayDict,
                                 summaryMetrics=summaryStats)
        bundleList.append(bundle)

    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def filtersWholeSurvey(colmap=None, runName='opsim', extraSql=None, extraMetadata=None):
    """Generate a set of metrics measuring the number and rate of filter changes over the entire survey.

    Parameters
    ----------
    colmap : dict, optional
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    run_name : str, optional
        The name of the simulated survey. Default is "opsim".
    extraSql : str, optional
        Additional constraint to add to any sql constraints (e.g. 'propId=1' or 'fieldID=522').
        Default None, for no additional constraints.
    extraMetadata : str, optional
        Additional metadata to add before any below (i.e. "WFD").  Default is None.

    Returns
    -------
    metricBundleDict
    """
    if colmap is None:
        colmap = ColMapDict('opsimV4')
    bundleList = []

    # Set up sql and metadata, if passed any additional information.
    sql = ''
    metadata = 'Whole Survey'
    metacaption = 'over the whole survey'
    if (extraSql is not None) and (len(extraSql) > 0):
        sql = extraSql
        if extraMetadata is None:
            metadata += ' %s' % extraSql
            metacaption += ', with %s selction' % extraSql
    if extraMetadata is not None:
        metadata += ' %s' % extraMetadata
        metacaption += ', %s only' % (extraMetadata)
    metacaption += '.'

    displayDict = {'group': 'Filter Changes', 'subgroup': metadata}

    slicer = slicers.UniSlicer()
    metricList, captionList = setupMetrics(colmap)
    for m, caption in zip(metricList, captionList):
        displayDict['caption'] = caption + metacaption
        bundle = mb.MetricBundle(m, slicer, sql, runName=runName, metadata=metadata,
                                 displayDict=displayDict)
        bundleList.append(bundle)

    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)
