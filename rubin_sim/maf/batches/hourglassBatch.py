"""Run the hourglass metric.
"""
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.metricBundles as mb
from .colMapDict import ColMapDict

__all__ = ['hourglassPlots']


def hourglassPlots(colmap=None, runName='opsim', nyears=10, extraSql=None, extraMetadata=None):
    """Run the hourglass metric, for each individual year.

    Parameters
    ----------
    colmap : dict, optional
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    run_name : str, optional
        The name of the simulated survey. Default is "opsim".
    nyears : int (10), optional
        How many years to attempt to make hourglass plots for. Default is 10.
    extraSql : str, optional
        Add an extra sql constraint before running metrics. Default None.
    extraMetadata : str, optional
        Add an extra piece of metadata before running metrics. Default None.
    """
    if colmap is None:
        colmap = ColMapDict('opsimV4')
    bundleList = []

    sql = ''
    metadata = ''
    # Add additional sql constraint (such as wfdWhere) and metadata, if provided.
    if (extraSql is not None) and (len(extraSql) > 0):
        sql = extraSql
        if extraMetadata is None:
            metadata = extraSql.replace('filter =', '').replace('filter=', '')
            metadata = metadata.replace('"', '').replace("'", '')
    if extraMetadata is not None:
        metadata = extraMetadata

    years = list(range(nyears + 1))
    displayDict = {'group': 'Hourglass'}
    for year in years[1:]:
        displayDict['subgroup'] = 'Year %d' % year
        displayDict['caption'] = 'Visualization of the filter usage of the telescope. ' \
                                 'The black wavy line indicates lunar phase; the red and blue ' \
                                 'solid lines indicate nautical and civil twilight.'
        sqlconstraint = 'night > %i and night <= %i' % (365.25 * (year - 1), 365.25 * year)
        if len(sql) > 0:
            sqlconstraint = '(%s) and (%s)' % (sqlconstraint, sql)
        md = metadata + ' year %i-%i' % (year - 1, year)
        slicer = slicers.HourglassSlicer()
        metric = metrics.HourglassMetric(nightCol=colmap['night'], mjdCol=colmap['mjd'],
                                         metricName='Hourglass')
        bundle = mb.MetricBundle(metric, slicer, constraint=sqlconstraint, metadata=md,
                                 displayDict=displayDict)
        bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)
