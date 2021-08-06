import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.metricBundles as mb
import rubin_sim.maf.plots as plots
from .colMapDict import ColMapDict
from .common import filterList

__all__ = ['altazHealpix', 'altazLambert']


def basicSetup(metricName, colmap=None, nside=64):

    if colmap is None:
        colmap = ColMapDict('opsimV4')

    slicer = slicers.HealpixSlicer(nside=nside, latCol=colmap['alt'], lonCol=colmap['az'],
                                   latLonDeg=colmap['raDecDeg'], useCache=False)
    metric = metrics.CountMetric(colmap['mjd'], metricName=metricName)

    return colmap, slicer, metric


def altazHealpix(colmap=None, runName='opsim', extraSql=None,
                 extraMetadata=None, metricName='NVisits Alt/Az'):

    """Generate a set of metrics measuring the number visits as a function of alt/az
    plotted on a HealpixSkyMap.

    Parameters
    ----------
    colmap : dict, optional
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : str, optional
        The name of the simulated survey. Default is "opsim".
    extraSql : str, optional
        Additional constraint to add to any sql constraints (e.g. 'propId=1' or 'fieldID=522').
        Default None, for no additional constraints.
    extraMetadata : str, optional
        Additional metadata to add before any below (i.e. "WFD").  Default is None.
    metricName : str, optional
        Unique name to assign to metric

    Returns
    -------
    metricBundleDict
    """

    colmap, slicer, metric = basicSetup(metricName=metricName, colmap=colmap)

    # Set up basic all and per filter sql constraints.
    filterlist, colors, orders, sqls, metadata = filterList(all=True,
                                                            extraSql=extraSql,
                                                            extraMetadata=extraMetadata)

    bundleList = []

    plotDict = {'rot': (90, 90, 90), 'flip': 'geo'}
    plotFunc = plots.HealpixSkyMap()

    for f in filterlist:
        if f == 'all':
            subgroup = 'All Observations'
        else:
            subgroup = 'Per filter'
        displayDict = {'group': 'Alt/Az', 'order': orders[f], 'subgroup': subgroup,
                       'caption':
                       'Pointing History on the alt-az sky (zenith center) for filter %s' % f}
        bundle = mb.MetricBundle(metric, slicer, sqls[f], plotDict=plotDict,
                                 runName=runName, metadata = metadata[f],
                                 plotFuncs=[plotFunc], displayDict=displayDict)
        bundleList.append(bundle)

    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def altazLambert(colmap=None, runName='opsim', extraSql=None,
                 extraMetadata=None, metricName='Nvisits as function of Alt/Az'):

    """Generate a set of metrics measuring the number visits as a function of alt/az
    plotted on a LambertSkyMap.

    Parameters
    ----------
    colmap : dict, optional
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : str, optional
        The name of the simulated survey. Default is "opsim".
    extraSql : str, optional
        Additional constraint to add to any sql constraints (e.g. 'propId=1' or 'fieldID=522').
        Default None, for no additional constraints.
    extraMetadata : str, optional
        Additional metadata to add before any below (i.e. "WFD").  Default is None.
    metricName : str, optional
        Unique name to assign to metric

    Returns
    -------
    metricBundleDict
    """

    colmap, slicer, metric = basicSetup(metricName=metricName, colmap=colmap)

    # Set up basic all and per filter sql constraints.
    filterlist, colors, orders, sqls, metadata = filterList(all=True,
                                                            extraSql=extraSql,
                                                            extraMetadata=extraMetadata)

    bundleList = []

    plotFunc = plots.LambertSkyMap()

    for f in filterlist:
        if f == 'all':
            subgroup = 'All Observations'
        else:
            subgroup = 'Per filter'
        displayDict = {'group': 'Alt/Az', 'order': orders[f], 'subgroup': subgroup,
                       'caption':
                       'Alt/Az pointing distribution for filter %s' % f}
        bundle = mb.MetricBundle(metric, slicer, sqls[f],
                                 runName=runName, metadata = metadata[f],
                                 plotFuncs=[plotFunc], displayDict=displayDict)
        bundleList.append(bundle)

    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)
