"""Sets of metrics to look at impact of cadence on science
"""
import numpy as np
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.plots as plots
import rubin_sim.maf.metricBundles as mb
from .colMapDict import ColMapDict
from .common import standardSummary, filterList, combineMetadata, radecCols

__all__ = ['phaseGap']

def phaseGap(colmap=None, runName='opsim', nside=64, extraSql=None, extraMetadata=None,
             ditherStacker=None, ditherkwargs=None):
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
    ditherStacker: str or rubin_sim.maf.stackers.BaseDitherStacker
        Optional dither stacker to use to define ra/dec columns.
    ditherkwargs: dict or None, optional
        Optional dictionary of kwargs for the dither stacker.

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

    raCol, decCol, degrees, ditherStacker, ditherMeta = radecCols(ditherStacker, colmap, ditherkwargs)
    metadata = combineMetadata(metadata, ditherMeta)

    bundleList = []
    standardStats = standardSummary()
    subsetPlots = [plots.HealpixSkyMap(), plots.HealpixHistogram()]

    slicer = slicers.HealpixSlicer(nside=nside, latCol=decCol, lonCol=raCol, latLonDeg=degrees)

    # largest phase gap for periods
    periods = [0.1, 1.0, 10., 100.]
    sqls = ['filter = "u"',
            'filter="r"',
            'filter="g" or filter="r" or filter="i" or filter="z"',
            '']
    filterNames = ['u', 'r', 'griz', 'all']
    metadatas = filterNames
    if metadata is not None:
        metadatas = [combineMetadata(m, metadata) for m in metadatas]
    if extraSql is not None and len(extraSql) > 0:
        for sql in sqls:
            sqls[sql] = '(%s) and (%s)' % (sqls[sql], extraSql)

    for sql, md, f in zip(sqls, metadatas, filterNames):
        for period in periods:
            displayDict = {'group': 'PhaseGap',
                           'subgroup': 'Filter %s: Period %.2f days' % (f, period),
                           'caption': 'Maximum phase gap, given a period of %.2f days.' % period}
            metric = metrics.PhaseGapMetric(nPeriods=1, periodMin=period, periodMax=period, nVisitsMin=5,
                                            metricName='PhaseGap %.1f day' % period)
            metric.reduceFuncs = {metric.reduceFuncs['reduceLargestGap']}
            metric.reduceOrder = {0}
            bundle = mb.MetricBundle(metric, slicer, constraint=sql, metadata=md,
                                     displayDict=displayDict, summaryMetrics=standardStats,
                                     plotFuncs=subsetPlots)
            bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)
