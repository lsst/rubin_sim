"""Sets of metrics to look at general sky coverage - nvisits/coadded depth/Teff.
"""
import numpy as np
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.plots as plots
import rubin_sim.maf.metricBundles as mb
import rubin_sim.maf.utils as mafUtils
from .colMapDict import ColMapDict, getColMap
from .common import standardSummary, filterList, radecCols, combineMetadata

__all__ = ['nvisitsM5Maps', 'tEffMetrics', 'nvisitsPerNight', 'nvisitsPerProp']


def nvisitsM5Maps(colmap=None, runName='opsim',
                  extraSql=None, extraMetadata=None,
                  nside=64, runLength=10.,
                  ditherStacker=None, ditherkwargs=None):
    """Generate number of visits and Coadded depth per RA/Dec point in all and per filters.

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
    nside : int, optional
        Nside value for healpix slicer. Default 64.
        If "None" is passed, the healpixslicer-based metrics will be skipped.
    runLength : float, optional
        Length of the simulated survey, for scaling values for the plot limits.
        Default 10.
    ditherStacker: str or rubin_sim.maf.stackers.BaseDitherStacker
        Optional dither stacker to use to define ra/dec columns.
    ditherkwargs: dict, optional
        Optional dictionary of kwargs for the dither stacker.

    Returns
    -------
    metricBundleDict
    """
    if colmap is None:
        colmap = ColMapDict('opsimV4')
    bundleList = []

    subgroup = extraMetadata
    if subgroup is None:
        subgroup = 'All visits'

    raCol, decCol, degrees, ditherStacker, ditherMeta = radecCols(ditherStacker, colmap, ditherkwargs)
    extraMetadata = combineMetadata(extraMetadata, ditherMeta)
    # Set up basic all and per filter sql constraints.
    filterlist, colors, orders, sqls, metadata = filterList(all=True,
                                                            extraSql=extraSql,
                                                            extraMetadata=extraMetadata)
    # Set up some values to make nicer looking plots.
    benchmarkVals = mafUtils.scaleBenchmarks(runLength, benchmark='design')
    # Check that nvisits is not set to zero (for very short run length).
    for f in benchmarkVals['nvisits']:
        if benchmarkVals['nvisits'][f] == 0:
            print('Updating benchmark nvisits value in %s to be nonzero' % (f))
            benchmarkVals['nvisits'][f] = 1
    benchmarkVals['coaddedDepth'] = mafUtils.calcCoaddedDepth(benchmarkVals['nvisits'],
                                                              benchmarkVals['singleVisitDepth'])
    # Scale the nvisit ranges for the runLength.
    nvisitsRange = {'u': [20, 80], 'g': [50, 150], 'r': [100, 250],
                    'i': [100, 250], 'z': [100, 300], 'y': [100, 300], 'all': [700, 1200]}
    scale = runLength / 10.0
    for f in nvisitsRange:
        for i in [0, 1]:
            nvisitsRange[f][i] = int(np.floor(nvisitsRange[f][i] * scale))

    # Generate Nvisit maps in all and per filters
    displayDict = {'group': 'Nvisits Maps', 'subgroup': subgroup}
    metric = metrics.CountMetric(colmap['mjd'], metricName='NVisits', units='')
    slicer = slicers.HealpixSlicer(nside=nside, latCol=decCol, lonCol=raCol,
                                   latLonDeg=degrees)
    for f in filterlist:
        sql = sqls[f]
        displayDict['caption'] = 'Number of visits per healpix in %s.' % metadata[f]
        displayDict['order'] = orders[f]
        binsize = 2
        if f == 'all':
            binsize = 5
        plotDict = {'xMin': nvisitsRange[f][0], 'xMax': nvisitsRange[f][1],
                    'colorMin': nvisitsRange[f][0], 'colorMax': nvisitsRange[f][1],
                    'binsize': binsize, 'color': colors[f]}
        bundle = mb.MetricBundle(metric, slicer, sql, metadata=metadata[f],
                                 stackerList=ditherStacker,
                                 displayDict=displayDict, plotDict=plotDict,
                                 summaryMetrics=standardSummary())
        bundleList.append(bundle)

    # Generate Coadded depth maps per filter
    displayDict = {'group': 'Coadded M5 Maps', 'subgroup': subgroup}
    metric = metrics.Coaddm5Metric(m5Col=colmap['fiveSigmaDepth'], metricName='CoaddM5')
    slicer = slicers.HealpixSlicer(nside=nside, latCol=decCol, lonCol=raCol,
                                   latLonDeg=degrees)
    for f in filterlist:
        # Skip "all" for coadded depth.
        if f == 'all':
            continue
        mag_zp = benchmarkVals['coaddedDepth'][f]
        sql = sqls[f]
        displayDict['caption'] = 'Coadded depth per healpix, with %s benchmark value subtracted (%.1f) ' \
                                 'in %s.' % (f, mag_zp, metadata[f])
        displayDict['caption'] += ' More positive numbers indicate fainter limiting magnitudes.'
        displayDict['order'] = orders[f]
        plotDict = {'zp': mag_zp, 'xMin': -0.6, 'xMax': 0.6,
                    'xlabel': 'coadded m5 - %.1f' % mag_zp,
                    'colorMin': -0.6, 'colorMax': 0.6, 'color': colors[f]}
        bundle = mb.MetricBundle(metric, slicer, sql, metadata=metadata[f],
                                 stackerList=ditherStacker,
                                 displayDict=displayDict, plotDict=plotDict,
                                 summaryMetrics=standardSummary())
        bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def tEffMetrics(colmap=None, runName='opsim',
                extraSql=None, extraMetadata=None, nside=64,
                ditherStacker=None, ditherkwargs=None):
    """Generate a series of Teff metrics. Teff total, per night, and sky maps (all and per filter).

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
    nside : int, optional
        Nside value for healpix slicer. Default 64.
        If "None" is passed, the healpixslicer-based metrics will be skipped.
    ditherStacker: str or rubin_sim.maf.stackers.BaseDitherStacker
        Optional dither stacker to use to define ra/dec columns.
    ditherkwargs: dict, optional
        Optional dictionary of kwargs for the dither stacker.

    Returns
    -------
    metricBundleDict
    """
    if colmap is None:
        colmap = ColMapDict('opsimV4')
    bundleList = []

    subgroup = extraMetadata
    if subgroup is None:
        subgroup = 'All visits'

    raCol, decCol, degrees, ditherStacker, ditherMeta = radecCols(ditherStacker, colmap, ditherkwargs)
    extraMetadata = combineMetadata(extraMetadata, ditherMeta)

    # Set up basic all and per filter sql constraints.
    filterlist, colors, orders, sqls, metadata = filterList(all=True,
                                                            extraSql=extraSql,
                                                            extraMetadata=extraMetadata)
    if metadata['all'] is None:
        metadata['all'] = 'All visits'

    subsetPlots = [plots.HealpixSkyMap(), plots.HealpixHistogram()]

    # Total Teff and normalized Teff.
    displayDict = {'group': 'T_eff Summary', 'subgroup': subgroup}
    displayDict['caption'] = 'Total effective time of the survey (see Teff metric).'
    displayDict['order'] = 0
    metric = metrics.TeffMetric(m5Col=colmap['fiveSigmaDepth'], filterCol=colmap['filter'],
                                normed=False, metricName='Total Teff')
    slicer = slicers.UniSlicer()
    bundle = mb.MetricBundle(metric, slicer, constraint=sqls['all'], displayDict=displayDict,
                             metadata=metadata['all'])
    bundleList.append(bundle)

    displayDict['caption'] = 'Normalized total effective time of the survey (see Teff metric).'
    displayDict['order'] = 1
    metric = metrics.TeffMetric(m5Col=colmap['fiveSigmaDepth'], filterCol=colmap['filter'],
                                normed=True, metricName='Normalized Teff')
    slicer = slicers.UniSlicer()
    bundle = mb.MetricBundle(metric, slicer, constraint=sqls['all'], displayDict=displayDict,
                             metadata=metadata['all'])
    bundleList.append(bundle)

    # Generate Teff maps in all and per filters
    displayDict = {'group': 'T_eff Maps', 'subgroup': subgroup}
    if ditherMeta is not None:
        for m in metadata:
            metadata[m] = combineMetadata(metadata[m], ditherMeta)

    metric = metrics.TeffMetric(m5Col=colmap['fiveSigmaDepth'], filterCol=colmap['filter'],
                                normed=True, metricName='Normalized Teff')
    slicer = slicers.HealpixSlicer(nside=nside, latCol=decCol, lonCol=raCol,
                                   latLonDeg=degrees)
    for f in filterlist:
        displayDict['caption'] = 'Normalized effective time of the survey, for %s' % metadata[f]
        displayDict['order'] = orders[f]
        plotDict = {'color': colors[f]}
        bundle = mb.MetricBundle(metric, slicer, sqls[f], metadata=metadata[f],
                                 stackerList=ditherStacker,
                                 displayDict=displayDict, plotFuncs=subsetPlots, plotDict=plotDict,
                                 summaryMetrics=standardSummary())
        bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def nvisitsPerNight(colmap=None, runName='opsim', binNights=1,
                    extraSql=None, extraMetadata=None, subgroup=None):
    """Count the number of visits per night through the survey.

    Parameters
    ----------
    colmap : dict or None, optional
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : str, optional
        The name of the simulated survey. Default is "opsim".
    binNights : int, optional
        Number of nights to count in each bin. Default = 1, count number of visits in each night.
    extraSql : str or None, optional
        Additional constraint to add to any sql constraints (e.g. 'propId=1' or 'fieldID=522').
        Default None, for no additional constraints.
    extraMetadata : str or None, optional
        Additional metadata to add before any below (i.e. "WFD").  Default is None.
    subgroup : str or None, optional
        Use this for the 'subgroup' in the displayDict, instead of metadata. Default is None.

    Returns
    -------
    metricBundleDict
    """
    if colmap is None:
        colmap = ColMapDict('opsimV4')

    subgroup = subgroup
    if subgroup is None:
        subgroup = extraMetadata
        if subgroup is None:
            subgroup = 'All visits'

    metadataCaption = extraMetadata
    if extraMetadata is None:
        if extraSql is not None:
            metadataCaption = extraSql
        else:
            metadataCaption = 'all visits'

    bundleList = []

    displayDict = {'group': 'Nvisits Per Night', 'subgroup': subgroup}
    displayDict['caption'] = 'Number of visits per night for %s.' % (metadataCaption)
    displayDict['order'] = 0
    metric = metrics.CountMetric(colmap['mjd'], metricName='Nvisits')
    slicer = slicers.OneDSlicer(sliceColName=colmap['night'], binsize=binNights)
    bundle = mb.MetricBundle(metric, slicer, extraSql, metadata=metadataCaption,
                             displayDict=displayDict, summaryMetrics=standardSummary())
    bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def nvisitsPerProp(opsdb, colmap=None, runName='opsim', binNights=1, extraSql=None):
    """Set up a group of all and per-proposal nvisits metrics.

    Parameters
    ----------
    opsdb : rubin_sim.maf.db.Database or rubin_sim.maf.db.OpsimDatabase* object
    colmap : dict or None, optional
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : str, optional
        The name of the simulated survey. Default is "opsim".
    binNights : int, optional
        Number of nights to count in each bin. Default = 1, count number of visits in each night.
    sqlConstraint : str or None, optional
        SQL constraint to add to all metrics.

    Returns
    -------
    metricBundle
    """
    if colmap is None:
        colmap = getColMap(opsdb)

    propids, proptags = opsdb.fetchPropInfo()

    bdict = {}
    bundleList = []

    totvisits = opsdb.fetchNVisits()

    metadata = 'All props'
    if extraSql is not None and len(extraSql) > 0:
        metadata += ' %s' % extraSql
    # Nvisits per night, all proposals.
    bdict.update(nvisitsPerNight(colmap=colmap, runName=runName, binNights=binNights,
                                 extraSql=extraSql, extraMetadata=metadata, subgroup='All proposals'))
    # Nvisits total, all proposals.
    metric = metrics.CountMetric(colmap['mjd'], metricName='Nvisits')
    slicer = slicers.UniSlicer()
    summaryMetrics = [metrics.IdentityMetric(metricName='Count'),
                      metrics.NormalizeMetric(normVal=totvisits, metricName='Fraction of total')]
    displayDict = {'group': 'Nvisit Summary', 'subgroup': 'Proposal distribution', 'order': -1}
    displayDict['caption'] = 'Total number of visits for all proposals.'
    if extraSql is not None and len(extraSql) > 0:
        displayDict['caption'] += ' (with constraint %s.)' % extraSql
    bundle = mb.MetricBundle(metric, slicer, extraSql, metadata=metadata,
                             displayDict=displayDict, summaryMetrics=summaryMetrics)
    bundleList.append(bundle)

    # Look for any multi-proposal groups that we should include.
    for tag in proptags:
        if len(proptags[tag]) > 1:
            pids = proptags[tag]
            sql = '('
            for pid in pids[:-1]:
                sql += '%s=%d or ' % (colmap['proposalId'], pid)
            sql += ' %s=%d)' % (colmap['proposalId'], pids[-1])
            metadata = '%s' % tag
            if extraSql is not None:
                sql = '(%s) and (%s)' % (sql, extraSql)
                metadata += ' %s' % (extraSql)
            bdict.update(nvisitsPerNight(colmap=colmap, runName=runName, binNights=binNights,
                                         extraSql=sql, extraMetadata=metadata, subgroup=tag))
            displayDict['order'] += 1
            displayDict['caption'] = 'Number of visits and fraction of total visits, for %s.' % metadata
            bundle = mb.MetricBundle(metric, slicer, sql, metadata=metadata,
                                     summaryMetrics=summaryMetrics, displayDict=displayDict)
            bundleList.append(bundle)

    # And each proposal separately.
    for propid in propids:
        sql = '%s=%d' % (colmap['proposalId'], propid)
        metadata = '%s' % (propids[propid])
        if extraSql is not None:
            sql += ' and (%s)' % (extraSql)
            metadata += ' %s' % extraSql
        bdict.update(nvisitsPerNight(colmap=colmap, runName=runName, binNights=binNights,
                                     extraSql=sql, extraMetadata=metadata, subgroup='Per proposal'))
        displayDict['order'] += 1
        displayDict['caption'] = 'Number of visits and fraction of total visits, for %s.' % metadata
        bundle = mb.MetricBundle(metric, slicer, constraint=sql, metadata=metadata,
                                 summaryMetrics=summaryMetrics, displayDict=displayDict)
        bundleList.append(bundle)

    for b in bundleList:
        b.setRunName(runName)
    bdict.update(mb.makeBundlesDictFromList(bundleList))
    return bdict
