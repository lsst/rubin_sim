"""Sets of slew metrics.
"""
import warnings
import numpy as np
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.metricBundles as mb
from .colMapDict import ColMapDict
from .common import standardMetrics, combineMetadata

__all__ = ['slewBasics', 'slewAngles', 'slewSpeeds', 'slewActivities']


def slewBasics(colmap=None, runName='opsim', sqlConstraint=None):
    """Generate a simple set of statistics about the slew times and distances.
    These slew statistics can be run on the summary or default tables.

    Parameters
    ----------
    colmap : dict or None, optional
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : str, optional
        The name of the simulated survey. Default is "opsim".
    sqlConstraint : str or None, optional
        SQL constraint to add to metrics. (note this runs on summary table).

    Returns
    -------
    metricBundleDict
    """

    if colmap is None:
        colmap = ColMapDict('opsimV4')

    bundleList = []

    # Calculate basic stats on slew times. (mean/median/min/max + total).
    slicer = slicers.UniSlicer()

    metadata = 'All visits'
    if sqlConstraint is not None and len(sqlConstraint) > 0:
        metadata = '%s' % (sqlConstraint)
    displayDict = {'group': 'Slew', 'subgroup': 'Slew Basics', 'order': -1, 'caption': None}
    # Add total number of slews.
    metric = metrics.CountMetric(colmap['slewtime'], metricName='Slew Count')
    displayDict['caption'] = 'Total number of slews recorded in summary table.'
    displayDict['order'] += 1
    bundle = mb.MetricBundle(metric, slicer, sqlConstraint, metadata=metadata, displayDict=displayDict)
    bundleList.append(bundle)
    for metric in standardMetrics(colmap['slewtime']):
        displayDict['caption'] = '%s in seconds.' % (metric.name)
        displayDict['order'] += 1
        bundle = mb.MetricBundle(metric, slicer, sqlConstraint, metadata=metadata, displayDict=displayDict)
        bundleList.append(bundle)

    # Slew Time histogram.
    slicer = slicers.OneDSlicer(sliceColName=colmap['slewtime'], binsize=2)
    metric = metrics.CountMetric(col=colmap['slewtime'], metricName='Slew Time Histogram')
    metadata = 'All visits'
    plotDict = {'logScale': True, 'ylabel': 'Count'}
    displayDict['caption'] = 'Histogram of slew times (seconds) for all visits.'
    displayDict['order'] += 1
    bundle = mb.MetricBundle(metric, slicer, sqlConstraint, metadata=metadata,
                             plotDict=plotDict, displayDict=displayDict)
    bundleList.append(bundle)
    # Zoom in on slew time histogram near 0.
    slicer = slicers.OneDSlicer(sliceColName=colmap['slewtime'], binsize=0.2, binMin=0, binMax=20)
    metric = metrics.CountMetric(col=colmap['slewtime'], metricName='Zoom Slew Time Histogram')
    metadata = 'All visits'
    plotDict = {'logScale': True, 'ylabel': 'Count'}
    displayDict['caption'] = 'Histogram of slew times (seconds) for all visits (zoom).'
    displayDict['order'] += 1
    bundle = mb.MetricBundle(metric, slicer, sqlConstraint, metadata=metadata,
                             plotDict=plotDict, displayDict=displayDict)
    bundleList.append(bundle)

    # Slew distance histogram, if available.
    if colmap['slewdist'] is not None:
        binsize = 2.0
        if not colmap['raDecDeg']:
            binsize = np.radians(binsize)
        slicer = slicers.OneDSlicer(sliceColName=colmap['slewdist'], binsize=binsize)
        metric = metrics.CountMetric(col=colmap['slewdist'], metricName='Slew Distance Histogram')
        plotDict = {'logScale': True, 'ylabel': 'Count'}
        displayDict['caption'] = 'Histogram of slew distances (angle) for all visits.'
        displayDict['order'] += 1
        bundle = mb.MetricBundle(metric, slicer, sqlConstraint, metadata=metadata,
                                 plotDict=plotDict, displayDict=displayDict)
        bundleList.append(bundle)
        # Zoom on slew distance histogram.
        binMax = 20.0
        if not colmap['raDecDeg']:
            binMax = np.radians(binMax)
        slicer = slicers.OneDSlicer(sliceColName=colmap['slewdist'], binsize=binsize/10.,
                                    binMin=0, binMax=binMax)
        metric = metrics.CountMetric(col=colmap['slewdist'], metricName='Zoom Slew Distance Histogram')
        plotDict = {'logScale': True, 'ylabel': 'Count'}
        displayDict['caption'] = 'Histogram of slew distances (angle) for all visits.'
        displayDict['order'] += 1
        bundle = mb.MetricBundle(metric, slicer, sqlConstraint, metadata=metadata,
                                 plotDict=plotDict, displayDict=displayDict)
        bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def slewAngles(colmap=None, runName='opsim', sqlConstraint=None):
    """Generate a set of slew statistics focused on the angles of each component (dome and telescope).
    These slew statistics must be run on the SlewFinalState or SlewInitialState table in opsimv4,
    and on the SlewState table in opsimv3.

    Parameters
    ----------
    colmap : dict or None, optional
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : str, optional
        The name of the simulated survey. Default is "opsim".
    sqlConstraint : str or None, optional
        SQL constraint to apply to metrics. Note this runs on Slew*State table, so constraints
        should generally be based on slew_slewCount.

    Returns
    -------
    metricBundleDict
    """
    if colmap is None:
        colmap = ColMapDict('opsimV4')
    bundleList = []

    # All of these metrics are run with a unislicer.
    slicer = slicers.UniSlicer()

    # For each angle, we will compute mean/median/min/max and rms.
    # Note that these angles can range over more than 360 degrees, because of cable wrap.
    # This is why we're not using the Angle metrics - here 380 degrees is NOT the same as 20 deg.
    # Stats for angle:
    angles = ['Tel Alt', 'Tel Az', 'Rot Tel Pos']

    displayDict = {'group': 'Slew', 'subgroup': 'Slew Angles', 'order': -1, 'caption': None}
    for angle in angles:
        metadata = combineMetadata(angle, sqlConstraint)
        metriclist = standardMetrics(colmap[angle], replace_colname='')
        metriclist += [metrics.RmsMetric(colmap[angle], metricName='RMS')]
        for metric in metriclist:
            displayDict['caption'] = '%s %s' % (metric.name, angle)
            displayDict['order'] += 1
            bundle = mb.MetricBundle(metric, slicer, sqlConstraint,
                                     displayDict=displayDict, metadata=metadata)
            bundleList.append(bundle)

    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def slewSpeeds(colmap=None, runName='opsim', sqlConstraint=None):
    """Generate a set of slew statistics focused on the speeds of each component (dome and telescope).
    These slew statistics must be run on the SlewMaxSpeeds table in opsimv4 and opsimv3.

    Parameters
    ----------
    colmap : dict or None, optional
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
        Note that for these metrics, the column names are distinctly different in v3/v4.
    runName : str, optional
        The name of the simulated survey. Default is "opsim".
    sqlConstraint : str or None, optional
        SQL constraint to apply to metrics. Note this runs on Slew*State table, so constraints
        should generally be based on slew_slewCount.

    Returns
    -------
    metricBundleDict
    """
    if colmap is None:
        colmap = ColMapDict('opsimV4')
    bundleList = []

    # All of these metrics run with a unislicer, on all the slew data.
    slicer = slicers.UniSlicer()

    speeds = ['Dome Alt Speed', 'Dome Az Speed', 'Tel Alt Speed', 'Tel Az Speed', 'Rotator Speed']

    displayDict = {'group': 'Slew', 'subgroup': 'Slew Speeds', 'order': -1, 'caption': None}
    for speed in speeds:
        metadata = combineMetadata(speed, sqlConstraint)
        metric = metrics.AbsMaxMetric(col=colmap[speed], metricName='Max (Abs)')
        displayDict['caption'] = 'Maximum absolute value of %s.' % speed
        displayDict['order'] += 1
        bundle = mb.MetricBundle(metric, slicer, sqlConstraint, displayDict=displayDict, metadata=metadata)
        bundleList.append(bundle)

        metric = metrics.AbsMeanMetric(col=colmap[speed], metricName='Mean (Abs)')
        displayDict['caption'] = 'Mean absolute value of %s.' % speed
        displayDict['order'] += 1
        bundle = mb.MetricBundle(metric, slicer, sqlConstraint, displayDict=displayDict, metadata=metadata)
        bundleList.append(bundle)

        metric = metrics.AbsMaxPercentMetric(col=colmap[speed], metricName='% @ Max')
        displayDict['caption'] = 'Percent of slews at the maximum %s (absolute value).' % speed
        displayDict['order'] += 1
        bundle = mb.MetricBundle(metric, slicer, sqlConstraint, displayDict=displayDict, metadata=metadata)
        bundleList.append(bundle)

    for b in bundleList:
        b.setRunName(runName)

    return mb.makeBundlesDictFromList(bundleList)


def slewActivities(colmap=None, runName='opsim', totalSlewN=1, sqlConstraint=None):
    """Generate a set of slew statistics focused on finding the contributions to the overall slew time.
    These slew statistics must be run on the SlewActivities table in opsimv4 and opsimv3.

    Note that the type of activities listed are different between v3 and v4.

    Parameters
    ----------
    colmap : dict or None, optional
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : str, optional
        The name of the simulated survey. Default is "opsim".
    totalSlewN : int, optional
        The total number of slews in the simulated survey.
        Used to calculate % of slew activities for each component.
        Default is 1.
    sqlConstraint : str or None, optional
        SQL constraint to apply to metrics. Note this runs on Slew*State table, so constraints
        should generally be based on slew_slewCount.


    Returns
    -------
    metricBundleDict
    """
    if totalSlewN == 1:
        warnings.warn('TotalSlewN should be set (using 1). Percents from activities may be incorrect.')

    if colmap is None:
        colmap = ColMapDict('opsimV4')
    bundleList = []

    # All of these metrics run with a unislicer, on all the slew data.
    slicer = slicers.UniSlicer()

    if 'slewactivities' not in colmap:
        raise ValueError("List of slewactivities not in colmap! Will not create slewActivities bundles.")

    slewTypeDict = colmap['slewactivities']

    displayDict = {'group': 'Slew', 'subgroup': 'Slew Activities', 'order': -1, 'caption': None}

    for slewType in slewTypeDict:
        metadata = combineMetadata(slewType, sqlConstraint)
        tableValue = slewTypeDict[slewType]

        # Metrics for all activities of this type.
        sql = 'activityDelay>0 and activity="%s"' % tableValue
        if sqlConstraint is not None:
            sql = '(%s) and (%s)' % (sql, sqlConstraint)

        # Percent of slews which include this activity.
        metric = metrics.CountRatioMetric(col='activityDelay', normVal=totalSlewN / 100.0,
                                          metricName='ActivePerc')
        displayDict['caption'] = 'Percent of total slews which include %s movement.' % slewType
        displayDict['order'] += 1
        bundle = mb.MetricBundle(metric, slicer, sql, displayDict=displayDict, metadata=metadata)
        bundleList.append(bundle)

        # Mean time for this activity, in all slews.
        metric = metrics.MeanMetric(col='activityDelay', metricName='Ave T(s)')
        displayDict['caption'] = 'Mean amount of time (in seconds) for %s movements.' % (slewType)
        displayDict['order'] += 1
        bundle = mb.MetricBundle(metric, slicer, sql, displayDict=displayDict, metadata=metadata)
        bundleList.append(bundle)

        # Maximum time for this activity, in all slews.
        metric = metrics.MaxMetric(col='activityDelay', metricName='Max T(s)')
        displayDict['caption'] = 'Max amount of time (in seconds) for %s movement.' % (slewType)
        displayDict['order'] += 1
        bundle = mb.MetricBundle(metric, slicer, sql, displayDict=displayDict, metadata=metadata)
        bundleList.append(bundle)

        # Metrics for activities of this type which are in the critical path.
        sql = 'activityDelay>0 and inCriticalPath="True" and activity="%s"' % tableValue
        if sqlConstraint is not None:
            sql = '(%s) and (%s)' % (sql, sqlConstraint)

        # Percent of slews which include this activity in the critical path.
        metric = metrics.CountRatioMetric(col='activityDelay', normVal=totalSlewN / 100.0,
                                          metricName='ActivePerc in crit')
        displayDict['caption'] = 'Percent of total slew which include %s movement, ' \
                                 'and are in critical path.' % (slewType)
        displayDict['order'] += 1
        bundle = mb.MetricBundle(metric, slicer, sql, displayDict=displayDict, metadata=metadata)
        bundleList.append(bundle)

        # Mean time for slews which include this activity, in the critical path.
        metric = metrics.MeanMetric(col='activityDelay', metricName='Ave T(s) in crit')
        displayDict['caption'] = 'Mean time (in seconds) for %s movements, ' \
                                 'when in critical path.' % (slewType)
        displayDict['order'] += 1
        bundle = mb.MetricBundle(metric, slicer, sql, displayDict=displayDict, metadata=metadata)
        bundleList.append(bundle)

        # Total time that this activity was in the critical path.
        metric = metrics.SumMetric(col='activityDelay', metricName='Total T(s) in crit')
        displayDict['caption'] = 'Total time (in seconds) for %s movements, ' \
                                 'when in critical path.' % (slewType)
        displayDict['order'] += 1
        bundle = mb.MetricBundle(metric, slicer, sql, displayDict=displayDict, metadata=metadata)
        bundleList.append(bundle)

    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)
