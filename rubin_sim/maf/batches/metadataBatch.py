"""Some basic physical quantity metrics.
"""
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.stackers as stackers
import rubin_sim.maf.plots as plots
import rubin_sim.maf.metricBundles as mb
from .colMapDict import ColMapDict
from .common import (
    standardSummary,
    extendedMetrics,
    standardAngleMetrics,
    filterList,
    radecCols,
    combineInfoLabels,
)

__all__ = [
    "metadataBasics",
    "metadataBasicsAngle",
    "allMetadata",
    "metadataMaps",
    "firstYearMetadata",
]


def metadataBasics(
    value,
    colmap=None,
    runName="opsim",
    valueName=None,
    groupName=None,
    extraSql=None,
    extraInfoLabel=None,
    slicer=None,
):
    """Calculate basic metrics on visit metadata 'value' (e.g. airmass, normalized airmass, seeing..).
    Calculates this around the sky (HealpixSlicer), makes histograms of all visits (OneDSlicer),
    and calculates statistics on all visits (UniSlicer) for the quantity in all visits and per filter.

    Currently have a hack for HA & normairmass.

    Parameters
    ----------
    value : str
        The column name for the quantity to evaluate. (column name in the database or created by a stacker).
    colmap : dict or None, optional
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : str, optional
        The name of the simulated survey. Default is "opsim".
    valueName : str, optional
        The name of the value to be reported in the resultsDb and added to the metric.
        This is intended to help standardize metric comparison between sim versions.
        value = name as it is in the database (seeingFwhmGeom, etc).
        valueName = name to be recorded ('seeingGeom', etc.).  Default is None, which will match 'value'.
    groupName : str, optional
        The group name for this quantity in the displayDict. Default is the same as 'valueName', capitalized.
    extraSql : str, optional
        Additional constraint to add to any sql constraints (e.g. 'propId=1' or 'fieldID=522').
        Default None, for no additional constraints.
    extraInfoLabel : str, optional
        Additional info_labels to add before any below (i.e. "WFD").  Default is None.
    slicer : `rubin_sim.maf.slicers.BaseSlicer`, optional
        Optionally use a different slicer than an nside=64 healpix slicer.

    Returns
    -------
    metricBundleDict
    """
    if colmap is None:
        colmap = ColMapDict("fbs")
    bundleList = []

    if valueName is None:
        valueName = value

    if groupName is None:
        groupName = valueName.capitalize()
        subgroup = extraInfoLabel
    else:
        groupName = groupName.capitalize()
        subgroup = valueName.capitalize()

    if subgroup is None:
        subgroup = "All visits"

    displayDict = {"group": groupName, "subgroup": subgroup}

    raCol, decCol, degrees, ditherStacker, ditherMeta = radecCols(None, colmap, None)
    extraInfoLabel = combineInfoLabels(extraInfoLabel, ditherMeta)
    # Set up basic all and per filter sql constraints.
    filterlist, colors, orders, sqls, info_label = filterList(
        all=True, extraSql=extraSql, extraInfoLabel=extraInfoLabel
    )

    if slicer is not None:
        skyslicer = slicer
    else:
        skyslicer = slicers.HealpixSlicer(
            nside=64, lonCol=raCol, latCol=decCol, latLonDeg=degrees
        )

    # Hack to make HA work, but really I need to account for any stackers/colmaps.
    if value == "HA":
        stackerList = [
            stackers.HourAngleStacker(
                lstCol=colmap["lst"], raCol=raCol, degrees=degrees
            )
        ]
    elif value == "normairmass":
        stackerList = [stackers.NormAirmassStacker(degrees=degrees)]
    else:
        stackerList = None

    # Summarize values over all and per filter (min/mean/median/max/percentiles/outliers/rms).
    slicer = slicers.UniSlicer()
    for f in filterlist:
        for m in extendedMetrics(value, replace_colname=valueName):
            displayDict["caption"] = "%s for %s." % (m.name, info_label[f])
            displayDict["order"] = orders[f]
            bundle = mb.MetricBundle(
                m,
                slicer,
                sqls[f],
                stackerList=stackerList,
                info_label=info_label[f],
                displayDict=displayDict,
            )
            bundleList.append(bundle)

    # Histogram values over all and per filter.
    for f in filterlist:
        displayDict["caption"] = "Histogram of %s" % (value)
        if valueName != value:
            displayDict["caption"] += " (%s)" % (valueName)
        displayDict["caption"] += " for %s." % (info_label[f])
        displayDict["order"] = orders[f]
        m = metrics.CountMetric(value, metricName="%s Histogram" % (valueName))
        slicer = slicers.OneDSlicer(sliceColName=value)
        bundle = mb.MetricBundle(
            m,
            slicer,
            sqls[f],
            stackerList=stackerList,
            info_label=info_label[f],
            displayDict=displayDict,
        )
        bundleList.append(bundle)

    # Make maps of min/median/max for all and per filter, per RA/Dec, with standard summary stats.
    plotDict = {"percentileClip": 98}
    mList = []
    mList.append(metrics.MinMetric(value, metricName="Min %s" % (valueName)))
    mList.append(metrics.MedianMetric(value, metricName="Median %s" % (valueName)))
    mList.append(metrics.MaxMetric(value, metricName="Max %s" % (valueName)))
    slicer = skyslicer
    for f in filterlist:
        for m in mList:
            displayDict["caption"] = "Map of %s" % m.name
            if valueName != value:
                displayDict["caption"] += " (%s)" % value
            displayDict["caption"] += " for %s." % info_label[f]
            displayDict["order"] = orders[f]
            bundle = mb.MetricBundle(
                m,
                slicer,
                sqls[f],
                stackerList=stackerList,
                info_label=info_label[f],
                plotDict=plotDict,
                displayDict=displayDict,
                summaryMetrics=standardSummary(),
            )
            bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def metadataBasicsAngle(
    value,
    colmap=None,
    runName="opsim",
    valueName=None,
    groupName=None,
    extraSql=None,
    extraInfoLabel=None,
    slicer=None,
):
    """Calculate basic metrics on visit metadata 'value', where value is a wrap-around angle.

    Calculates extended standard metrics (with unislicer) on the quantity (all visits and per filter),
    makes histogram of the value (all visits and per filter),


    Parameters
    ----------
    value : `str`
        The column name for the quantity to evaluate. (column name in the database or created by a stacker).
    colmap : `dict` or None, optional
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : `str`, optional
        The name of the simulated survey. Default is "opsim".
    valueName : `str`, optional
        The name of the value to be reported in the resultsDb and added to the metric.
        This is intended to help standardize metric comparison between sim versions.
        value = name as it is in the database (seeingFwhmGeom, etc).
        valueName = name to be recorded ('seeingGeom', etc.).  Default is None, which will match 'value'.
    groupName : `str`, optional
        The group name for this quantity in the displayDict. Default is the same as 'valueName', capitalized.
    extraSql : `str`, optional
        Additional constraint to add to any sql constraints (e.g. 'propId=1' or 'fieldID=522').
        Default None, for no additional constraints.
    extraInfoLabel : `str`, optional
        Additional info_label to add before any below (i.e. "WFD").  Default is None.
    slicer : `rubin_sim.maf.slicer.BaseSlicer` or None, optional

    Returns
    -------
    metricBundleDict
    """
    if colmap is None:
        colmap = ColMapDict("opsimV4")
    bundleList = []

    if valueName is None:
        valueName = value

    if groupName is None:
        groupName = valueName.capitalize()
        subgroup = extraInfoLabel
    else:
        groupName = groupName.capitalize()
        subgroup = valueName.capitalize()

    if subgroup is None:
        subgroup = "All visits"

    displayDict = {"group": groupName, "subgroup": subgroup}

    raCol, decCol, degrees, ditherStacker, ditherMeta = radecCols(None, colmap, None)
    extraInfoLabel = combineInfoLabels(extraInfoLabel, ditherMeta)
    # Set up basic all and per filter sql constraints.
    filterlist, colors, orders, sqls, info_label = filterList(
        all=True, extraSql=extraSql, extraInfoLabel=extraInfoLabel
    )

    if slicer is not None:
        skyslicer = slicer
    else:
        skyslicer = slicers.HealpixSlicer(
            nside=64, lonCol=raCol, latCol=decCol, latLonDeg=degrees
        )

    # Summarize values over all and per filter.
    slicer = slicers.UniSlicer()
    for f in filterlist:
        for m in standardAngleMetrics(value, replace_colname=valueName):
            displayDict["caption"] = "%s for %s." % (m.name, info_label[f])
            displayDict["order"] = orders[f]
            bundle = mb.MetricBundle(
                m,
                slicer,
                sqls[f],
                info_label=info_label[f],
                displayDict=displayDict,
            )
            bundleList.append(bundle)

    # Histogram values over all and per filter.
    for f in filterlist:
        displayDict["caption"] = "Histogram of %s" % (value)
        if valueName != value:
            displayDict["caption"] += " (%s)" % (valueName)
        displayDict["caption"] += " for %s." % (info_label[f])
        displayDict["order"] = orders[f]
        m = metrics.CountMetric(value, metricName="%s Histogram" % (valueName))
        slicer = slicers.OneDSlicer(sliceColName=value)
        bundle = mb.MetricBundle(
            m,
            slicer,
            sqls[f],
            info_label=info_label[f],
            displayDict=displayDict,
        )
        bundleList.append(bundle)

    # Make maps of min/median/max for all and per filter, per RA/Dec, with standard summary stats.
    mList = []
    mList.append(
        metrics.MeanAngleMetric(value, metricName="AngleMean %s" % (valueName))
    )
    mList.append(
        metrics.FullRangeAngleMetric(value, metricName="AngleRange %s" % (valueName))
    )
    mList.append(metrics.RmsAngleMetric(value, metricName="AngleRms %s" % (valueName)))
    plotDict = {"percentileClip": 98}
    slicer = skyslicer
    for f in filterlist:
        for m in mList:
            displayDict["caption"] = "Map of %s" % m.name
            if valueName != value:
                displayDict["caption"] += " (%s)" % value
            displayDict["caption"] += " for %s." % info_label[f]
            displayDict["order"] = orders[f]
            bundle = mb.MetricBundle(
                m,
                slicer,
                sqls[f],
                info_label=info_label[f],
                plotDict=plotDict,
                displayDict=displayDict,
                summaryMetrics=standardSummary(),
            )
            bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def allMetadata(
    colmap=None, runName="opsim", extraSql=None, extraInfoLabel=None, slicer=None
):
    """Generate a large set of metrics about the metadata of each visit -
    distributions of airmass, normalized airmass, seeing, sky brightness, single visit depth,
    hour angle, distance to the moon, and solar elongation.
    The exact metadata which is analyzed is set by the colmap['metadataList'] value.

    Parameters
    ----------
    colmap : `dict` or None, optional
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : `str`, optional
        The name of the simulated survey. Default is "opsim".
    extraSql : `str`, optional
        Sql constraint (such as WFD only). Default is None.
    extraInfoLabel : `str`, optional
        Metadata to identify the sql constraint (such as WFD). Default is None.
    slicer : `rubin_sim.maf.slicer.BaseSlicer` or None, optional
        Optionally use something other than an nside=64 healpix slicer.

    Returns
    -------
    metricBundleDict
    """

    if colmap is None:
        colmap = ColMapDict("opsimV4")

    bdict = {}

    for valueName in colmap["metadataList"]:
        if valueName in colmap:
            value = colmap[valueName]
        else:
            value = valueName
        mdict = metadataBasics(
            value,
            colmap=colmap,
            runName=runName,
            valueName=valueName,
            extraSql=extraSql,
            extraInfoLabel=extraInfoLabel,
            slicer=slicer,
        )
        bdict.update(mdict)
    for valueName in colmap["metadataAngleList"]:
        if valueName in colmap:
            value = colmap[valueName]
        else:
            value = valueName
        mdict = metadataBasicsAngle(
            value,
            colmap=colmap,
            runName=runName,
            valueName=valueName,
            extraSql=extraSql,
            extraInfoLabel=extraInfoLabel,
            slicer=slicer,
        )
        bdict.update(mdict)
    return bdict


def metadataMaps(
    value,
    colmap=None,
    runName="opsim",
    valueName=None,
    groupName=None,
    extraSql=None,
    extraInfoLabel=None,
    slicer=None,
):
    """Calculate 25/50/75 percentile values on maps across sky for a single metadata value.

    Parameters
    ----------
    value : `str`
        The column name for the quantity to evaluate. (column name in the database or created by a stacker).
    colmap : `dict` or None, optional
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : `str`, optional
        The name of the simulated survey. Default is "opsim".
    valueName : `str`, optional
        The name of the value to be reported in the resultsDb and added to the metric.
        This is intended to help standardize metric comparison between sim versions.
        value = name as it is in the database (seeingFwhmGeom, etc).
        valueName = name to be recorded ('seeingGeom', etc.).  Default is None, which will match 'value'.
    groupName : `str`, optional
        The group name for this quantity in the displayDict. Default is the same as 'valueName', capitalized.
    extraSql : `str`, optional
        Additional constraint to add to any sql constraints (e.g. 'propId=1' or 'fieldID=522').
        Default None, for no additional constraints.
    extraInfoLabel : `str`, optional
        Additional info_label to add before any below (i.e. "WFD").  Default is None.
    slicer : `rubin_sim.maf.slicer.BaseSlicer` or None, optional
        Optionally use something other than an nside=64 HealpixSlicer

    Returns
    -------
    metricBundleDict
    """
    if colmap is None:
        colmap = ColMapDict("opsimV4")
    bundleList = []

    if valueName is None:
        valueName = value

    if groupName is None:
        groupName = valueName.capitalize()
        subgroup = extraInfoLabel
    else:
        groupName = groupName.capitalize()
        subgroup = valueName.capitalize()

    if subgroup is None:
        subgroup = "All visits"

    displayDict = {"group": groupName, "subgroup": subgroup}

    raCol, decCol, degrees, ditherStacker, ditherMeta = radecCols(None, colmap, None)
    extraInfoLabel = combineInfoLabels(extraInfoLabel, ditherMeta)
    # Set up basic all and per filter sql constraints.
    filterlist, colors, orders, sqls, info_label = filterList(
        all=True, extraSql=extraSql, extraInfoLabel=extraInfoLabel
    )

    # Hack to make HA work, but really I need to account for any stackers/colmaps.
    if value == "HA":
        stackerList = [
            stackers.HourAngleStacker(
                lstCol=colmap["lst"], raCol=raCol, latLonDeg=degrees
            )
        ]
    elif value == "normairmass":
        stackerList = [stackers.NormAirmassStacker(degrees=degrees)]
    else:
        stackerList = None

    # Make maps of 25/median/75 for all and per filter, per RA/Dec, with standard summary stats.
    mList = []
    mList.append(
        metrics.PercentileMetric(
            value, percentile=25, metricName="25thPercentile %s" % (valueName)
        )
    )
    mList.append(metrics.MedianMetric(value, metricName="Median %s" % (valueName)))
    mList.append(
        metrics.PercentileMetric(
            value, percentile=75, metricName="75thPercentile %s" % (valueName)
        )
    )
    if slicer is None:
        slicer = slicers.HealpixSlicer(
            nside=64, latCol=decCol, lonCol=raCol, latLonDeg=degrees
        )

    for f in filterlist:
        for m in mList:
            displayDict["caption"] = "Map of %s" % m.name
            if valueName != value:
                displayDict["caption"] += " (%s)" % value
            displayDict["caption"] += " for %s." % info_label[f]
            displayDict["order"] = orders[f]
            bundle = mb.MetricBundle(
                m,
                slicer,
                sqls[f],
                stackerList=stackerList,
                info_label=info_label[f],
                displayDict=displayDict,
                summaryMetrics=standardSummary(),
            )
            bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)

    return mb.makeBundlesDictFromList(bundleList)


def firstYearMetadata(
    colmap=None, runName="opsim", extraSql=None, extraInfoLabel=None, slicer=None
):
    """Measure the distribution of some basic metadata in the first year of operations -
    distributions of airmass, seeing, sky brightness, single visit depth.

    Parameters
    ----------
    colmap : `dict` or None, optional
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : `str`, optional
        The name of the simulated survey. Default is "opsim".
    extraSql : `str`, optional
        Sql constraint (such as WFD only). Default is None.
    extraInfoLabel : `str`, optional
        Metadata to identify the sql constraint (such as WFD). Default is None.
    slicer : `rubin_sim.maf.slicer.BaseSlicer` or None, optional
        Optionally use something other than an nside=64 healpix slicer.

    Returns
    -------
    metricBundleDict
    """

    if colmap is None:
        colmap = ColMapDict("opsimV4")

    bdict = {}

    firstYr = "night < 365.5"
    if extraSql is not None:
        extraSql = f"({firstYr}) and ({extraSql})"
    else:
        extraSql = firstYr
    extraInfoLabel = combineInfoLabels("Yr 1", extraInfoLabel)

    subset = ["airmass", "seeingEff", "seeingGeom", "skyBrightness", "fiveSigmaDepth"]
    for valueName in subset:
        if valueName in colmap:
            value = colmap[valueName]
        else:
            value = valueName
        mdict = metadataBasics(
            value,
            colmap=colmap,
            runName=runName,
            valueName=valueName,
            extraSql=extraSql,
            extraInfoLabel=extraInfoLabel,
            slicer=slicer,
        )
        bdict.update(mdict)

    return bdict
