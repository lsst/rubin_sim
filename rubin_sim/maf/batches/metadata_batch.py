"""Some basic physical quantity metrics.
"""

__all__ = (
    "metadataBasics",
    "metadataBasicsAngle",
    "allMetadata",
    "metadataMaps",
    "firstYearMetadata",
)

import rubin_sim.maf.metric_bundles as mb
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.stackers as stackers

from .col_map_dict import col_map_dict
from .common import (
    combine_info_labels,
    extended_metrics,
    filter_list,
    standard_angle_metrics,
    standard_summary,
)


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
    """Calculate basic metrics on visit metadata 'value'
    (e.g. airmass, normalized airmass, seeing..).
    Calculates this around the sky (HealpixSlicer),
    makes histograms of all visits (OneDSlicer),
    and calculates statistics on all visits (UniSlicer) for the quantity,
    in all visits and per filter.

    Currently have a hack for HA & normairmass.

    Parameters
    ----------
    value : `str`
        The column name for the quantity to evaluate.
        (column name in the database or created by a stacker).
    colmap : `dict` or None, optional
        A dictionary with a mapping of column names.
    runName : `str`, optional
        The name of the simulated survey.
    valueName : `str`, optional
        The name of the value to be reported in the results_db and
        added to the metric.
        This is intended to help standardize metric comparison between
        sim versions.
        value = name as it is in the database (seeingFwhmGeom, etc).
        valueName = name to be recorded ('seeingGeom', etc.).
    groupName : `str`, optional
        The group name for this quantity in the display_dict.
        None will default to the same as 'valueName', capitalized.
    extraSql : `str`, optional
        Additional constraint to add to any sql constraints.
    extraInfoLabel : `str`, optional
        Additional info_labels to add before any below (i.e. "WFD").
    slicer : `rubin_sim.maf.slicers.BaseSlicer`, optional
        Optionally use a different slicer than an nside=64 healpix slicer.

    Returns
    -------
    metric_bundleDict : `dict` of `maf.MetricBundle`
    """
    if colmap is None:
        colmap = col_map_dict()
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

    raCol = colmap["ra"]
    decCol = colmap["dec"]
    degrees = colmap["raDecDeg"]

    # Set up basic all and per filter sql constraints.
    filterlist, colors, orders, sqls, info_label = filter_list(
        all=True, extra_sql=extraSql, extra_info_label=extraInfoLabel
    )

    if slicer is not None:
        skyslicer = slicer
    else:
        skyslicer = slicers.HealpixSlicer(nside=64, lon_col=raCol, lat_col=decCol, lat_lon_deg=degrees)

    # Hack to make HA work, but really need to account for any stackers
    if value == "HA":
        stackerList = [stackers.HourAngleStacker(lst_col=colmap["lst"], ra_col=raCol, degrees=degrees)]
    elif value == "normairmass":
        stackerList = [stackers.NormAirmassStacker(degrees=degrees)]
    else:
        stackerList = None

    # Summarize values over all and per filter
    # (min/mean/median/max/percentiles/outliers/rms).
    slicer = slicers.UniSlicer()
    for f in filterlist:
        for m in extended_metrics(value, replace_colname=valueName):
            displayDict["caption"] = "%s for %s." % (m.name, info_label[f])
            displayDict["order"] = orders[f]
            bundle = mb.MetricBundle(
                m,
                slicer,
                sqls[f],
                stacker_list=stackerList,
                info_label=info_label[f],
                display_dict=displayDict,
            )
            bundleList.append(bundle)

    # Histogram values over all and per filter.
    for f in filterlist:
        displayDict["caption"] = "Histogram of %s" % (value)
        if valueName != value:
            displayDict["caption"] += " (%s)" % (valueName)
        displayDict["caption"] += " for %s." % (info_label[f])
        displayDict["order"] = orders[f]
        m = metrics.CountMetric(value, metric_name="%s Histogram" % (valueName))
        slicer = slicers.OneDSlicer(slice_col_name=value)
        bundle = mb.MetricBundle(
            m,
            slicer,
            sqls[f],
            stacker_list=stackerList,
            info_label=info_label[f],
            display_dict=displayDict,
        )
        bundleList.append(bundle)

    # Make maps of min/median/max for all and per filter,
    # per RA/Dec, with standard summary stats.
    plotDict = {"percentile_clip": 98}
    mList = []
    mList.append(metrics.MinMetric(value, metric_name="Min %s" % (valueName)))
    mList.append(metrics.MedianMetric(value, metric_name="Median %s" % (valueName)))
    mList.append(metrics.MaxMetric(value, metric_name="Max %s" % (valueName)))
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
                stacker_list=stackerList,
                info_label=info_label[f],
                plot_dict=plotDict,
                display_dict=displayDict,
                summary_metrics=standard_summary(),
            )
            bundleList.append(bundle)

    # Set the run_name for all bundles and return the bundleDict.
    for b in bundleList:
        b.set_run_name(runName)
    return mb.make_bundles_dict_from_list(bundleList)


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
    """Calculate basic metrics on visit metadata 'value',
    where value is a wrap-around angle.

    Calculates extended standard metrics (with unislicer) on the quantity
    (all visits and per filter),
    makes histogram of the value (all visits and per filter),


    Parameters
    ----------
    value : `str`
        The column name for the quantity to evaluate.
        (column name in the database or created by a stacker).
    colmap : `dict` or None, optional
        A dictionary with a mapping of column names.
    runName : `str`, optional
        The name of the simulated survey.
    valueName : `str`, optional
        The name of the value to be reported in the results_db
        and added to the metric.
        This is intended to help standardize metric comparison
        between sim versions.
        value = name as it is in the database (seeingFwhmGeom, etc).
        valueName = name to be recorded ('seeingGeom', etc.).
    groupName : `str`, optional
        The group name for this quantity in the display_dict.
        None will default to the same as 'valueName', capitalized.
    extraSql : `str`, optional
        Additional constraint to add to any sql constraints.
    extraInfoLabel : `str`, optional
        Additional info_label to add before any below (i.e. "WFD").
    slicer : `rubin_sim.maf.slicer.BaseSlicer` or None, optional

    Returns
    -------
    metric_bundleDict : `dict` of `maf.MetricBundle`
    """
    if colmap is None:
        colmap = col_map_dict()
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

    raCol = colmap["ra"]
    decCol = colmap["dec"]
    degrees = colmap["raDecDeg"]
    # Set up basic all and per filter sql constraints.
    filterlist, colors, orders, sqls, info_label = filter_list(
        all=True, extra_sql=extraSql, extra_info_label=extraInfoLabel
    )

    if slicer is not None:
        skyslicer = slicer
    else:
        skyslicer = slicers.HealpixSlicer(nside=64, lon_col=raCol, lat_col=decCol, lat_lon_deg=degrees)

    # Summarize values over all and per filter.
    slicer = slicers.UniSlicer()
    for f in filterlist:
        for m in standard_angle_metrics(value, replace_colname=valueName):
            displayDict["caption"] = "%s for %s." % (m.name, info_label[f])
            displayDict["order"] = orders[f]
            bundle = mb.MetricBundle(
                m,
                slicer,
                sqls[f],
                info_label=info_label[f],
                display_dict=displayDict,
            )
            bundleList.append(bundle)

    # Histogram values over all and per filter.
    for f in filterlist:
        displayDict["caption"] = "Histogram of %s" % (value)
        if valueName != value:
            displayDict["caption"] += " (%s)" % (valueName)
        displayDict["caption"] += " for %s." % (info_label[f])
        displayDict["order"] = orders[f]
        m = metrics.CountMetric(value, metric_name="%s Histogram" % (valueName))
        slicer = slicers.OneDSlicer(slice_col_name=value)
        bundle = mb.MetricBundle(
            m,
            slicer,
            sqls[f],
            info_label=info_label[f],
            display_dict=displayDict,
        )
        bundleList.append(bundle)

    # Make maps of min/median/max for all and per filter,
    # per RA/Dec, with standard summary stats.
    mList = []
    mList.append(metrics.MeanAngleMetric(value, metric_name="AngleMean %s" % (valueName)))
    mList.append(metrics.FullRangeAngleMetric(value, metric_name="AngleRange %s" % (valueName)))
    mList.append(metrics.RmsAngleMetric(value, metric_name="AngleRms %s" % (valueName)))
    plotDict = {"percentile_clip": 98}
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
                plot_dict=plotDict,
                display_dict=displayDict,
                summary_metrics=standard_summary(),
            )
            bundleList.append(bundle)

    # Set the run_name for all bundles and return the bundleDict.
    for b in bundleList:
        b.set_run_name(runName)
    return mb.make_bundles_dict_from_list(bundleList)


def allMetadata(colmap=None, runName="opsim", extraSql=None, extraInfoLabel=None, slicer=None):
    """Generate a large set of metrics about the metadata of each visit -
    distributions of airmass, normalized airmass, seeing, sky brightness,
    single visit depth, hour angle, distance to the moon, and solar elongation.
    The exact metadata which is analyzed is set by the colmap['metadataList'].

    Parameters
    ----------
    colmap : `dict` or None, optional
        A dictionary with a mapping of column names.
    runName : `str`, optional
        The name of the simulated survey.
    extraSql : `str`, optional
        Sql constraint (such as WFD only).
    extraInfoLabel : `str`, optional
        Metadata to identify the sql constraint (such as WFD).
    slicer : `rubin_sim.maf.slicer.BaseSlicer` or None, optional
        Optionally use something other than an nside=64 healpix slicer.

    Returns
    -------
    metric_bundleDict : `dict` of `maf.MetricBundle`
    """

    if colmap is None:
        colmap = col_map_dict()

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
    """Calculate 25/50/75 percentile values on maps across sky
    for a single metadata value.

    Parameters
    ----------
    value : `str`
        The column name for the quantity to evaluate.
        (column name in the database or created by a stacker).
    colmap : `dict` or None, optional
        A dictionary with a mapping of column names.
    runName : `str`, optional
        The name of the simulated survey.
    valueName : `str`, optional
        The name of the value to be reported in the results_db
        and added to the metric.
        This is intended to help standardize metric comparison
        between sim versions.
        value = name as it is in the database (seeingFwhmGeom, etc).
        valueName = name to be recorded ('seeingGeom', etc.).
    groupName : `str`, optional
        The group name for this quantity in the display_dict.
    extraSql : `str`, optional
        Additional constraint to add to any sql constraints.
    extraInfoLabel : `str`, optional
        Additional info_label to add before any below (i.e. "WFD").
    slicer : `rubin_sim.maf.slicer.BaseSlicer` or None, optional
        Optionally use something other than an nside=64 HealpixSlicer

    Returns
    -------
    metric_bundleDict : `dict` of `maf.MetricBundle`
    """
    if colmap is None:
        colmap = col_map_dict()
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

    ra_col = colmap["ra"]
    dec_col = colmap["dec"]
    degrees = colmap["raDecDeg"]
    # Set up basic all and per filter sql constraints.
    filterlist, colors, orders, sqls, info_label = filter_list(
        all=True, extra_sql=extraSql, extra_info_label=extraInfoLabel
    )

    # Hack to make HA work, but really need to account for any stackers
    if value == "HA":
        stackerList = [stackers.HourAngleStacker(lst_col=colmap["lst"], ra_col=ra_col, degrees=degrees)]
    elif value == "normairmass":
        stackerList = [stackers.NormAirmassStacker(degrees=degrees)]
    else:
        stackerList = None

    # Make maps of 25/median/75 for all and per filter,
    # per RA/Dec, with standard summary stats.
    mList = []
    mList.append(
        metrics.PercentileMetric(value, percentile=25, metric_name="25thPercentile %s" % (valueName))
    )
    mList.append(metrics.MedianMetric(value, metric_name="Median %s" % (valueName)))
    mList.append(
        metrics.PercentileMetric(value, percentile=75, metric_name="75thPercentile %s" % (valueName))
    )
    if slicer is None:
        slicer = slicers.HealpixSlicer(nside=64, lat_col=dec_col, lon_col=ra_col, lat_lon_deg=degrees)

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
                stacker_list=stackerList,
                info_label=info_label[f],
                display_dict=displayDict,
                summary_metrics=standard_summary(),
            )
            bundleList.append(bundle)

    # Set the run_name for all bundles and return the bundleDict.
    for b in bundleList:
        b.set_run_name(runName)

    return mb.make_bundles_dict_from_list(bundleList)


def firstYearMetadata(colmap=None, runName="opsim", extraSql=None, extraInfoLabel=None, slicer=None):
    """Measure the distribution of some basic metadata in the first year
    of operations -
    distributions of airmass, seeing, sky brightness, single visit depth.

    Parameters
    ----------
    colmap : `dict` or None, optional
        A dictionary with a mapping of column names.
    runName : `str`, optional
        The name of the simulated survey.
    extraSql : `str`, optional
        Sql constraint (such as WFD only).
    extraInfoLabel : `str`, optional
        Metadata to identify the sql constraint (such as WFD).
    slicer : `rubin_sim.maf.slicer.BaseSlicer` or None, optional
        Optionally use something other than an nside=64 healpix slicer.

    Returns
    -------
    metric_bundleDict : `dict` of `maf.MetricBundle`
    """

    if colmap is None:
        colmap = col_map_dict()

    bdict = {}

    firstYr = "night < 365.5"
    if extraSql is not None:
        extraSql = f"({firstYr}) and ({extraSql})"
    else:
        extraSql = firstYr
    extraInfoLabel = combine_info_labels("Yr 1", extraInfoLabel)

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
