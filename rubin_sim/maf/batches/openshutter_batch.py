"""Evaluate the open shutter fraction.
"""

__all__ = ("openshutterFractions",)

import rubin_sim.maf.metric_bundles as mb
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers

from .col_map_dict import col_map_dict
from .common import standard_summary


def openshutterFractions(colmap=None, runName="opsim", extraSql=None, extraInfoLabel=None):
    """Evaluate open shutter fraction over whole survey and per night.

    Parameters
    ----------
    colmap : `dict`, optional
        A dictionary with a mapping of column names.
    runName : `str`, optional
        The name of the simulated survey.
    extraSql : `str`, optional
        Additional constraint to add to any sql constraints (e.g. 'night<365')
    extraInfoLabel : `str`, optional
        Additional info_label to add before any below (i.e. "WFD").

    Returns
    -------
    metric_bundleDict : `dict` of `maf.MetricBundle`
    """
    if colmap is None:
        colmap = col_map_dict()
    bundleList = []

    group = "Open Shutter Fraction"

    subgroup = "All visits"
    if extraInfoLabel is not None:
        subgroup = extraInfoLabel + " " + subgroup.lower()
    elif extraSql is not None and extraInfoLabel is None:
        subgroup = subgroup + " " + extraSql

    # Open Shutter fraction over whole survey.
    displayDict = {"group": group, "subgroup": subgroup, "order": 0}
    displayDict["caption"] = "Total open shutter fraction over %s. " % subgroup.lower()
    displayDict["caption"] += "Does not include downtime due to weather."
    metric = metrics.OpenShutterFractionMetric(
        slew_time_col=colmap["slewtime"],
        exp_time_col=colmap["exptime"],
        visit_time_col=colmap["visittime"],
    )
    slicer = slicers.UniSlicer()
    bundle = mb.MetricBundle(metric, slicer, extraSql, info_label=subgroup, display_dict=displayDict)
    bundleList.append(bundle)
    # Count the number of nights on-sky in the survey.
    displayDict["caption"] = "Number of nights on the sky during the survey, %s." % subgroup.lower()
    metric = metrics.CountUniqueMetric(colmap["night"])
    slicer = slicers.UniSlicer()
    bundle = mb.MetricBundle(metric, slicer, extraSql, info_label=subgroup, display_dict=displayDict)
    bundleList.append(bundle)
    # Count the number of nights total in the survey
    # (start to finish of observations).
    displayDict["caption"] = "Number of nights from start to finish of survey, %s." % subgroup.lower()
    metric = metrics.FullRangeMetric(colmap["night"])
    slicer = slicers.UniSlicer()
    bundle = mb.MetricBundle(metric, slicer, extraSql, info_label=subgroup, display_dict=displayDict)
    bundleList.append(bundle)

    # Open shutter fraction per night.
    subgroup = "Per night"
    if extraInfoLabel is not None:
        subgroup = extraInfoLabel + " " + subgroup.lower()
    elif extraSql is not None and extraInfoLabel is None:
        subgroup = subgroup + " " + extraSql
    displayDict = {"group": group, "subgroup": subgroup, "order": 0}
    displayDict["caption"] = "Open shutter fraction %s." % (subgroup.lower())
    displayDict["caption"] += (
        " This compares on-sky image time against on-sky time + slews + filter "
        "changes + readout, but does not include downtime due to weather."
    )
    metric = metrics.OpenShutterFractionMetric(
        slew_time_col=colmap["slewtime"],
        exp_time_col=colmap["exptime"],
        visit_time_col=colmap["visittime"],
    )
    slicer = slicers.OneDSlicer(slice_col_name=colmap["night"], bin_size=1)
    bundle = mb.MetricBundle(
        metric,
        slicer,
        extraSql,
        info_label=subgroup,
        summary_metrics=standard_summary(),
        display_dict=displayDict,
    )
    bundleList.append(bundle)

    # Set the run_name for all bundles and return the bundleDict.
    for b in bundleList:
        b.set_run_name(runName)
    return mb.make_bundles_dict_from_list(bundleList)
