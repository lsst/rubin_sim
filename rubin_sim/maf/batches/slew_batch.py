"""Sets of slew metrics.
"""

__all__ = ("slewBasics",)

import numpy as np

import rubin_sim.maf.metric_bundles as mb
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers

from .col_map_dict import col_map_dict
from .common import standard_metrics


def slewBasics(colmap=None, run_name="opsim", sql_constraint=None):
    """Generate a simple set of statistics about the slew times and distances.

    Parameters
    ----------
    colmap : `dict` or None, optional
        A dictionary with a mapping of column names.
    runName : `str`, optional
        The name of the simulated survey.
    sqlConstraint : `str` or None, optional
        SQL constraint to add to metrics.

    Returns
    -------
    metric_bundleDict : `dict` of `maf.MetricBundle`
    """

    if colmap is None:
        colmap = col_map_dict()

    bundleList = []

    # Calculate basic stats on slew times. (mean/median/min/max + total).
    slicer = slicers.UniSlicer()

    info_label = "All visits"
    if sql_constraint is not None and len(sql_constraint) > 0:
        info_label = "%s" % (sql_constraint)
    displayDict = {
        "group": "Slew",
        "subgroup": "Slew Basics",
        "order": -1,
        "caption": None,
    }
    # Add total number of slews.
    metric = metrics.CountMetric(colmap["slewtime"], metric_name="Slew Count")
    displayDict["caption"] = "Total number of slews recorded."
    displayDict["order"] += 1
    bundle = mb.MetricBundle(metric, slicer, sql_constraint, info_label=info_label, display_dict=displayDict)
    bundleList.append(bundle)
    for metric in standard_metrics(colmap["slewtime"]):
        displayDict["caption"] = "%s in seconds." % (metric.name)
        displayDict["order"] += 1
        bundle = mb.MetricBundle(
            metric,
            slicer,
            sql_constraint,
            info_label=info_label,
            display_dict=displayDict,
        )
        bundleList.append(bundle)

    # Slew Time histogram.
    slicer = slicers.OneDSlicer(slice_col_name=colmap["slewtime"], bin_size=2)
    metric = metrics.CountMetric(col=colmap["slewtime"], metric_name="Slew Time Histogram")
    info_label = "All visits"
    plotDict = {"log_scale": True, "ylabel": "Count"}
    displayDict["caption"] = "Histogram of slew times (seconds) for all visits."
    displayDict["order"] += 1
    bundle = mb.MetricBundle(
        metric,
        slicer,
        sql_constraint,
        info_label=info_label,
        plot_dict=plotDict,
        display_dict=displayDict,
    )
    bundleList.append(bundle)
    # Zoom in on slew time histogram near 0.
    slicer = slicers.OneDSlicer(slice_col_name=colmap["slewtime"], bin_size=0.2, bin_min=0, bin_max=20)
    metric = metrics.CountMetric(col=colmap["slewtime"], metric_name="Zoom Slew Time Histogram")
    info_label = "All visits"
    plotDict = {"log_scale": True, "ylabel": "Count"}
    displayDict["caption"] = "Histogram of slew times (seconds) for all visits (zoom)."
    displayDict["order"] += 1
    bundle = mb.MetricBundle(
        metric,
        slicer,
        sql_constraint,
        info_label=info_label,
        plot_dict=plotDict,
        display_dict=displayDict,
    )
    bundleList.append(bundle)

    # Slew distance histogram, if available.
    if colmap["slewdist"] is not None:
        bin_size = 2.0
        if not colmap["raDecDeg"]:
            bin_size = np.radians(bin_size)
        slicer = slicers.OneDSlicer(slice_col_name=colmap["slewdist"], bin_size=bin_size)
        metric = metrics.CountMetric(col=colmap["slewdist"], metric_name="Slew Distance Histogram")
        plotDict = {"log_scale": True, "ylabel": "Count"}
        displayDict["caption"] = "Histogram of slew distances (angle) for all visits."
        displayDict["order"] += 1
        bundle = mb.MetricBundle(
            metric,
            slicer,
            sql_constraint,
            info_label=info_label,
            plot_dict=plotDict,
            display_dict=displayDict,
        )
        bundleList.append(bundle)
        # Zoom on slew distance histogram.
        bin_max = 20.0
        if not colmap["raDecDeg"]:
            bin_max = np.radians(bin_max)
        slicer = slicers.OneDSlicer(
            slice_col_name=colmap["slewdist"],
            bin_size=bin_size / 10.0,
            bin_min=0,
            bin_max=bin_max,
        )
        metric = metrics.CountMetric(col=colmap["slewdist"], metric_name="Zoom Slew Distance Histogram")
        plotDict = {"log_scale": True, "ylabel": "Count"}
        displayDict["caption"] = "Histogram of slew distances (angle) for all visits."
        displayDict["order"] += 1
        bundle = mb.MetricBundle(
            metric,
            slicer,
            sql_constraint,
            info_label=info_label,
            plot_dict=plotDict,
            display_dict=displayDict,
        )
        bundleList.append(bundle)

    # Set the run_name for all bundles and return the bundleDict.
    for b in bundleList:
        b.set_run_name(run_name)
    return mb.make_bundles_dict_from_list(bundleList)
