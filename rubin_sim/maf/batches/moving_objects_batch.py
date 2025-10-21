__all__ = (
    "ss_population_defaults",
    "quick_discovery_batch",
    "discovery_batch",
    "run_completeness_summary",
    "plot_completeness",
    "characterization_inner_batch",
    "characterization_outer_batch",
    "run_fraction_summary",
    "plot_fractions",
    "plot_single",
    "plot_activity",
    "read_and_combine",
    "combine_subsets",
)

import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

import rubin_sim.maf.metric_bundles as mb
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.plots as plots
import rubin_sim.maf.stackers as stackers
from rubin_sim.maf.metric_bundles import MoMetricBundle

from .col_map_dict import col_map_dict
from .common import (
    fraction_population_at_threshold,
    summary_completeness_at_time,
    summary_completeness_over_h,
)


def ss_population_defaults(objtype):
    """Provide useful default ranges for H,
    based on objtype of population type.
    """
    defaults = {}
    defaults["Vatira"] = {
        "h_range": [16, 28, 0.2],
        "h_mark": 22,
        "magtype": "asteroid",
        "char": "inner",
    }
    defaults["PHA"] = {
        "h_range": [16, 28, 0.2],
        "h_mark": 22,
        "magtype": "asteroid",
        "char": "inner",
    }
    defaults["NEO"] = {
        "h_range": [16, 28, 0.2],
        "h_mark": 22,
        "magtype": "asteroid",
        "char": "inner",
    }
    defaults["MBA"] = {
        "h_range": [16, 26, 0.2],
        "h_mark": 20,
        "magtype": "asteroid",
        "char": "inner",
    }
    defaults["MBC"] = {
        "h_range": [16, 26, 0.2],
        "h_mark": 20,
        "magtype": "asteroid",
        "char": "inner",
    }
    defaults["Trojan"] = {
        "h_range": [14, 22, 0.2],
        "h_mark": 18,
        "magtype": "asteroid",
        "char": "inner",
    }
    defaults["TNO"] = {
        "h_range": [4, 12, 0.2],
        "h_mark": 8,
        "magtype": "asteroid",
        "char": "outer",
    }
    defaults["SDO"] = {
        "h_range": [4, 12, 0.2],
        "h_mark": 8,
        "magtype": "asteroid",
        "char": "outer",
    }
    defaults["LPC"] = {
        "h_range": [6, 22, 0.5],
        "h_mark": 13,
        "magtype": "comet_oort",
        "char": "outer",
    }
    defaults["SPC"] = {
        "h_range": [4, 20, 0.5],
        "h_mark": 8,
        "magtype": "comet_short",
        "char": "outer",
    }
    defaults["generic"] = {
        "h_range": [4, 28, 0.5],
        "h_mark": 10,
        "magtype": "asteroid",
        "char": "inner",
    }
    # Some objtypes may be provided by other names
    if objtype.upper().startswith("GRANVIK"):
        objtype = "NEO"
    if objtype.upper().startswith("L7"):
        objtype = "TNO"
    if objtype.upper().startswith("OCC"):
        objtype = "LPC"

    if objtype not in defaults:
        print(
            f"## Could not find {objtype} in default keys ({defaults.keys()}). \n"
            f"## Using generic default values instead."
        )
        objtype = "generic"

    return defaults[objtype]


def quick_discovery_batch(
    slicer,
    colmap=None,
    run_name="run_name",
    detection_losses="detection",
    objtype="",
    albedo=None,
    h_mark=None,
    np_reduce=np.mean,
    constraint_info_label="",
    constraint=None,
    magtype="asteroid",
):
    """A subset of discovery metrics, using only the default discovery
    criteria of 3 pairs in 15 or 30 nights.
    """
    if colmap is None:
        colmap = col_map_dict()
    bundleList = []

    basicPlotDict = {
        "albedo": albedo,
        "h_mark": h_mark,
        "np_reduce": np_reduce,
        "nxbins": 200,
        "nybins": 200,
    }
    plot_funcs = [plots.MetricVsH()]
    display_dict = {"group": f"{objtype}", "subgroup": "Discovery"}

    if constraint_info_label == "" and constraint is not None:
        constraint_info_label = constraint.replace("filter", "").replace("==", "").replace("  ", " ")
    info_label = objtype + " " + constraint_info_label
    info_label = info_label.rstrip(" ")

    if detection_losses not in ("detection", "trailing"):
        raise ValueError("Please choose detection or trailing as options for detection_losses.")
    if detection_losses == "trailing":
        magStacker = stackers.MoMagStacker(loss_col="dmag_trail", magtype=magtype)
        detection_losses = " trailing loss"
    else:
        magStacker = stackers.MoMagStacker(loss_col="dmag_detect", magtype=magtype)
        detection_losses = " detection loss"

    # Set up a dictionary to pass to each metric for the column names.
    colkwargs = {
        "mjd_col": colmap["mjd"],
        "seeing_col": colmap["seeingGeom"],
        "exp_time_col": colmap["exptime"],
        "m5_col": colmap["fiveSigmaDepth"],
        "night_col": colmap["night"],
        "filter_col": colmap["filter"],
    }

    def _setup_child_metrics(parentMetric):
        childMetrics = {}
        childMetrics["Time"] = metrics.DiscoveryTimeMetric(parentMetric, **colkwargs)
        childMetrics["N_Chances"] = metrics.DiscoveryNChancesMetric(parentMetric, **colkwargs)
        # Could expand to add N_chances per year, but not really necessary.
        return childMetrics

    def _configure_child_bundles(parentBundle):
        dispDict = {
            "group": f"{objtype}",
            "subgroup": "Completeness Over Time",
            "caption": "Time of discovery of objects",
            "order": 0,
        }
        parentBundle.child_bundles["Time"].set_display_dict(dispDict)
        dispDict = {
            "group": f"{objtype}",
            "subgroup": "N Chances",
            "caption": "Number of chances for discovery of objects",
            "order": 0,
        }
        parentBundle.child_bundles["N_Chances"].set_display_dict(dispDict)
        return

    t_min = 5.0 / 60.0 / 24.0
    t_max = 90.0 / 60.0 / 24.0

    # 3 pairs in 15
    md = info_label + " 3 pairs in 15 nights" + detection_losses
    # Set up plot dict.
    plotDict = {"title": "%s: %s" % (run_name, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(
        n_obs_per_night=2,
        t_min=t_min,
        t_max=t_max,
        n_nights_per_window=3,
        t_window=15,
        **colkwargs,
    )
    childMetrics = _setup_child_metrics(metric)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=[magStacker],
        run_name=run_name,
        info_label=md,
        child_metrics=childMetrics,
        plot_dict=plotDict,
        plot_funcs=plot_funcs,
        display_dict=display_dict,
    )
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # 3 pairs in 30
    md = info_label + " 3 pairs in 30 nights" + detection_losses
    plotDict = {"title": "%s: %s" % (run_name, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(
        n_obs_per_night=2,
        t_min=t_min,
        t_max=t_max,
        n_nights_per_window=3,
        t_window=30,
        **colkwargs,
    )
    childMetrics = _setup_child_metrics(metric)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=[magStacker],
        run_name=run_name,
        info_label=md,
        child_metrics=childMetrics,
        plot_dict=plotDict,
        plot_funcs=plot_funcs,
        display_dict=display_dict,
    )
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # Set the run_name for all bundles and return the bundleDict.
    for b in bundleList:
        b.set_run_name(run_name)
    return mb.make_bundles_dict_from_list(bundleList)


def discovery_batch(
    slicer,
    colmap=None,
    run_name="run_name",
    detection_losses="detection",
    objtype="",
    albedo=None,
    h_mark=None,
    np_reduce=np.mean,
    constraint_info_label="",
    constraint=None,
    magtype="asteroid",
):
    """A comprehensive set of discovery metrics, using a wide range
    of discovery criteria.
    """
    if colmap is None:
        colmap = col_map_dict()
    bundleList = []

    basicPlotDict = {
        "albedo": albedo,
        "h_mark": h_mark,
        "np_reduce": np_reduce,
        "nxbins": 200,
        "nybins": 200,
    }
    plot_funcs = [plots.MetricVsH()]
    display_dict = {"group": f"{objtype}", "subgroup": "Discovery"}

    if constraint_info_label == "" and constraint is not None:
        constraint_info_label = constraint.replace("filter", "").replace("==", "").replace("  ", " ")
    info_label = objtype + " " + constraint_info_label
    info_label = info_label.rstrip(" ")

    if detection_losses not in ("detection", "trailing"):
        raise ValueError("Please choose detection or trailing as options for detection_losses.")
    if detection_losses == "trailing":
        # These are the SNR-losses only.
        magStacker = stackers.MoMagStacker(loss_col="dmag_trail", magtype=magtype)
        detection_losses = " trailing loss"
    else:
        # SNR losses, plus additional loss due to detecting with stellar PSF.
        magStacker = stackers.MoMagStacker(loss_col="dmag_detect", magtype=magtype)
        detection_losses = " detection loss"

    # Set up a dictionary to pass to each metric for the column names.
    colkwargs = {
        "mjd_col": colmap["mjd"],
        "seeing_col": colmap["seeingGeom"],
        "exp_time_col": colmap["exptime"],
        "m5_col": colmap["fiveSigmaDepth"],
        "night_col": colmap["night"],
        "filter_col": colmap["filter"],
    }

    def _setup_child_metrics(parentMetric):
        childMetrics = {}
        childMetrics["Time"] = metrics.DiscoveryTimeMetric(parentMetric, **colkwargs)
        childMetrics["N_Chances"] = metrics.DiscoveryNChancesMetric(parentMetric, **colkwargs)
        # Could expand to add N_chances per year, but not really necessary.
        return childMetrics

    def _configure_child_bundles(parentBundle):
        dispDict = {
            "group": f"{objtype}",
            "subgroup": "Completeness Over Time",
            "caption": "Time of discovery of objects",
            "order": 0,
        }
        parentBundle.child_bundles["Time"].set_display_dict(dispDict)
        dispDict = {
            "group": f"{objtype}",
            "subgroup": "N Chances",
            "caption": "Number of chances for discovery of objects",
            "order": 0,
        }
        parentBundle.child_bundles["N_Chances"].set_display_dict(dispDict)

    t_min = 5.0 / 60.0 / 24.0
    t_max = 90.0 / 60.0 / 24.0

    # 3 pairs in 15 and 3 pairs in 30 done in 'quickDiscoveryBatch' (with vis).

    # 4 pairs in 20
    md = info_label + " 4 pairs in 20 nights" + detection_losses
    plotDict = {"title": "%s: %s" % (run_name, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(
        n_obs_per_night=2,
        t_min=t_min,
        t_max=t_max,
        n_nights_per_window=4,
        t_window=20,
        **colkwargs,
    )
    childMetrics = _setup_child_metrics(metric)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=[magStacker],
        run_name=run_name,
        info_label=md,
        child_metrics=childMetrics,
        plot_dict=plotDict,
        plot_funcs=plot_funcs,
        display_dict=display_dict,
    )
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # 3 triplets in 30
    md = info_label + " 3 triplets in 30 nights" + detection_losses
    plotDict = {"title": "%s: %s" % (run_name, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(
        n_obs_per_night=3,
        t_min=t_min,
        t_max=120.0 / 60.0 / 24.0,
        n_nights_per_window=3,
        t_window=30,
        **colkwargs,
    )
    childMetrics = _setup_child_metrics(metric)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=[magStacker],
        run_name=run_name,
        info_label=md,
        child_metrics=childMetrics,
        plot_dict=plotDict,
        plot_funcs=plot_funcs,
        display_dict=display_dict,
    )
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # 1 quad
    md = info_label + " 1 quad in 1 night" + detection_losses
    plotDict = {"title": "%s: %s" % (run_name, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(
        n_obs_per_night=4,
        t_min=t_min,
        t_max=150.0 / 60.0 / 24.0,
        n_nights_per_window=1,
        t_window=2,
        **colkwargs,
    )
    childMetrics = _setup_child_metrics(metric)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=[magStacker],
        run_name=run_name,
        info_label=md,
        child_metrics=childMetrics,
        plot_dict=plotDict,
        plot_funcs=plot_funcs,
        display_dict=display_dict,
    )
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # Play with SNR.
    # First standard SNR / probabilistic visibility (SNR~5)
    # 3 pairs in 15
    md = info_label + " 3 pairs in 15 nights SNR=5" + detection_losses
    # Set up plot dict.
    plotDict = {"title": "%s: %s" % (run_name, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(
        n_obs_per_night=2,
        t_min=t_min,
        t_max=t_max,
        n_nights_per_window=3,
        t_window=15,
        snr_limit=5,
        **colkwargs,
    )
    childMetrics = _setup_child_metrics(metric)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=[magStacker],
        run_name=run_name,
        info_label=md,
        child_metrics=childMetrics,
        plot_dict=plotDict,
        plot_funcs=plot_funcs,
        display_dict=display_dict,
    )
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # 3 pairs in 15, SNR=4.
    md = info_label + " 3 pairs in 15 nights SNR=4" + detection_losses
    plotDict = {"title": "%s: %s" % (run_name, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(
        n_obs_per_night=2,
        t_min=t_min,
        t_max=t_max,
        n_nights_per_window=3,
        t_window=15,
        snr_limit=4,
        **colkwargs,
    )
    childMetrics = _setup_child_metrics(metric)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=[magStacker],
        run_name=run_name,
        info_label=md,
        child_metrics=childMetrics,
        plot_dict=plotDict,
        plot_funcs=plot_funcs,
        display_dict=display_dict,
    )
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # 3 pairs in 15, SNR=3
    md = info_label + " 3 pairs in 15 nights SNR=3" + detection_losses
    plotDict = {"title": "%s: %s" % (run_name, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(
        n_obs_per_night=2,
        t_min=t_min,
        t_max=t_max,
        n_nights_per_window=3,
        t_window=15,
        snr_limit=3,
        **colkwargs,
    )
    childMetrics = _setup_child_metrics(metric)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=[magStacker],
        run_name=run_name,
        info_label=md,
        child_metrics=childMetrics,
        plot_dict=plotDict,
        plot_funcs=plot_funcs,
        display_dict=display_dict,
    )
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # SNR = 0
    # 3 pairs in 15, SNR=0
    md = info_label + " 3 pairs in 15 nights SNR=0" + detection_losses
    plotDict = {"title": "%s: %s" % (run_name, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(
        n_obs_per_night=2,
        t_min=t_min,
        t_max=t_max,
        n_nights_per_window=3,
        t_window=15,
        snr_limit=0,
        **colkwargs,
    )
    childMetrics = _setup_child_metrics(metric)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=[magStacker],
        run_name=run_name,
        info_label=md,
        child_metrics=childMetrics,
        plot_dict=plotDict,
        plot_funcs=plot_funcs,
        display_dict=display_dict,
    )
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # Play with weird strategies.
    # Single detection.
    md = info_label + " Single detection" + detection_losses
    plotDict = {"title": "%s: %s" % (run_name, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(
        n_obs_per_night=1,
        t_min=t_min,
        t_max=t_max,
        n_nights_per_window=1,
        t_window=5,
        **colkwargs,
    )
    childMetrics = _setup_child_metrics(metric)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=[magStacker],
        run_name=run_name,
        info_label=md,
        child_metrics=childMetrics,
        plot_dict=plotDict,
        plot_funcs=plot_funcs,
        display_dict=display_dict,
    )
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # Single pair of detections.
    md = info_label + " Single pair" + detection_losses
    plotDict = {"title": "%s: %s" % (run_name, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(
        n_obs_per_night=2,
        t_min=t_min,
        t_max=t_max,
        n_nights_per_window=1,
        t_window=5,
        **colkwargs,
    )
    childMetrics = _setup_child_metrics(metric)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=[magStacker],
        run_name=run_name,
        info_label=md,
        child_metrics=childMetrics,
        plot_dict=plotDict,
        plot_funcs=plot_funcs,
        display_dict=display_dict,
    )
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # High velocity discovery.
    md = info_label + " High velocity pair" + detection_losses
    plotDict = {"title": "%s: %s" % (run_name, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.HighVelocityNightsMetric(psf_factor=2.0, n_obs_per_night=2, **colkwargs)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=[magStacker],
        run_name=run_name,
        info_label=md,
        plot_dict=plotDict,
        plot_funcs=plot_funcs,
        display_dict=display_dict,
    )
    bundleList.append(bundle)

    # "magic" detection - 6 in 60 days.
    md = info_label + " 6 detections in 60 nights" + detection_losses
    plotDict = {"title": "%s: %s" % (run_name, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.MagicDiscoveryMetric(n_obs=6, t_window=60, **colkwargs)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=[magStacker],
        run_name=run_name,
        info_label=md,
        plot_dict=plotDict,
        plot_funcs=plot_funcs,
        display_dict=display_dict,
    )
    bundleList.append(bundle)

    # Set the run_name for all bundles and return the bundleDict.
    for b in bundleList:
        b.set_run_name(run_name)
    return mb.make_bundles_dict_from_list(bundleList)


def run_completeness_summary(bdict, h_mark, times, out_dir, results_db):
    """Calculate completeness and create completeness bundles from all
    N_Chances and Time (child) metrics of the (discovery) bundles in bdict,
    and write completeness at h_mark to results_db, save bundle to disk.

    This should be done after combining any sub-sets of the metric results.

    Parameters
    ----------
    bdict : `dict` of `maf.MetricBundle`
        Dict containing ~rubin_sim.maf.MoMetricBundles,
        including bundles we're expecting to contain completeness.
    h_mark : `float`
        h_mark value to add to completeness plotting dict.
        If not defined (None), then the h_mark from the plotdict from the
        metric bundle will be used if available.
        If None and h_mark not in plot_dict, then median of h_range values
        will be used.
    times : `np.ndarray`
        The times at which to calculate completeness (over time).
    out_dir : `str`
        Output directory to save completeness bundles to disk.
    results_db : `maf.ResultsDb`
        Results database to save information about completeness bundle.

    Returns
    -------
    metricDict : `dict` of `maf.MetricBundles`
        A dictionary of the new completeness bundles. Keys match original keys,
        with additions of "[Differential,Cumulative]Completeness@Time"
        and "[Differential,Cumulative]Completeness" to distinguish new entries.
    """
    # Add completeness bundles and write completeness at h_mark to results_db.
    completeness = {}

    def _compbundles(b, bundle, h_mark, results_db):
        # Find h_mark if not set (this may be different for different bundles).
        if h_mark is None and "h_mark" in bundle.plot_dict:
            h_mark = bundle.plot_dict["h_mark"]
        if h_mark is None:
            h_mark = np.median(bundle.slicer.slice_points["H"])
        # Set up the summary metrics.
        summaryTimeMetrics = summary_completeness_at_time(times, h_val=h_mark, h_index=0.33)
        summaryTimeMetrics2 = summary_completeness_at_time(times, h_val=h_mark - 2, h_index=0.33)
        summaryHMetrics = summary_completeness_over_h(requiredChances=1, Hindex=0.33)
        comp = {}
        # Bundle = single metric bundle.
        # Add differential and cumulative completeness.
        if "Time" in bundle.metric.name:
            for metric in summaryTimeMetrics:
                newkey = b + " " + metric.name
                comp[newkey] = mb.make_completeness_bundle(bundle, metric, h_mark=None, results_db=results_db)
                comp[newkey].plot_dict["times"] = times
                comp[newkey].plot_dict["h_val"] = metric.hval
            for metric in summaryTimeMetrics2:
                newkey = b + " " + metric.name
                comp[newkey] = mb.make_completeness_bundle(bundle, metric, h_mark=None, results_db=results_db)
                comp[newkey].plot_dict["times"] = times
                comp[newkey].plot_dict["h_val"] = metric.hval
        elif "NChances" in bundle.metric.name or "N_Chances" in bundle.metric.name:
            for metric in summaryHMetrics:
                newkey = b + " " + metric.name
                comp[newkey] = mb.make_completeness_bundle(
                    bundle, metric, h_mark=h_mark, results_db=results_db
                )
        elif "MagicDiscovery" in bundle.metric.name:
            for metric in summaryHMetrics:
                newkey = b + " " + metric.name
                comp[newkey] = mb.make_completeness_bundle(
                    bundle, metric, h_mark=h_mark, results_db=results_db
                )
        elif "HighVelocity" in bundle.metric.name:
            for metric in summaryHMetrics:
                newkey = b + " " + metric.name
                comp[newkey] = mb.make_completeness_bundle(
                    bundle, metric, h_mark=h_mark, results_db=results_db
                )
        return comp

    # Generate the completeness bundles for the various discovery metrics.
    for b, bundle in bdict.items():
        if "Discovery" in bundle.metric.name:
            completeness.update(_compbundles(b, bundle, h_mark, results_db))
        if "MagicDiscovery" in bundle.metric.name:
            completeness.update(_compbundles(b, bundle, h_mark, results_db))
        if "HighVelocity" in bundle.metric.name:
            completeness.update(_compbundles(b, bundle, h_mark, results_db))
    # Write the completeness bundles to disk, so we can re-read them later.
    # (also set the display dict properties, for the results_db output).
    for b, bundle in completeness.items():
        bundle.display_dict["subgroup"] = "Completeness"
        bundle.write(out_dir=out_dir, results_db=results_db)

    # Calculate total number of objects - currently for NEOs and PHAs only
    for b, bundle in completeness.items():
        if "DifferentialCompleteness" in b and "@Time" not in b:
            if "NEO" in bundle.info_label:
                nobj_metrics = [
                    metrics.TotalNumberSSO(h_mark=22, dndh_func=metrics.neo_dndh_granvik),
                    metrics.TotalNumberSSO(h_mark=25, dndh_func=metrics.neo_dndh_granvik),
                ]
                bundle.set_summary_metrics(nobj_metrics)
                bundle.compute_summary_stats(results_db)
            if "PHA" in bundle.info_label:
                nobj_metrics = [metrics.TotalNumberSSO(h_mark=22, dndh_func=metrics.pha_dndh_granvik)]
                bundle.set_summary_metrics(nobj_metrics)
                bundle.compute_summary_stats(results_db)
    return completeness


def plot_completeness(
    bdictCompleteness,
    figroot=None,
    run_name=None,
    results_db=None,
    out_dir=".",
    fig_format="pdf",
):
    """Plot a minor subset of the completeness results."""
    # Separate some subsets to plot together -
    # first just the simple 15 and 30 night detection loss metrics.
    keys = [
        "3_pairs_in_30_nights_detection_loss",
        "3_pairs_in_15_nights_detection_loss",
    ]
    plotTimes = {}
    plotComp = {}
    plotDiff = {}
    for k in bdictCompleteness:
        for key in keys:
            if key in k:
                if "Time" in k:
                    if "Cumulative" in k:
                        plotTimes[k] = bdictCompleteness[k]
                elif "Chances" in k:
                    if "Differential" in k:
                        plotDiff[k] = bdictCompleteness[k]
                    elif "Cumulative" in k:
                        plotComp[k] = bdictCompleteness[k]

    # Add plot dictionaries to code 30 nights red, 15 nights blue,
    # differentials dotted.
    def _codePlot(key):
        plotDict = {}
        if "Differential" in k:
            plotDict["linestyle"] = ":"
        else:
            plotDict["linestyle"] = "-"
        if "30_nights" in k:
            plotDict["color"] = "r"
        if "15_nights" in k:
            plotDict["color"] = "b"
        return plotDict

    # Apply color-coding.
    for k, b in plotTimes.items():
        b.set_plot_dict(_codePlot(k))
    for k, b in plotDiff.items():
        b.set_plot_dict(_codePlot(k))
    for k, b in plotComp.items():
        b.set_plot_dict(_codePlot(k))

    first = bdictCompleteness[list(bdictCompleteness.keys())[0]]
    if run_name is None:
        run_name = first.run_name
    if figroot is None:
        figroot = run_name
    display_dict = deepcopy(first.display_dict)

    # Plot completeness as a function of time.
    # Make custom plot, then save it with PlotHandler.
    fig = plt.figure(figsize=(8, 6))
    for k in plotTimes:
        plt.plot(
            plotTimes[k].plot_dict["times"],
            plotTimes[k].metric_values[0, :],
            label=plotTimes[k].plot_dict["label"] + " @H=%.2f" % plotTimes[k].plot_dict["h_val"],
        )
    plt.legend()
    plt.xlabel("Time (MJD)")
    plt.ylabel("Completeness")
    plt.grid(True, alpha=0.3)
    # Make a PlotHandler to deal with savings/results_db, etc.
    ph = plots.PlotHandler(fig_format=fig_format, results_db=results_db, out_dir=out_dir)
    display_dict["subgroup"] = "Completeness over time"
    display_dict["caption"] = "Completeness over time, for H values indicated in legend."
    ph.save_fig(
        fig,
        f"{figroot}_CompletenessOverTime",
        "Combo",
        "CompletenessOverTime",
        "MoObjSlicer",
        figroot,
        None,
        None,
        display_dict=display_dict,
    )

    plt.savefig(
        os.path.join(out_dir, f"{figroot}_CompletenessOverTime.{fig_format}"),
        format=fig_format,
    )

    # Plot cumulative completeness.
    ph = plots.PlotHandler(fig_format=fig_format, results_db=results_db, out_dir=out_dir)
    ph.set_metric_bundles(plotComp)
    plotDict = {"ylabel": "Completeness", "figsize": (8, 6), "albedo": 0.14}
    ph.plot(
        plot_func=plots.MetricVsH(),
        plot_dicts=plotDict,
        outfile_root=figroot + "_CumulativeCompleteness",
    )

    # Plot differential completeness.
    ph = plots.PlotHandler(fig_format=fig_format, results_db=results_db, out_dir=out_dir)
    ph.set_metric_bundles(plotDiff)
    plotDict = {"ylabel": "Completeness", "figsize": (8, 6)}
    ph.plot(
        plot_func=plots.MetricVsH(),
        plot_dicts=plotDict,
        outfile_root=figroot + "_DifferentialCompleteness",
    )

    # And add the rest of the completeness calculations.
    allComp = []
    for k in bdictCompleteness:
        if "DiscoveryNChances" in k:
            if "Cumulative" in k:
                allComp.append(bdictCompleteness[k])
        if "Magic" in k:
            if "Cumulative" in k:
                allComp.append(bdictCompleteness[k])
    ph = plots.PlotHandler(fig_format=fig_format, results_db=results_db, out_dir=out_dir)
    ph.set_metric_bundles(allComp)
    plotDict = {
        "ylabel": "Completeness",
        "figsize": (8, 6),
        "legendloc": (1.01, 0.1),
        "color": None,
    }
    display_dict["subgroup"] = "Completeness all criteria"
    display_dict["caption"] = "Plotting all of the cumulative completeness curves together."
    ph.plot(
        plot_func=plots.MetricVsH(),
        plot_dicts=plotDict,
        display_dict=display_dict,
        outfile_root=figroot + "_Many_CumulativeCompleteness",
    )


def characterization_inner_batch(
    slicer,
    colmap=None,
    run_name="run_name",
    objtype="",
    magtype="asteroid",
    albedo=None,
    h_mark=None,
    constraint_info_label="",
    constraint=None,
    npReduce=np.mean,
    windows=None,
    bins=None,
):
    """Characterization metrics for inner solar system objects."""
    if colmap is None:
        colmap = col_map_dict()
    bundleList = []

    # Set up a dictionary to pass to each metric for the column names.
    colkwargs = {
        "mjd_col": colmap["mjd"],
        "seeing_col": colmap["seeingGeom"],
        "exp_time_col": colmap["exptime"],
        "m5_col": colmap["fiveSigmaDepth"],
        "night_col": colmap["night"],
        "filter_col": colmap["filter"],
    }

    basicPlotDict = {
        "albedo": albedo,
        "h_mark": h_mark,
        "np_reduce": npReduce,
        "nxbins": 200,
        "nybins": 200,
    }
    plot_funcs = [plots.MetricVsH()]

    if constraint_info_label == "" and constraint is not None:
        constraint_info_label = constraint.replace("filter", "").replace("==", "").replace("  ", " ")
    info_label = objtype + " " + constraint_info_label
    info_label = info_label.rstrip(" ")

    display_dict = {"group": f"{objtype}"}

    # Stackers
    magStacker = stackers.MoMagStacker(loss_col="dmag_detect", magtype=magtype)
    eclStacker = stackers.EclStacker()
    stackerList = [magStacker, eclStacker]

    # Windows are the different 'length of activity'
    if windows is None:
        windows = np.arange(10, 200, 30.0)
    # Bins are the different 'anomaly variations' of activity
    if bins is None:
        bins = np.arange(5, 185, 20.0)

    # Number of observations.
    md = info_label
    display_dict["subgroup"] = "N Obs"
    plotDict = {
        "ylabel": "Number of observations (#)",
        "title": "%s: Number of observations %s" % (run_name, md),
    }
    plotDict.update(basicPlotDict)
    metric = metrics.NObsMetric(**colkwargs)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=stackerList,
        run_name=run_name,
        info_label=md,
        plot_dict=plotDict,
        plot_funcs=plot_funcs,
        display_dict=display_dict,
    )
    bundleList.append(bundle)

    # Observational arc.
    md = info_label
    display_dict["subgroup"] = "Obs Arc"
    plotDict = {
        "ylabel": "Observational Arc (days)",
        "title": "%s: Observational Arc Length %s" % (run_name, md),
    }
    plotDict.update(basicPlotDict)
    metric = metrics.ObsArcMetric(**colkwargs)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=stackerList,
        run_name=run_name,
        info_label=md,
        plot_dict=plotDict,
        plot_funcs=plot_funcs,
        display_dict=display_dict,
    )
    bundleList.append(bundle)

    # Activity detection.
    display_dict["subgroup"] = "Activity"
    for w in windows:
        md = info_label + " activity lasting %.0f days" % w
        plotDict = {
            "title": "%s: Chances of detecting %s" % (run_name, md),
            "ylabel": "Probability of detection per %.0f day window" % w,
        }
        metric_name = "Chances of detecting activity lasting %.0f days" % w
        metric = metrics.ActivityOverTimeMetric(w, metric_name=metric_name, **colkwargs)
        bundle = MoMetricBundle(
            metric,
            slicer,
            constraint,
            stacker_list=stackerList,
            run_name=run_name,
            info_label=info_label,
            plot_dict=plotDict,
            plot_funcs=plot_funcs,
            display_dict=display_dict,
        )
        bundleList.append(bundle)

    for b in bins:
        md = info_label + " activity covering %.0f deg" % (b)
        plotDict = {
            "title": "%s: Chances of detecting %s" % (run_name, md),
            "ylabel": "Probability of detection per %.0f deg window" % b,
        }
        metric_name = "Chances of detecting activity covering %.0f deg" % (b)
        metric = metrics.ActivityOverPeriodMetric(b, metric_name=metric_name, **colkwargs)
        bundle = MoMetricBundle(
            metric,
            slicer,
            constraint,
            stacker_list=stackerList,
            run_name=run_name,
            info_label=info_label,
            plot_dict=plotDict,
            plot_funcs=plot_funcs,
            display_dict=display_dict,
        )
        bundleList.append(bundle)

    # Lightcurve inversion.
    md = info_label
    display_dict["subgroup"] = "Color/Inversion"
    plotDict = {
        "y_min": 0,
        "y_max": 1,
        "ylabel": "Fraction of objects",
        "title": "%s: Fraction with potential lightcurve inversion %s" % (run_name, md),
    }
    plotDict.update(basicPlotDict)
    metric = metrics.LightcurveInversionAsteroidMetric(**colkwargs)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=stackerList,
        run_name=run_name,
        info_label=md,
        plot_dict=plotDict,
        plot_funcs=plot_funcs,
        display_dict=display_dict,
    )
    bundleList.append(bundle)

    # Color determination.
    md = info_label
    plotDict = {
        "y_min": 0,
        "y_max": 1,
        "ylabel": "Fraction of objects",
        "title": "%s: Fraction of population with colors in X filters %s" % (run_name, md),
    }
    plotDict.update(basicPlotDict)
    metric = metrics.ColorAsteroidMetric(**colkwargs)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=stackerList,
        run_name=run_name,
        info_label=md,
        plot_dict=plotDict,
        plot_funcs=plot_funcs,
        display_dict=display_dict,
    )
    bundleList.append(bundle)

    # Set the run_name for all bundles and return the bundleDict.
    for b in bundleList:
        b.set_run_name(run_name)
    return mb.make_bundles_dict_from_list(bundleList)


def characterization_outer_batch(
    slicer,
    colmap=None,
    run_name="run_name",
    objtype="",
    magtype="asteroid",
    albedo=None,
    h_mark=None,
    constraint_info_label="",
    constraint=None,
    npReduce=np.mean,
    windows=None,
    bins=None,
):
    """Characterization metrics for outer solar system objects."""
    if colmap is None:
        colmap = col_map_dict()
    bundleList = []

    # Set up a dictionary to pass to each metric for the column names.
    colkwargs = {
        "mjd_col": colmap["mjd"],
        "seeing_col": colmap["seeingGeom"],
        "exp_time_col": colmap["exptime"],
        "m5_col": colmap["fiveSigmaDepth"],
        "night_col": colmap["night"],
        "filter_col": colmap["filter"],
    }

    basicPlotDict = {
        "albedo": albedo,
        "h_mark": h_mark,
        "np_reduce": npReduce,
        "nxbins": 200,
        "nybins": 200,
    }
    plot_funcs = [plots.MetricVsH()]

    if constraint_info_label == "" and constraint is not None:
        constraint_info_label = constraint.replace("filter", "").replace("==", "").replace("  ", " ")
    info_label = objtype + " " + constraint_info_label
    info_label = info_label.rstrip(" ")

    display_dict = {"group": f"{objtype}"}

    # Stackers
    magStacker = stackers.MoMagStacker(loss_col="dmag_detect", magtype=magtype)
    eclStacker = stackers.EclStacker()
    stackerList = [magStacker, eclStacker]

    # Windows are the different 'length of activity'
    if windows is None:
        windows = np.arange(10, 200, 30.0)
    # Bins are the different 'anomaly variations' of activity
    if bins is None:
        bins = np.arange(5, 185, 20.0)

    # Number of observations.
    md = info_label
    display_dict["subgroup"] = "N Obs"
    plotDict = {
        "ylabel": "Number of observations (#)",
        "title": "%s: Number of observations %s" % (run_name, md),
    }
    plotDict.update(basicPlotDict)
    metric = metrics.NObsMetric(**colkwargs)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=stackerList,
        run_name=run_name,
        info_label=md,
        plot_dict=plotDict,
        plot_funcs=plot_funcs,
        display_dict=display_dict,
    )
    bundleList.append(bundle)

    # Observational arc.
    md = info_label
    display_dict["subgroup"] = "Obs Arc"
    plotDict = {
        "ylabel": "Observational Arc (days)",
        "title": "%s: Observational Arc Length %s" % (run_name, md),
    }
    plotDict.update(basicPlotDict)
    metric = metrics.ObsArcMetric(**colkwargs)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=stackerList,
        run_name=run_name,
        info_label=md,
        plot_dict=plotDict,
        plot_funcs=plot_funcs,
        display_dict=display_dict,
    )
    bundleList.append(bundle)

    # Activity detection.
    display_dict["subgroup"] = "Activity"
    for w in windows:
        md = info_label + " activity lasting %.0f days" % w
        plotDict = {
            "title": "%s: Chances of detecting %s" % (run_name, md),
            "ylabel": "Probability of detection per %.0f day window" % w,
        }
        metric_name = "Chances of detecting activity lasting %.0f days" % w
        metric = metrics.ActivityOverTimeMetric(w, metric_name=metric_name, **colkwargs)
        bundle = MoMetricBundle(
            metric,
            slicer,
            constraint,
            stacker_list=stackerList,
            run_name=run_name,
            info_label=info_label,
            plot_dict=plotDict,
            plot_funcs=plot_funcs,
            display_dict=display_dict,
        )
        bundleList.append(bundle)

    for b in bins:
        md = info_label + " activity covering %.0f deg" % (b)
        plotDict = {
            "title": "%s: Chances of detecting %s" % (run_name, md),
            "ylabel": "Probability of detection per %.2f deg window" % b,
        }
        metric_name = "Chances of detecting activity covering %.0f deg" % (b)
        metric = metrics.ActivityOverPeriodMetric(b, metric_name=metric_name, **colkwargs)
        bundle = MoMetricBundle(
            metric,
            slicer,
            constraint,
            stacker_list=stackerList,
            run_name=run_name,
            info_label=info_label,
            plot_dict=plotDict,
            plot_funcs=plot_funcs,
            display_dict=display_dict,
        )
        bundleList.append(bundle)

    # Color determination.
    md = info_label
    display_dict["subgroup"] = "Color/Inversion"
    plotDict = {
        "y_min": 0,
        "y_max": 1,
        "ylabel": "Fraction of objects",
        "title": "%s: Fraction of population with colors in X filters %s" % (run_name, md),
    }
    plotDict.update(basicPlotDict)
    metric = metrics.LightcurveColorOuterMetric(**colkwargs)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=stackerList,
        run_name=run_name,
        info_label=md,
        plot_dict=plotDict,
        plot_funcs=plot_funcs,
        display_dict=display_dict,
    )
    bundleList.append(bundle)

    # Set the run_name for all bundles and return the bundleDict.
    for b in bundleList:
        b.set_run_name(run_name)
    return mb.make_bundles_dict_from_list(bundleList)


def run_fraction_summary(bdict, h_mark, out_dir, results_db):
    """
    Calculate fractional completeness of the population for
    color and lightcurve metrics.

    This should be done after combining any sub-sets of the metric results.

    Parameters
    ----------
    bdict : `dict` of `maf.MoMetricBundle`
        Dict containing bundles contianing lightcurve/color evaluations.
    h_mark : `float`
        h_mark value to add to completeness plotting dict.
        If defined, this value is used.
        If None, but h_mark in plot_dict for metric, then this value (-2) is
        used. If h_mark not in plotdict, then use the median h_range value-2.
    times : `np.ndarray`
        The times at which to calculate completeness (over time).
    out_dir : `str`
        Output directory to save completeness bundles to disk.
    results_db : `maf.ResultsDb`
        Results database to save information about completeness bundle.

    Returns
    -------
    metricDict : `dict` of `maf.MetricBundle`
        Dictionary of the metric bundles for the fractional evaluation
        of the population.
    """
    fractions = {}

    # Look for metrics from asteroid or outer solar system
    # color/lightcurve metrics.
    inversionSummary = fraction_population_at_threshold([1], ["Lightcurve Inversion"])
    asteroidColorSummary = fraction_population_at_threshold(
        [4, 3, 2, 1],
        ["6 of ugrizy", "5 of grizy", "4 of grizy", "2 of g, r or i, z or y"],
    )
    asteroidSummaryMetrics = {
        "LightcurveInversionAsteroid": inversionSummary,
        "LightcurveInversion_Asteroid": inversionSummary,
        "ColorAsteroid": asteroidColorSummary,
        "Color_Asteroid": asteroidColorSummary,
    }

    outerColorSummary = fraction_population_at_threshold(
        [6, 5, 4, 3, 2, 1],
        ["6 filters", "5 filters", "4 filters", "3 filters", "2 filters", "1 filters"],
    )
    outerSummaryMetrics = {
        "LightcurveColorOuter": outerColorSummary,
        "lightcurveColor_Outer": outerColorSummary,
    }

    for b, bundle in bdict.items():
        # Find h_mark if not set (this may be different for different bundles).
        if h_mark is None and "h_mark" in bundle.plot_dict:
            h_mark = bundle.plot_dict["h_mark"] - 2
        if h_mark is None:
            h_mark = np.median(bundle.slicer.slice_points["H"]) - 2
        # Make sure we didn't push h_mark outside the range of H values
        if h_mark < bundle.slicer.slice_points["H"].min():
            h_mark = bundle.slicer.slice_points["H"].min()
        for k in asteroidSummaryMetrics:
            if k in b:
                for summary_metric in asteroidSummaryMetrics[k]:
                    newkey = b + " " + summary_metric.name
                    fractions[newkey] = mb.make_completeness_bundle(
                        bundle, summary_metric, h_mark=h_mark, results_db=results_db
                    )
        for k in outerSummaryMetrics:
            if k in b:
                for summary_metric in outerSummaryMetrics[k]:
                    newkey = b + " " + summary_metric.name
                    fractions[newkey] = mb.make_completeness_bundle(
                        bundle, summary_metric, h_mark=h_mark, results_db=results_db
                    )

    # Write fractional populations bundles to disk, so we can re-read later.
    for b, bundle in fractions.items():
        bundle.write(out_dir=out_dir, results_db=results_db)
    return fractions


def plot_fractions(
    bdictFractions,
    figroot=None,
    run_name=None,
    results_db=None,
    out_dir=".",
    fig_format="pdf",
):
    # Set colors for the fractions.
    for b in bdictFractions.values():
        k = b.metric.name
        if "6" in k:
            b.plot_dict["color"] = "b"
        if "5" in k:
            b.plot_dict["color"] = "cyan"
        if "4" in k:
            b.plot_dict["color"] = "orange"
        if "2" in k:
            b.plot_dict["color"] = "r"
        if "1" in k:
            b.plot_dict["color"] = "magenta"
        if "Lightcurve Inversion" in k:
            b.plot_dict["color"] = "k"
            b.plot_dict["linestyle"] = ":"
            b.plot_dict["linewidth"] = 3

    first = bdictFractions[list(bdictFractions.keys())[0]]
    if run_name is None:
        run_name = first.run_name
    if figroot is None:
        figroot = run_name
    display_dict = deepcopy(first.display_dict)
    display_dict["subgroup"] = "Characterization Fraction"

    ph = plots.PlotHandler(fig_format=fig_format, results_db=results_db, out_dir=out_dir)
    ph.set_metric_bundles(bdictFractions)
    ph.joint_metric_names = "Fraction of population for colors or lightcurve inversion"
    plotDict = {"ylabel": "Fraction of population", "figsize": (8, 6)}
    ph.plot(
        plot_func=plots.MetricVsH(),
        plot_dicts=plotDict,
        display_dict=display_dict,
        outfile_root=figroot + "_characterization",
    )


def plot_single(bundle, results_db=None, out_dir=".", fig_format="pdf"):
    """Plot 5%/25%/50%/75%/95% iles for a metric value."""
    pDict = {
        "95%ile": {
            "color": "k",
            "linestyle": "--",
            "label": "95th %ile",
            "np_reduce": lambda x, axis: np.percentile(x, 95, axis=axis),
        },
        "75%ile": {
            "color": "magenta",
            "linestyle": ":",
            "label": "75th %ile",
            "np_reduce": lambda x, axis: np.percentile(x, 75, axis=axis),
        },
        "Median": {
            "color": "b",
            "linestyle": "-",
            "label": "Median",
            "np_reduce": lambda x, axis: np.median(x, axis=axis),
        },
        "Mean": {
            "color": "g",
            "linestyle": "--",
            "label": "Mean",
            "np_reduce": np.mean,
        },
        "25%ile": {
            "color": "magenta",
            "linestyle": ":",
            "label": "25th %ile",
            "np_reduce": lambda x, axis: np.percentile(x, 25, axis=axis),
        },
        "5%ile": {
            "color": "k",
            "linestyle": "--",
            "label": "5th %ile",
            "np_reduce": lambda x, axis: np.percentile(x, 5, axis=axis),
        },
    }
    ph = plots.PlotHandler(fig_format=fig_format, results_db=results_db, out_dir=out_dir)
    plot_bundles = []
    plot_dicts = []
    for percentile in pDict:
        plot_bundles.append(bundle)
        plot_dicts.append(pDict[percentile])
    plot_dicts[0].update({"figsize": (8, 6), "legendloc": "upper right", "y_min": 0})
    # Remove the h_mark line because these plots get complicated already.
    for r in plot_dicts:
        r["h_mark"] = None
    ph.set_metric_bundles(plot_bundles)
    ph.plot(
        plot_func=plots.MetricVsH(),
        plot_dicts=plot_dicts,
        display_dict=bundle.display_dict,
    )


def plot_not_found(nChances, h_mark):
    pass


def plot_activity(bdict, figroot=None, results_db=None, out_dir=".", fig_format="pdf"):
    activity_deg = {}
    activity_days = {}
    for k in bdict:
        if "Chances_of_detecting_activity" in k:
            if "deg" in k:
                activity_deg[k] = bdict[k]
            if "days" in k:
                activity_days[k] = bdict[k]

    first = bdict[list(bdict.keys())[0]]
    if figroot is None:
        figroot = first.run_name
    display_dict = deepcopy(first.display_dict)

    if len(activity_days) > 0:
        # Plot (mean) likelihood of detection of activity over X days
        ph = plots.PlotHandler(fig_format=fig_format, results_db=results_db, out_dir=out_dir)
        ph.set_metric_bundles(activity_days)
        ph.joint_metric_names = "Chances of detecting activity lasting X days"
        plot_dict = {"ylabel": "Mean likelihood of detection", "figsize": (8, 6)}
        ph.plot(
            plot_func=plots.MetricVsH(),
            plot_dicts=plot_dict,
            display_dict=display_dict,
            outfile_root=figroot + "_activityDays",
        )
    if len(activity_deg) > 0:
        # Plot mean likelihood of detection of activity over X amount of orbit
        ph = plots.PlotHandler(fig_format=fig_format, results_db=results_db, out_dir=out_dir)
        ph.set_metric_bundles(activity_deg)
        ph.joint_metric_names = "Chances of detecting activity covering X deg"
        plot_dict = {"ylabel": "Mean likelihood of detection", "figsize": (8, 6)}
        ph.plot(
            plot_func=plots.MetricVsH(),
            plot_dicts=plot_dict,
            display_dict=display_dict,
            outfile_root=figroot + "_activityDeg",
        )


def read_and_combine(orbitRoot, baseDir, splits, metricfile):
    """Read and combine the metric results from split locations,
    returning a single bundle.

    This will read the files from
    baseDir/orbitRoot_[split]/metricfile
    where split = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], etc.
    (the subsets the original orbit file was split into).

    Parameters
    ----------
    orbitRoot : `str`
        The root of the orbit file - l7_5k, mbas_5k, etc.
    baseDir: `str`
        The root directory containing the subset directories. (e.g. '.' often)
    splits:` np.ndarray` or `list` of `ints`
        The integers describing the split directories
        (e.g. [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    metricfile: `str`
        The metric filename.

    Returns
    -------
    metric_bundle : `~rubin_sim.maf.MoMetricBundle`
        A single metric bundle containing the combined data from the subsets.

    Note that this won't work for particularly complex metric values,
    such as the parent Discovery metrics. However, you can read and combine
    their child metrics, as for these we can propagate the data masks.
    """
    subsets = {}
    for i in splits:
        subsets[i] = mb.create_empty_mo_metric_bundle()
        ddir = os.path.join(baseDir, f"{orbitRoot}_{i}")
        subsets[i].read(os.path.join(ddir, metricfile))
    bundle = combine_subsets(subsets)
    return bundle


def combine_subsets(mbSubsets):
    # Combine the data from the subset metric bundles.
    # The first bundle will be used a template for the slicer.
    if isinstance(mbSubsets, dict):
        first = mbSubsets[list(mbSubsets.keys())[0]]
    else:
        first = mbSubsets[0]
        subsetdict = {}
        for i, b in enumerate(mbSubsets):
            subsetdict[i] = b
        mbSubsets = subsetdict
    joint = mb.create_empty_mo_metric_bundle()
    # Check if they're the same slicer.
    slicer = deepcopy(first.slicer)
    for i in mbSubsets:
        if np.any(slicer.slice_points["H"] != mbSubsets[i].slicer.slice_points["H"]):
            if np.any(slicer.slice_points["orbits"] != mbSubsets[i].slicer.slice_points["orbits"]):
                raise ValueError("Bundle %s has a different slicer than the first bundle" % (i))
    # Join metric values.
    joint.slicer = slicer
    joint.metric = first.metric
    # Don't just use the slicer shape to define the metric_values,
    # because of CompletenessBundles.
    metric_values = np.zeros(first.metric_values.shape, float)
    metric_values_mask = np.zeros(first.metric_values.shape, bool)
    for i in mbSubsets:
        metric_values += mbSubsets[i].metric_values.filled(0)
        metric_values_mask = np.where(metric_values_mask & mbSubsets[i].metric_values.mask, True, False)
    joint.metricValues = ma.MaskedArray(data=metric_values, mask=metric_values_mask, fill_value=0)
    joint.info_label = first.info_label
    joint.run_name = first.run_name
    joint.file_root = first.file_root.replace(".npz", "")
    joint.plotDict = first.plotDict
    joint.display_dict = first.display_dict
    return joint
