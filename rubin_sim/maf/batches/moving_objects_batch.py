from __future__ import print_function, division
from copy import deepcopy
import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import rubin_sim.maf.metrics as metrics
from rubin_sim.maf.slicers import MoObjSlicer
import rubin_sim.maf.stackers as stackers
import rubin_sim.maf.plots as plots
import rubin_sim.maf.metric_bundles as mb
from rubin_sim.maf.metric_bundles import MoMetricBundle
from .col_map_dict import col_map_dict
from .common import (
    summaryCompletenessAtTime,
    summaryCompletenessOverH,
    fractionPopulationAtThreshold,
)

__all__ = [
    "ss_population_defaults",
    "setupMoSlicer",
    "quickDiscoveryBatch",
    "discoveryBatch",
    "runCompletenessSummary",
    "plotCompleteness",
    "characterizationInnerBatch",
    "characterizationOuterBatch",
    "runFractionSummary",
    "plotFractions",
    "plotSingle",
    "plotActivity",
    "readAndCombine",
    "combineSubsets",
]


def ss_population_defaults(objtype):
    "Provide useful default ranges for H, based on objtype of population type."
    defaults = {}
    defaults["Vatira"] = {
        "Hrange": [16, 28, 0.2],
        "Hmark": 22,
        "magtype": "asteroid",
        "char": "inner",
    }
    defaults["PHA"] = {
        "Hrange": [16, 28, 0.2],
        "Hmark": 22,
        "magtype": "asteroid",
        "char": "inner",
    }
    defaults["NEO"] = {
        "Hrange": [16, 28, 0.2],
        "Hmark": 22,
        "magtype": "asteroid",
        "char": "inner",
    }
    defaults["MBA"] = {
        "Hrange": [16, 26, 0.2],
        "Hmark": 20,
        "magtype": "asteroid",
        "char": "inner",
    }
    defaults["Trojan"] = {
        "Hrange": [14, 22, 0.2],
        "Hmark": 18,
        "magtype": "asteroid",
        "char": "inner",
    }
    defaults["TNO"] = {
        "Hrange": [4, 12, 0.2],
        "Hmark": 8,
        "magtype": "asteroid",
        "char": "outer",
    }
    defaults["SDO"] = {
        "Hrange": [4, 12, 0.2],
        "Hmark": 8,
        "magtype": "asteroid",
        "char": "outer",
    }
    defaults["LPC"] = {
        "Hrange": [6, 22, 0.5],
        "Hmark": 13,
        "magtype": "comet_oort",
        "char": "outer",
    }
    defaults["SPC"] = {
        "Hrange": [4, 20, 0.5],
        "Hmark": 8,
        "magtype": "comet_short",
        "char": "outer",
    }
    defaults["generic"] = {
        "Hrange": [4, 28, 0.5],
        "Hmark": 10,
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


def setupMoSlicer(orbitFile, Hrange, obsFile=None):
    """
    Set up the slicer and read orbitFile and obsFile from disk.

    Parameters
    ----------
    orbitFile : str
        The file containing the orbit information.
    Hrange : numpy.ndarray or None
        The Hrange parameter to pass to slicer.readOrbits
    obsFile : str, optional
        The file containing the observations of each object, optional.
        If not provided (default, None), then the slicer will not be able to 'slice', but can still plot.

    Returns
    -------
    ~rubin_sim.maf.slicer.MoObjSlicer
    """
    # Read the orbit file and set the H values for the slicer.
    slicer = MoObjSlicer(Hrange=Hrange)
    slicer.setupSlicer(orbitFile=orbitFile, obsFile=obsFile)
    return slicer


def quickDiscoveryBatch(
    slicer,
    colmap=None,
    runName="opsim",
    detectionLosses="detection",
    objtype="",
    albedo=None,
    Hmark=None,
    npReduce=np.mean,
    constraintInfoLabel="",
    constraint=None,
    magtype="asteroid",
):
    if colmap is None:
        colmap = col_map_dict("opsimV4")
    bundleList = []

    basicPlotDict = {
        "albedo": albedo,
        "Hmark": Hmark,
        "npReduce": npReduce,
        "nxbins": 200,
        "nybins": 200,
    }
    plotFuncs = [plots.MetricVsH()]
    displayDict = {"group": f"{objtype}", "subgroup": "Discovery"}

    if constraintInfoLabel == "" and constraint is not None:
        constraintInfoLabel = (
            constraint.replace("filter", "").replace("==", "").replace("  ", " ")
        )
    info_label = objtype + " " + constraintInfoLabel
    info_label = info_label.rstrip(" ")

    if detectionLosses not in ("detection", "trailing"):
        raise ValueError(
            "Please choose detection or trailing as options for detectionLosses."
        )
    if detectionLosses == "trailing":
        magStacker = stackers.MoMagStacker(lossCol="dmagTrail", magtype=magtype)
        detectionLosses = " trailing loss"
    else:
        magStacker = stackers.MoMagStacker(lossCol="dmagDetect", magtype=magtype)
        detectionLosses = " detection loss"

    # Set up a dictionary to pass to each metric for the column names.
    colkwargs = {
        "mjdCol": colmap["mjd"],
        "seeingCol": colmap["seeingGeom"],
        "expTimeCol": colmap["exptime"],
        "m5Col": colmap["fiveSigmaDepth"],
        "nightCol": colmap["night"],
        "filterCol": colmap["filter"],
    }

    def _setup_child_metrics(parentMetric):
        childMetrics = {}
        childMetrics["Time"] = metrics.Discovery_TimeMetric(parentMetric, **colkwargs)
        childMetrics["N_Chances"] = metrics.Discovery_N_ChancesMetric(
            parentMetric, **colkwargs
        )
        # Could expand to add N_chances per year, but not really necessary.
        return childMetrics

    def _configure_child_bundles(parentBundle):
        dispDict = {
            "group": f"{objtype}",
            "subgroup": f"Completeness Over Time",
            "caption": "Time of discovery of objects",
            "order": 0,
        }
        parentBundle.child_bundles["Time"].set_display_dict(dispDict)
        dispDict = {
            "group": f"{objtype}",
            "subgroup": f"N Chances",
            "caption": "Number of chances for discovery of objects",
            "order": 0,
        }
        parentBundle.child_bundles["N_Chances"].set_display_dict(dispDict)
        return

    tMin = 5.0 / 60.0 / 24.0
    tMax = 90.0 / 60.0 / 24.0

    # 3 pairs in 15
    md = info_label + " 3 pairs in 15 nights" + detectionLosses
    # Set up plot dict.
    plotDict = {"title": "%s: %s" % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(
        nObsPerNight=2,
        tMin=tMin,
        tMax=tMax,
        nNightsPerWindow=3,
        tWindow=15,
        **colkwargs,
    )
    childMetrics = _setup_child_metrics(metric)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=[magStacker],
        run_name=runName,
        info_label=md,
        child_metrics=childMetrics,
        plot_dict=plotDict,
        plot_funcs=plotFuncs,
        display_dict=displayDict,
    )
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # 3 pairs in 30
    md = info_label + " 3 pairs in 30 nights" + detectionLosses
    plotDict = {"title": "%s: %s" % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(
        nObsPerNight=2,
        tMin=tMin,
        tMax=tMax,
        nNightsPerWindow=3,
        tWindow=30,
        **colkwargs,
    )
    childMetrics = _setup_child_metrics(metric)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=[magStacker],
        run_name=runName,
        info_label=md,
        child_metrics=childMetrics,
        plot_dict=plotDict,
        plot_funcs=plotFuncs,
        display_dict=displayDict,
    )
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.set_run_name(runName)
    return mb.make_bundles_dict_from_list(bundleList)


def discoveryBatch(
    slicer,
    colmap=None,
    runName="opsim",
    detectionLosses="detection",
    objtype="",
    albedo=None,
    Hmark=None,
    npReduce=np.mean,
    constraintInfoLabel="",
    constraint=None,
    magtype="asteroid",
):
    if colmap is None:
        colmap = col_map_dict("opsimV4")
    bundleList = []

    basicPlotDict = {
        "albedo": albedo,
        "Hmark": Hmark,
        "npReduce": npReduce,
        "nxbins": 200,
        "nybins": 200,
    }
    plotFuncs = [plots.MetricVsH()]
    displayDict = {"group": f"{objtype}", "subgroup": "Discovery"}

    if constraintInfoLabel == "" and constraint is not None:
        constraintInfoLabel = (
            constraint.replace("filter", "").replace("==", "").replace("  ", " ")
        )
    info_label = objtype + " " + constraintInfoLabel
    info_label = info_label.rstrip(" ")

    if detectionLosses not in ("detection", "trailing"):
        raise ValueError(
            "Please choose detection or trailing as options for detectionLosses."
        )
    if detectionLosses == "trailing":
        # These are the SNR-losses only.
        magStacker = stackers.MoMagStacker(lossCol="dmagTrail", magtype=magtype)
        detectionLosses = " trailing loss"
    else:
        # This is SNR losses, plus additional loss due to detecting with stellar PSF.
        magStacker = stackers.MoMagStacker(lossCol="dmagDetect", magtype=magtype)
        detectionLosses = " detection loss"

    # Set up a dictionary to pass to each metric for the column names.
    colkwargs = {
        "mjdCol": colmap["mjd"],
        "seeingCol": colmap["seeingGeom"],
        "expTimeCol": colmap["exptime"],
        "m5Col": colmap["fiveSigmaDepth"],
        "nightCol": colmap["night"],
        "filterCol": colmap["filter"],
    }

    def _setup_child_metrics(parentMetric):
        childMetrics = {}
        childMetrics["Time"] = metrics.Discovery_TimeMetric(parentMetric, **colkwargs)
        childMetrics["N_Chances"] = metrics.Discovery_N_ChancesMetric(
            parentMetric, **colkwargs
        )
        # Could expand to add N_chances per year, but not really necessary.
        return childMetrics

    def _configure_child_bundles(parentBundle):
        dispDict = {
            "group": f"{objtype}",
            "subgroup": f"Completeness Over Time",
            "caption": "Time of discovery of objects",
            "order": 0,
        }
        parentBundle.child_bundles["Time"].set_display_dict(dispDict)
        dispDict = {
            "group": f"{objtype}",
            "subgroup": f"N Chances",
            "caption": "Number of chances for discovery of objects",
            "order": 0,
        }
        parentBundle.child_bundles["N_Chances"].set_display_dict(dispDict)

    tMin = 5.0 / 60.0 / 24.0
    tMax = 90.0 / 60.0 / 24.0

    # 3 pairs in 15 and 3 pairs in 30 done in 'quickDiscoveryBatch' (with vis).

    # 4 pairs in 20
    md = info_label + " 4 pairs in 20 nights" + detectionLosses
    plotDict = {"title": "%s: %s" % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(
        nObsPerNight=2,
        tMin=tMin,
        tMax=tMax,
        nNightsPerWindow=4,
        tWindow=20,
        **colkwargs,
    )
    childMetrics = _setup_child_metrics(metric)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=[magStacker],
        run_name=runName,
        info_label=md,
        child_metrics=childMetrics,
        plot_dict=plotDict,
        plot_funcs=plotFuncs,
        display_dict=displayDict,
    )
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # 3 triplets in 30
    md = info_label + " 3 triplets in 30 nights" + detectionLosses
    plotDict = {"title": "%s: %s" % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(
        nObsPerNight=3,
        tMin=tMin,
        tMax=120.0 / 60.0 / 24.0,
        nNightsPerWindow=3,
        tWindow=30,
        **colkwargs,
    )
    childMetrics = _setup_child_metrics(metric)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=[magStacker],
        run_name=runName,
        info_label=md,
        child_metrics=childMetrics,
        plot_dict=plotDict,
        plot_funcs=plotFuncs,
        display_dict=displayDict,
    )
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # 1 quad
    md = info_label + " 1 quad in 1 night" + detectionLosses
    plotDict = {"title": "%s: %s" % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(
        nObsPerNight=4,
        tMin=tMin,
        tMax=150.0 / 60.0 / 24.0,
        nNightsPerWindow=1,
        tWindow=2,
        **colkwargs,
    )
    childMetrics = _setup_child_metrics(metric)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=[magStacker],
        run_name=runName,
        info_label=md,
        child_metrics=childMetrics,
        plot_dict=plotDict,
        plot_funcs=plotFuncs,
        display_dict=displayDict,
    )
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # Play with SNR.
    # First standard SNR / probabilistic visibility (SNR~5)
    # 3 pairs in 15
    md = info_label + " 3 pairs in 15 nights SNR=5" + detectionLosses
    # Set up plot dict.
    plotDict = {"title": "%s: %s" % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(
        nObsPerNight=2,
        tMin=tMin,
        tMax=tMax,
        nNightsPerWindow=3,
        tWindow=15,
        snrLimit=5,
        **colkwargs,
    )
    childMetrics = _setup_child_metrics(metric)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=[magStacker],
        run_name=runName,
        info_label=md,
        child_metrics=childMetrics,
        plot_dict=plotDict,
        plot_funcs=plotFuncs,
        display_dict=displayDict,
    )
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # 3 pairs in 15, SNR=4.
    md = info_label + " 3 pairs in 15 nights SNR=4" + detectionLosses
    plotDict = {"title": "%s: %s" % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(
        nObsPerNight=2,
        tMin=tMin,
        tMax=tMax,
        nNightsPerWindow=3,
        tWindow=15,
        snrLimit=4,
        **colkwargs,
    )
    childMetrics = _setup_child_metrics(metric)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=[magStacker],
        run_name=runName,
        info_label=md,
        child_metrics=childMetrics,
        plot_dict=plotDict,
        plot_funcs=plotFuncs,
        display_dict=displayDict,
    )
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # 3 pairs in 15, SNR=3
    md = info_label + " 3 pairs in 15 nights SNR=3" + detectionLosses
    plotDict = {"title": "%s: %s" % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(
        nObsPerNight=2,
        tMin=tMin,
        tMax=tMax,
        nNightsPerWindow=3,
        tWindow=15,
        snrLimit=3,
        **colkwargs,
    )
    childMetrics = _setup_child_metrics(metric)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=[magStacker],
        run_name=runName,
        info_label=md,
        child_metrics=childMetrics,
        plot_dict=plotDict,
        plot_funcs=plotFuncs,
        display_dict=displayDict,
    )
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # SNR = 0
    # 3 pairs in 15, SNR=0
    md = info_label + " 3 pairs in 15 nights SNR=0" + detectionLosses
    plotDict = {"title": "%s: %s" % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(
        nObsPerNight=2,
        tMin=tMin,
        tMax=tMax,
        nNightsPerWindow=3,
        tWindow=15,
        snrLimit=0,
        **colkwargs,
    )
    childMetrics = _setup_child_metrics(metric)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=[magStacker],
        run_name=runName,
        info_label=md,
        child_metrics=childMetrics,
        plot_dict=plotDict,
        plot_funcs=plotFuncs,
        display_dict=displayDict,
    )
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # Play with weird strategies.
    # Single detection.
    md = info_label + " Single detection" + detectionLosses
    plotDict = {"title": "%s: %s" % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(
        nObsPerNight=1,
        tMin=tMin,
        tMax=tMax,
        nNightsPerWindow=1,
        tWindow=5,
        **colkwargs,
    )
    childMetrics = _setup_child_metrics(metric)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=[magStacker],
        run_name=runName,
        info_label=md,
        child_metrics=childMetrics,
        plot_dict=plotDict,
        plot_funcs=plotFuncs,
        display_dict=displayDict,
    )
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # Single pair of detections.
    md = info_label + " Single pair" + detectionLosses
    plotDict = {"title": "%s: %s" % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(
        nObsPerNight=2,
        tMin=tMin,
        tMax=tMax,
        nNightsPerWindow=1,
        tWindow=5,
        **colkwargs,
    )
    childMetrics = _setup_child_metrics(metric)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=[magStacker],
        run_name=runName,
        info_label=md,
        child_metrics=childMetrics,
        plot_dict=plotDict,
        plot_funcs=plotFuncs,
        display_dict=displayDict,
    )
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # High velocity discovery.
    md = info_label + " High velocity pair" + detectionLosses
    plotDict = {"title": "%s: %s" % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.HighVelocityNightsMetric(
        psfFactor=2.0, nObsPerNight=2, **colkwargs
    )
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=[magStacker],
        run_name=runName,
        info_label=md,
        plot_dict=plotDict,
        plot_funcs=plotFuncs,
        display_dict=displayDict,
    )
    bundleList.append(bundle)

    # "magic" detection - 6 in 60 days.
    md = info_label + " 6 detections in 60 nights" + detectionLosses
    plotDict = {"title": "%s: %s" % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.MagicDiscoveryMetric(nObs=6, tWindow=60, **colkwargs)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=[magStacker],
        run_name=runName,
        info_label=md,
        plot_dict=plotDict,
        plot_funcs=plotFuncs,
        display_dict=displayDict,
    )
    bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.set_run_name(runName)
    return mb.make_bundles_dict_from_list(bundleList)


def runCompletenessSummary(bdict, Hmark, times, outDir, resultsDb):
    """
    Calculate completeness and create completeness bundles from all N_Chances and Time (child) metrics
    of the (discovery) bundles in bdict, and write completeness at Hmark to resultsDb, save bundle to disk.

    This should be done after combining any sub-sets of the metric results.

    Parameters
    ----------
    bdict : dict of metric_bundles
        Dict containing ~rubin_sim.maf.MoMetricBundles,
        including bundles we're expecting to contain completeness.
    Hmark : float
        Hmark value to add to completeness plotting dict.
        If not defined (None), then the Hmark from the plotdict from the metric will be used if available.
        If None and Hmark not in plotDict, then median of Hrange value will be used.
    times : np.ndarray
        The times at which to calculate completeness (over time).
    outDir : str
        Output directory to save completeness bundles to disk.
    resultsDb : ~rubin_sim.maf.db.ResultsDb
        Results database to save information about completeness bundle.

    Returns
    -------
    dict of metric_bundles
        A dictionary of the new completeness bundles. Keys match original keys,
        with additions of "[Differential,Cumulative]Completeness@Time"
        and "[Differential,Cumulative]Completeness" to distinguish new entries.
    """
    # Add completeness bundles and write completeness at Hmark to resultsDb.
    completeness = {}

    def _compbundles(b, bundle, Hmark, resultsDb):
        # Find Hmark if not set (this may be different for different bundles).
        if Hmark is None and "Hmark" in bundle.plotDict:
            Hmark = bundle.plotDict["Hmark"]
        if Hmark is None:
            Hmark = np.median(bundle.slicer.slicePoints["H"])
        # Set up the summary metrics.
        summaryTimeMetrics = summaryCompletenessAtTime(times, Hval=Hmark, Hindex=0.33)
        summaryTimeMetrics2 = summaryCompletenessAtTime(
            times, Hval=Hmark - 2, Hindex=0.33
        )
        summaryHMetrics = summaryCompletenessOverH(requiredChances=1, Hindex=0.33)
        comp = {}
        # Bundle = single metric bundle. Add differential and cumulative completeness.
        if "Time" in bundle.metric.name:
            for metric in summaryTimeMetrics:
                newkey = b + " " + metric.name
                comp[newkey] = mb.make_completeness_bundle(
                    bundle, metric, h_mark=None, results_db=resultsDb
                )
                comp[newkey].plotDict["times"] = times
                comp[newkey].plotDict["Hval"] = metric.Hval
            for metric in summaryTimeMetrics2:
                newkey = b + " " + metric.name
                comp[newkey] = mb.make_completeness_bundle(
                    bundle, metric, h_mark=None, results_db=resultsDb
                )
                comp[newkey].plotDict["times"] = times
                comp[newkey].plotDict["Hval"] = metric.Hval
        elif "N_Chances" in bundle.metric.name:
            for metric in summaryHMetrics:
                newkey = b + " " + metric.name
                comp[newkey] = mb.make_completeness_bundle(
                    bundle, metric, h_mark=Hmark, results_db=resultsDb
                )
        elif "MagicDiscovery" in bundle.metric.name:
            for metric in summaryHMetrics:
                newkey = b + " " + metric.name
                comp[newkey] = mb.make_completeness_bundle(
                    bundle, metric, h_mark=Hmark, results_db=resultsDb
                )
        elif "HighVelocity" in bundle.metric.name:
            for metric in summaryHMetrics:
                newkey = b + " " + metric.name
                comp[newkey] = mb.make_completeness_bundle(
                    bundle, metric, h_mark=Hmark, results_db=resultsDb
                )
        return comp

    # Generate the completeness bundles for the various discovery metrics.
    for b, bundle in bdict.items():
        if "Discovery" in bundle.metric.name:
            completeness.update(_compbundles(b, bundle, Hmark, resultsDb))
        if "MagicDiscovery" in bundle.metric.name:
            completeness.update(_compbundles(b, bundle, Hmark, resultsDb))
        if "HighVelocity" in bundle.metric.name:
            completeness.update(_compbundles(b, bundle, Hmark, resultsDb))

    # Write the completeness bundles to disk, so we can re-read them later.
    # (also set the display dict properties, for the resultsDb output).
    for b, bundle in completeness.items():
        bundle.displayDict["subgroup"] = f"Completeness"
        bundle.write(out_dir=outDir, results_db=resultsDb)

    # Calculate total number of objects - currently for NEOs and PHAs only
    for b, bundle in completeness.items():
        if "DifferentialCompleteness" in b and "@Time" not in b:
            if "NEO" in bundle.info_label:
                nobj_metrics = [
                    metrics.TotalNumberSSO(
                        Hmark=22, dndh_func=metrics.neo_dndh_granvik
                    ),
                    metrics.TotalNumberSSO(
                        Hmark=25, dndh_func=metrics.neo_dndh_granvik
                    ),
                ]
                bundle.set_summary_metrics(nobj_metrics)
                bundle.compute_summary_stats(resultsDb)
            if "PHA" in bundle.info_label:
                nobj_metrics = [
                    metrics.TotalNumberSSO(Hmark=22, dndh_func=metrics.pha_dndh_granvik)
                ]
                bundle.set_summary_metrics(nobj_metrics)
                bundle.compute_summary_stats(resultsDb)
    return completeness


def plotCompleteness(
    bdictCompleteness,
    figroot=None,
    runName=None,
    resultsDb=None,
    outDir=".",
    figformat="pdf",
):
    """Plot a minor subset of the completeness results."""
    # Separate some subsets to plot together - first just the simple 15 and 30 night detection loss metrics.
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
                if "Discovery_Time" in k:
                    if "Cumulative" in k:
                        plotTimes[k] = bdictCompleteness[k]
                elif "Discovery_N_Chances" in k:
                    if "Differential" in k:
                        plotDiff[k] = bdictCompleteness[k]
                    elif "Cumulative" in k:
                        plotComp[k] = bdictCompleteness[k]

    # Add plot dictionaries to code 30 nights red, 15 nights blue, differentials dotted.
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
    if runName is None:
        runName = first.run_name
    if figroot is None:
        figroot = runName
    displayDict = deepcopy(first.displayDict)

    # Plot completeness as a function of time. Make custom plot, then save it with PlotHandler.
    fig = plt.figure(figsize=(8, 6))
    for k in plotTimes:
        plt.plot(
            plotTimes[k].plotDict["times"],
            plotTimes[k].metricValues[0, :],
            label=plotTimes[k].plotDict["label"]
            + " @H=%.2f" % plotTimes[k].plotDict["Hval"],
        )
    plt.legend()
    plt.xlabel("Time (MJD)")
    plt.ylabel("Completeness")
    plt.grid(True, alpha=0.3)
    # Make a PlotHandler to deal with savings/resultsDb, etc.
    ph = plots.PlotHandler(figformat=figformat, resultsDb=resultsDb, outDir=outDir)
    displayDict["subgroup"] = f"Completeness over time"
    displayDict["caption"] = "Completeness over time, for H values indicated in legend."
    ph.saveFig(
        fig.number,
        f"{figroot}_CompletenessOverTime",
        "Combo",
        "CompletenessOverTime",
        "MoObjSlicer",
        figroot,
        None,
        None,
        displayDict=displayDict,
    )

    plt.savefig(
        os.path.join(outDir, f"{figroot}_CompletenessOverTime.{figformat}"),
        format=figformat,
    )

    # Plot cumulative completeness.
    ph = plots.PlotHandler(figformat=figformat, resultsDb=resultsDb, outDir=outDir)
    ph.setMetricBundles(plotComp)
    plotDict = {"ylabel": "Completeness", "figsize": (8, 6), "albedo": 0.14}
    ph.plot(
        plotFunc=plots.MetricVsH(),
        plotDicts=plotDict,
        outfileRoot=figroot + "_CumulativeCompleteness",
    )

    # Plot differential completeness.
    ph = plots.PlotHandler(figformat=figformat, resultsDb=resultsDb, outDir=outDir)
    ph.setMetricBundles(plotDiff)
    plotDict = {"ylabel": "Completeness", "figsize": (8, 6)}
    ph.plot(
        plotFunc=plots.MetricVsH(),
        plotDicts=plotDict,
        outfileRoot=figroot + "_DifferentialCompleteness",
    )

    # And add the rest of the completeness calculations.
    allComp = []
    for k in bdictCompleteness:
        if "Discovery_N_Chances" in k:
            if "Cumulative" in k:
                allComp.append(bdictCompleteness[k])
        if "Magic" in k:
            if "Cumulative" in k:
                allComp.append(bdictCompleteness[k])
    ph = plots.PlotHandler(figformat=figformat, resultsDb=resultsDb, outDir=outDir)
    ph.setMetricBundles(allComp)
    plotDict = {
        "ylabel": "Completeness",
        "figsize": (8, 6),
        "legendloc": (1.01, 0.1),
        "color": None,
    }
    displayDict["subgroup"] = f"Completeness all criteria"
    displayDict[
        "caption"
    ] = "Plotting all of the cumulative completeness curves together."
    ph.plot(
        plotFunc=plots.MetricVsH(),
        plotDicts=plotDict,
        displayDict=displayDict,
        outfileRoot=figroot + "_Many_CumulativeCompleteness",
    )


def characterizationInnerBatch(
    slicer,
    colmap=None,
    runName="opsim",
    objtype="",
    albedo=None,
    Hmark=None,
    constraintInfoLabel="",
    constraint=None,
    npReduce=np.mean,
    windows=None,
    bins=None,
):
    """Characterization metrics for inner solar system objects."""
    if colmap is None:
        colmap = col_map_dict("opsimV4")
    bundleList = []

    # Set up a dictionary to pass to each metric for the column names.
    colkwargs = {
        "mjdCol": colmap["mjd"],
        "seeingCol": colmap["seeingGeom"],
        "expTimeCol": colmap["exptime"],
        "m5Col": colmap["fiveSigmaDepth"],
        "nightCol": colmap["night"],
        "filterCol": colmap["filter"],
    }

    basicPlotDict = {
        "albedo": albedo,
        "Hmark": Hmark,
        "npReduce": npReduce,
        "nxbins": 200,
        "nybins": 200,
    }
    plotFuncs = [plots.MetricVsH()]

    if constraintInfoLabel == "" and constraint is not None:
        constraintInfoLabel = (
            constraint.replace("filter", "").replace("==", "").replace("  ", " ")
        )
    info_label = objtype + " " + constraintInfoLabel
    info_label = info_label.rstrip(" ")

    displayDict = {"group": f"{objtype}"}

    # Stackers
    magStacker = stackers.MoMagStacker(lossCol="dmagDetect")
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
    displayDict["subgroup"] = f"N Obs"
    plotDict = {
        "ylabel": "Number of observations (#)",
        "title": "%s: Number of observations %s" % (runName, md),
    }
    plotDict.update(basicPlotDict)
    metric = metrics.NObsMetric(**colkwargs)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=stackerList,
        run_name=runName,
        info_label=md,
        plot_dict=plotDict,
        plot_funcs=plotFuncs,
        display_dict=displayDict,
    )
    bundleList.append(bundle)

    # Observational arc.
    md = info_label
    displayDict["subgroup"] = f"Obs Arc"
    plotDict = {
        "ylabel": "Observational Arc (days)",
        "title": "%s: Observational Arc Length %s" % (runName, md),
    }
    plotDict.update(basicPlotDict)
    metric = metrics.ObsArcMetric(**colkwargs)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=stackerList,
        run_name=runName,
        info_label=md,
        plot_dict=plotDict,
        plot_funcs=plotFuncs,
        display_dict=displayDict,
    )
    bundleList.append(bundle)

    # Activity detection.
    displayDict["subgroup"] = f"Activity"
    for w in windows:
        md = info_label + " activity lasting %.0f days" % w
        plotDict = {
            "title": "%s: Chances of detecting %s" % (runName, md),
            "ylabel": "Probability of detection per %.0f day window" % w,
        }
        metricName = "Chances of detecting activity lasting %.0f days" % w
        metric = metrics.ActivityOverTimeMetric(w, metricName=metricName, **colkwargs)
        bundle = MoMetricBundle(
            metric,
            slicer,
            constraint,
            stacker_list=stackerList,
            run_name=runName,
            info_label=info_label,
            plot_dict=plotDict,
            plot_funcs=plotFuncs,
            display_dict=displayDict,
        )
        bundleList.append(bundle)

    for b in bins:
        md = info_label + " activity covering %.0f deg" % (b)
        plotDict = {
            "title": "%s: Chances of detecting %s" % (runName, md),
            "ylabel": "Probability of detection per %.0f deg window" % b,
        }
        metricName = "Chances of detecting activity covering %.0f deg" % (b)
        metric = metrics.ActivityOverPeriodMetric(b, metricName=metricName, **colkwargs)
        bundle = MoMetricBundle(
            metric,
            slicer,
            constraint,
            stacker_list=stackerList,
            run_name=runName,
            info_label=info_label,
            plot_dict=plotDict,
            plot_funcs=plotFuncs,
            display_dict=displayDict,
        )
        bundleList.append(bundle)

    # Lightcurve inversion.
    md = info_label
    displayDict["subgroup"] = f"Color/Inversion"
    plotDict = {
        "yMin": 0,
        "yMax": 1,
        "ylabel": "Fraction of objects",
        "title": "%s: Fraction with potential lightcurve inversion %s" % (runName, md),
    }
    plotDict.update(basicPlotDict)
    metric = metrics.LightcurveInversion_AsteroidMetric(**colkwargs)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=stackerList,
        run_name=runName,
        info_label=md,
        plot_dict=plotDict,
        plot_funcs=plotFuncs,
        display_dict=displayDict,
    )
    bundleList.append(bundle)

    # Color determination.
    md = info_label
    plotDict = {
        "yMin": 0,
        "yMax": 1,
        "ylabel": "Fraction of objects",
        "title": "%s: Fraction of population with colors in X filters %s"
        % (runName, md),
    }
    plotDict.update(basicPlotDict)
    metric = metrics.Color_AsteroidMetric(**colkwargs)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=stackerList,
        run_name=runName,
        info_label=md,
        plot_dict=plotDict,
        plot_funcs=plotFuncs,
        display_dict=displayDict,
    )
    bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.set_run_name(runName)
    return mb.make_bundles_dict_from_list(bundleList)


def characterizationOuterBatch(
    slicer,
    colmap=None,
    runName="opsim",
    objtype="",
    albedo=None,
    Hmark=None,
    constraintInfoLabel="",
    constraint=None,
    npReduce=np.mean,
    windows=None,
    bins=None,
):
    """Characterization metrics for outer solar system objects."""
    if colmap is None:
        colmap = col_map_dict("opsimV4")
    bundleList = []

    # Set up a dictionary to pass to each metric for the column names.
    colkwargs = {
        "mjdCol": colmap["mjd"],
        "seeingCol": colmap["seeingGeom"],
        "expTimeCol": colmap["exptime"],
        "m5Col": colmap["fiveSigmaDepth"],
        "nightCol": colmap["night"],
        "filterCol": colmap["filter"],
    }

    basicPlotDict = {
        "albedo": albedo,
        "Hmark": Hmark,
        "npReduce": npReduce,
        "nxbins": 200,
        "nybins": 200,
    }
    plotFuncs = [plots.MetricVsH()]

    if constraintInfoLabel == "" and constraint is not None:
        constraintInfoLabel = (
            constraint.replace("filter", "").replace("==", "").replace("  ", " ")
        )
    info_label = objtype + " " + constraintInfoLabel
    info_label = info_label.rstrip(" ")

    displayDict = {"group": f"{objtype}"}

    # Stackers
    magStacker = stackers.MoMagStacker(lossCol="dmagDetect")
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
    displayDict["subgroup"] = f"N Obs"
    plotDict = {
        "ylabel": "Number of observations (#)",
        "title": "%s: Number of observations %s" % (runName, md),
    }
    plotDict.update(basicPlotDict)
    metric = metrics.NObsMetric(**colkwargs)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=stackerList,
        run_name=runName,
        info_label=md,
        plot_dict=plotDict,
        plot_funcs=plotFuncs,
        display_dict=displayDict,
    )
    bundleList.append(bundle)

    # Observational arc.
    md = info_label
    displayDict["subgroup"] = f"Obs Arc"
    plotDict = {
        "ylabel": "Observational Arc (days)",
        "title": "%s: Observational Arc Length %s" % (runName, md),
    }
    plotDict.update(basicPlotDict)
    metric = metrics.ObsArcMetric(**colkwargs)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=stackerList,
        run_name=runName,
        info_label=md,
        plot_dict=plotDict,
        plot_funcs=plotFuncs,
        display_dict=displayDict,
    )
    bundleList.append(bundle)

    # Activity detection.
    displayDict["subgroup"] = f"Activity"
    for w in windows:
        md = info_label + " activity lasting %.0f days" % w
        plotDict = {
            "title": "%s: Chances of detecting %s" % (runName, md),
            "ylabel": "Probability of detection per %.0f day window" % w,
        }
        metricName = "Chances of detecting activity lasting %.0f days" % w
        metric = metrics.ActivityOverTimeMetric(w, metricName=metricName, **colkwargs)
        bundle = MoMetricBundle(
            metric,
            slicer,
            constraint,
            stacker_list=stackerList,
            run_name=runName,
            info_label=info_label,
            plot_dict=plotDict,
            plot_funcs=plotFuncs,
            display_dict=displayDict,
        )
        bundleList.append(bundle)

    for b in bins:
        md = info_label + " activity covering %.0f deg" % (b)
        plotDict = {
            "title": "%s: Chances of detecting %s" % (runName, md),
            "ylabel": "Probability of detection per %.2f deg window" % b,
        }
        metricName = "Chances of detecting activity covering %.0f deg" % (b)
        metric = metrics.ActivityOverPeriodMetric(b, metricName=metricName, **colkwargs)
        bundle = MoMetricBundle(
            metric,
            slicer,
            constraint,
            stacker_list=stackerList,
            run_name=runName,
            info_label=info_label,
            plot_dict=plotDict,
            plot_funcs=plotFuncs,
            display_dict=displayDict,
        )
        bundleList.append(bundle)

    # Color determination.
    md = info_label
    displayDict["subgroup"] = f"Color/Inversion"
    plotDict = {
        "yMin": 0,
        "yMax": 1,
        "ylabel": "Fraction of objects",
        "title": "%s: Fraction of population with colors in X filters %s"
        % (runName, md),
    }
    plotDict.update(basicPlotDict)
    metric = metrics.LightcurveColor_OuterMetric(**colkwargs)
    bundle = MoMetricBundle(
        metric,
        slicer,
        constraint,
        stacker_list=stackerList,
        run_name=runName,
        info_label=md,
        plot_dict=plotDict,
        plot_funcs=plotFuncs,
        display_dict=displayDict,
    )
    bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.set_run_name(runName)
    return mb.make_bundles_dict_from_list(bundleList)


def runFractionSummary(bdict, Hmark, outDir, resultsDb):
    """
    Calculate fractional completeness of the population for color and lightcurve metrics.

    This should be done after combining any sub-sets of the metric results.

    Parameters
    ----------
    bdict : dict of metric_bundles
        Dict containing ~rubin_sim.maf.MoMetricBundles,
        including bundles we're expecting to contain lightcurve/color evaluations.
    Hmark : float
        Hmark value to add to completeness plotting dict.
        If defined, this value is used. If None, but Hmark in plotDict for metric, then this value (-2) is
        used. If Hmark not in plotdict, then the median Hrange value - 2 is used.
    times : np.ndarray
        The times at which to calculate completeness (over time).
    outDir : str
        Output directory to save completeness bundles to disk.
    resultsDb : ~rubin_sim.maf.db.ResultsDb
        Results database to save information about completeness bundle.

    Returns
    -------
    dict of metric_bundles
        Dictionary of the metric bundles for the fractional evaluation of the population.
    """
    fractions = {}

    # Look for metrics from asteroid or outer solar system color/lightcurve metrics.
    inversionSummary = fractionPopulationAtThreshold([1], ["Lightcurve Inversion"])
    asteroidColorSummary = fractionPopulationAtThreshold(
        [4, 3, 2, 1],
        ["6 of ugrizy", "5 of grizy", "4 of grizy", "2 of g, r or i, z or y"],
    )
    asteroidSummaryMetrics = {
        "LightcurveInversion_Asteroid": inversionSummary,
        "Color_Asteroid": asteroidColorSummary,
    }

    outerColorSummary = fractionPopulationAtThreshold(
        [6, 5, 4, 3, 2, 1],
        ["6 filters", "5 filters", "4 filters", "3 filters", "2 filters", "1 filters"],
    )
    outerSummaryMetrics = {"LightcurveColor_Outer": outerColorSummary}

    for b, bundle in bdict.items():
        # Find Hmark if not set (this may be different for different bundles).
        if Hmark is None and "Hmark" in bundle.plotDict:
            Hmark = bundle.plotDict["Hmark"] - 2
        if Hmark is None:
            Hmark = np.median(bundle.slicer.slicePoints["H"]) - 2
        # Make sure we didn't push Hmark outside the range of H values for metric
        if Hmark < bundle.slicer.slicePoints["H"].min():
            Hmark = bundle.slicer.slicePoints["H"].min()
        for k in asteroidSummaryMetrics:
            if k in b:
                for summary_metric in asteroidSummaryMetrics[k]:
                    newkey = b + " " + summary_metric.name
                    fractions[newkey] = mb.make_completeness_bundle(
                        bundle, summary_metric, h_mark=Hmark, results_db=resultsDb
                    )
        for k in outerSummaryMetrics:
            if k in b:
                for summary_metric in outerSummaryMetrics[k]:
                    newkey = b + " " + summary_metric.name
                    fractions[newkey] = mb.make_completeness_bundle(
                        bundle, summary_metric, h_mark=Hmark, results_db=resultsDb
                    )
    # Write the fractional populations bundles to disk, so we can re-read them later.
    for b, bundle in fractions.items():
        bundle.write(out_dir=outDir, results_db=resultsDb)
    return fractions


def plotFractions(
    bdictFractions,
    figroot=None,
    runName=None,
    resultsDb=None,
    outDir=".",
    figformat="pdf",
):
    # Set colors for the fractions.
    for b in bdictFractions.values():
        k = b.metric.name
        if "6" in k:
            b.plotDict["color"] = "b"
        if "5" in k:
            b.plotDict["color"] = "cyan"
        if "4" in k:
            b.plotDict["color"] = "orange"
        if "2" in k:
            b.plotDict["color"] = "r"
        if "1" in k:
            b.plotDict["color"] = "magenta"
        if "Lightcurve Inversion" in k:
            b.plotDict["color"] = "k"
            b.plotDict["linestyle"] = ":"
            b.plotDict["linewidth"] = 3

    first = bdictFractions[list(bdictFractions.keys())[0]]
    if figroot is None:
        figroot = first.run_name
    displayDict = deepcopy(first.displayDict)
    displayDict["subgroup"] = f"Characterization Fraction"

    ph = plots.PlotHandler(figformat=figformat, resultsDb=resultsDb, outDir=outDir)
    ph.setMetricBundles(bdictFractions)
    ph.jointMetricNames = "Fraction of population for colors or lightcurve inversion"
    plotDict = {"ylabel": "Fraction of population", "figsize": (8, 6)}
    ph.plot(
        plotFunc=plots.MetricVsH(),
        plotDicts=plotDict,
        displayDict=displayDict,
        outfileRoot=figroot + "_characterization",
    )


def plotSingle(bundle, resultsDb=None, outDir=".", figformat="pdf"):
    """Plot 5%/25%/50%/75%/95% iles for a metric value."""
    pDict = {
        "95%ile": {
            "color": "k",
            "linestyle": "--",
            "label": "95th %ile",
            "npReduce": lambda x, axis: np.percentile(x, 95, axis=axis),
        },
        "75%ile": {
            "color": "magenta",
            "linestyle": ":",
            "label": "75th %ile",
            "npReduce": lambda x, axis: np.percentile(x, 75, axis=axis),
        },
        "Median": {
            "color": "b",
            "linestyle": "-",
            "label": "Median",
            "npReduce": lambda x, axis: np.median(x, axis=axis),
        },
        "Mean": {"color": "g", "linestyle": "--", "label": "Mean", "npReduce": np.mean},
        "25%ile": {
            "color": "magenta",
            "linestyle": ":",
            "label": "25th %ile",
            "npReduce": lambda x, axis: np.percentile(x, 25, axis=axis),
        },
        "5%ile": {
            "color": "k",
            "linestyle": "--",
            "label": "5th %ile",
            "npReduce": lambda x, axis: np.percentile(x, 5, axis=axis),
        },
    }
    ph = plots.PlotHandler(figformat=figformat, resultsDb=resultsDb, outDir=outDir)
    plotBundles = []
    plotDicts = []
    for percentile in pDict:
        plotBundles.append(bundle)
        plotDicts.append(pDict[percentile])
    plotDicts[0].update({"figsize": (8, 6), "legendloc": "upper right", "yMin": 0})
    # Remove the Hmark line because these plots get complicated already.
    for r in plotDicts:
        r["Hmark"] = None
    ph.setMetricBundles(plotBundles)
    ph.plot(
        plotFunc=plots.MetricVsH(), plotDicts=plotDicts, displayDict=bundle.displayDict
    )


def plotNotFound(nChances, Hmark):
    pass


def plotActivity(bdict, figroot=None, resultsDb=None, outDir=".", figformat="pdf"):
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
    displayDict = deepcopy(first.displayDict)

    if len(activity_days) > 0:
        # Plot (mean) likelihood of detection of activity over X days
        ph = plots.PlotHandler(figformat=figformat, resultsDb=resultsDb, outDir=outDir)
        ph.setMetricBundles(activity_days)
        ph.jointMetricNames = "Chances of detecting activity lasting X days"
        plotDict = {"ylabel": "Mean likelihood of detection", "figsize": (8, 6)}
        ph.plot(
            plotFunc=plots.MetricVsH(),
            plotDicts=plotDict,
            displayDict=displayDict,
            outfileRoot=figroot + "_activityDays",
        )
    if len(activity_deg) > 0:
        # Plot (mean) likelihood of detection of activity over X amount of orbit
        ph = plots.PlotHandler(figformat=figformat, resultsDb=resultsDb, outDir=outDir)
        ph.setMetricBundles(activity_deg)
        ph.jointMetricNames = "Chances of detecting activity covering X deg"
        plotDict = {"ylabel": "Mean likelihood of detection", "figsize": (8, 6)}
        ph.plot(
            plotFunc=plots.MetricVsH(),
            plotDicts=plotDict,
            displayDict=displayDict,
            outfileRoot=figroot + "_activityDeg",
        )


def readAndCombine(orbitRoot, baseDir, splits, metricfile):
    """Read and combine the metric results from split locations, returning a single bundle.

    This will read the files from
    baseDir/orbitRoot_[split]/metricfile
    where split = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], etc. (the subsets the original orbit file was split into).

    Parameters
    ----------
    orbitRoot: str
        The root of the orbit file - l7_5k, mbas_5k, etc.
    baseDir: str
        The root directory containing the subset directories. (e.g. '.' often)
    splits: np.ndarray or list of ints
        The integers describing the split directories (e.g. [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    metricfile: str
        The metric filename.

    Returns
    -------
    ~rubin_sim.maf.bundle
        A single metric bundle containing the combined data from each of the subsets.

    Note that this won't work for particularly complex metric values, such as the parent Discovery metrics.
    However, you can read and combine their child metrics, as for these we can propagate the data masks.
    """
    subsets = {}
    for i in splits:
        subsets[i] = mb.create_empty_mo_metric_bundle()
        ddir = os.path.join(baseDir, f"{orbitRoot}_{i}")
        subsets[i].read(os.path.join(ddir, metricfile))
    bundle = combineSubsets(subsets)
    return bundle


def combineSubsets(mbSubsets):
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
        if np.any(slicer.slicePoints["H"] != mbSubsets[i].slicer.slicePoints["H"]):
            if np.any(
                slicer.slicePoints["orbits"]
                != mbSubsets[i].slicer.slicePoints["orbits"]
            ):
                raise ValueError(
                    "Bundle %s has a different slicer than the first bundle" % (i)
                )
    # Join metric values.
    joint.slicer = slicer
    joint.metric = first.metric
    # Don't just use the slicer shape to define the metricValues, because of CompletenessBundles.
    metricValues = np.zeros(first.metricValues.shape, float)
    metricValuesMask = np.zeros(first.metricValues.shape, bool)
    for i in mbSubsets:
        metricValues += mbSubsets[i].metricValues.filled(0)
        metricValuesMask = np.where(
            metricValuesMask & mbSubsets[i].metricValues.mask, True, False
        )
    joint.metricValues = ma.MaskedArray(
        data=metricValues, mask=metricValuesMask, fill_value=0
    )
    joint.info_label = first.info_label
    joint.run_name = first.run_name
    joint.file_root = first.file_root.replace(".npz", "")
    joint.plotDict = first.plotDict
    joint.displayDict = first.displayDict
    return joint
