#!/usr/bin/env python

import argparse
import matplotlib

# Set matplotlib backend (to create plots where DISPLAY is not set).
matplotlib.use("Agg")
from . import db as db
from . import metrics as metrics
from . import slicers as slicers
from . import metricBundles as metricBundles
from . import plots as plots
from . import utils as utils

from .batches import ColMapDict


def makeBundleList(
    dbFile,
    night=1,
    nside=64,
    latCol="fieldDec",
    lonCol="fieldRA",
    notes=True,
    colmap=None,
):
    """
    Make a bundleList of things to run
    """

    if colmap is None:
        colmap = ColMapDict("opsimV4")

    mjdCol = "observationStartMJD"
    altCol = "altitude"
    azCol = "azimuth"
    # Construct sql queries for each filter and all filters
    filters = ["u", "g", "r", "i", "z", "y"]
    sqls = ['night=%i and filter="%s"' % (night, f) for f in filters]
    sqls.append("night=%i" % night)

    bundleList = []
    plotFuncs_lam = [plots.LambertSkyMap()]

    # Hourglass
    hourslicer = slicers.HourglassSlicer()
    displayDict = {"group": "Hourglass"}
    md = ""
    sql = "night=%i" % night
    metric = metrics.HourglassMetric(
        nightCol=colmap["night"], mjdCol=colmap["mjd"], metricName="Hourglass"
    )
    bundle = metricBundles.MetricBundle(
        metric, hourslicer, constraint=sql, info_label=md, display_dict=displayDict
    )
    bundleList.append(bundle)

    reg_slicer = slicers.HealpixSlicer(
        nside=nside, lon_col=lonCol, lat_col=latCol, lat_lon_deg=True
    )
    altaz_slicer = slicers.HealpixSlicer(
        nside=nside, lat_col=altCol, lat_lon_deg=True, lon_col=azCol, useCache=False
    )

    unislicer = slicers.UniSlicer()
    for sql in sqls:

        # Number of exposures
        metric = metrics.CountMetric(mjdCol, metricName="N visits")
        bundle = metricBundles.MetricBundle(metric, reg_slicer, sql)
        bundleList.append(bundle)
        metric = metrics.CountMetric(mjdCol, metricName="N visits alt az")
        bundle = metricBundles.MetricBundle(
            metric, altaz_slicer, sql, plot_funcs=plotFuncs_lam
        )
        bundleList.append(bundle)

        metric = metrics.MeanMetric(mjdCol, metricName="Mean Visit Time")
        bundle = metricBundles.MetricBundle(metric, reg_slicer, sql)
        bundleList.append(bundle)
        metric = metrics.MeanMetric(mjdCol, metricName="Mean Visit Time alt az")
        bundle = metricBundles.MetricBundle(
            metric, altaz_slicer, sql, plot_funcs=plotFuncs_lam
        )
        bundleList.append(bundle)

        metric = metrics.CountMetric(mjdCol, metricName="N_visits")
        bundle = metricBundles.MetricBundle(metric, unislicer, sql)
        bundleList.append(bundle)

        # Need pairs in window to get a map of how well it gathered SS pairs.

    # Moon phase.

    metric = metrics.NChangesMetric(col="filter", metricName="Filter Changes")
    bundle = metricBundles.MetricBundle(metric, unislicer, "night=%i" % night)
    bundleList.append(bundle)

    metric = metrics.BruteOSFMetric()
    bundle = metricBundles.MetricBundle(metric, unislicer, "night=%i" % night)
    bundleList.append(bundle)

    metric = metrics.MeanMetric("slewTime")
    bundle = metricBundles.MetricBundle(metric, unislicer, "night=%i" % night)
    bundleList.append(bundle)

    metric = metrics.MinMetric("slewTime")
    bundle = metricBundles.MetricBundle(metric, unislicer, "night=%i" % night)
    bundleList.append(bundle)

    metric = metrics.MaxMetric("slewTime")
    bundle = metricBundles.MetricBundle(metric, unislicer, "night=%i" % night)
    bundleList.append(bundle)

    # Make plots of the solar system pairs that were taken in the night
    metric = metrics.PairMetric(mjdCol=mjdCol)
    sql = 'night=%i and (filter ="r" or filter="g" or filter="i")' % night
    bundle = metricBundles.MetricBundle(metric, reg_slicer, sql)
    bundleList.append(bundle)

    metric = metrics.PairMetric(mjdCol=mjdCol, metricName="z Pairs")
    sql = 'night=%i and filter="z"' % night
    bundle = metricBundles.MetricBundle(metric, reg_slicer, sql)
    bundleList.append(bundle)

    # Plot up each visit
    metric = metrics.NightPointingMetric(mjdCol=mjdCol)
    slicer = slicers.UniSlicer()
    sql = "night=%i" % night
    plotFuncs = [plots.NightPointingPlotter()]
    bundle = metricBundles.MetricBundle(metric, slicer, sql, plot_funcs=plotFuncs)
    bundleList.append(bundle)

    # stats from the note column
    if notes:
        displayDict = {"group": "Basic Stats", "subgroup": "Percent stats"}
        metric = metrics.StringCountMetric(
            col="note", percent=True, metricName="Percents"
        )
        bundle = metricBundles.MetricBundle(
            metric, unislicer, sql, display_dict=displayDict
        )
        bundleList.append(bundle)
        displayDict["subgroup"] = "Count Stats"
        metric = metrics.StringCountMetric(col="note", metricName="Counts")
        bundle = metricBundles.MetricBundle(
            metric, unislicer, sql, display_dict=displayDict
        )
        bundleList.append(bundle)

    return metricBundles.make_bundles_dict_from_list(bundleList)


def maf_night_report():
    """Generate a report on a single night."""

    parser = argparse.ArgumentParser(
        description="Python script to generate a report on a single night."
    )
    parser.add_argument(
        "dbFile", type=str, default=None, help="full file path to the opsim sqlite file"
    )
    parser.add_argument(
        "--outDir",
        type=str,
        default="./Out",
        help="Output directory for MAF outputs." + ' Default "Out"',
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=64,
        help="Resolution to run Healpix grid at (must be 2^x). Default 64.",
    )
    parser.add_argument(
        "--lonCol",
        type=str,
        default="fieldRA",
        help="Column to use for RA values (can be a stacker dither column)."
        + " Default=fieldRA.",
    )
    parser.add_argument(
        "--latCol",
        type=str,
        default="fieldDec",
        help="Column to use for Dec values (can be a stacker dither column)."
        + " Default=fieldDec.",
    )
    parser.add_argument("--night", type=int, default=1)
    parser.add_argument("--runName", type=str, default="runName")

    parser.set_defaults()
    args, extras = parser.parse_known_args()

    bundleDict = makeBundleList(
        args.dbFile,
        nside=args.nside,
        lonCol=args.lonCol,
        latCol=args.latCol,
        night=args.night,
    )

    for key in bundleDict:
        bundleDict[key].set_run_name(args.run_name)

    # Set up / connect to resultsDb.
    resultsDb = db.ResultsDb(out_dir=args.outDir)
    # Connect to opsimdb.
    opsdb = db.OpsimDatabase(args.dbFile)

    # Set up metricBundleGroup.
    group = metricBundles.MetricBundleGroup(
        bundleDict, opsdb, out_dir=args.outDir, results_db=resultsDb
    )
    group.run_all()
    group.plotAll()
