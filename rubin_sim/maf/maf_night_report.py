__all__ = ("make_bundle_list", "maf_night_report")

import argparse

import matplotlib

# Set matplotlib backend (to create plots where DISPLAY is not set).
matplotlib.use("Agg")
from . import db as db
from . import metricBundles as metricBundles
from . import metrics as metrics
from . import plots as plots
from . import slicers as slicers
from . import utils as utils
from .batches import col_map_dict


def make_bundle_list(
    db_file,
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
        colmap = col_map_dict("opsimV4")

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
    display_dict = {"group": "Hourglass"}
    md = ""
    sql = "night=%i" % night
    metric = metrics.HourglassMetric(
        night_col=colmap["night"], mjd_col=colmap["mjd"], metric_name="Hourglass"
    )
    bundle = metricBundles.MetricBundle(
        metric, hourslicer, constraint=sql, info_label=md, display_dict=display_dict
    )
    bundleList.append(bundle)

    reg_slicer = slicers.HealpixSlicer(nside=nside, lon_col=lonCol, lat_col=latCol, lat_lon_deg=True)
    altaz_slicer = slicers.HealpixSlicer(
        nside=nside, lat_col=altCol, lat_lon_deg=True, lon_col=azCol, use_cache=False
    )

    unislicer = slicers.UniSlicer()
    for sql in sqls:
        # Number of exposures
        metric = metrics.CountMetric(mjdCol, metric_name="N visits")
        bundle = metricBundles.MetricBundle(metric, reg_slicer, sql)
        bundleList.append(bundle)
        metric = metrics.CountMetric(mjdCol, metric_name="N visits alt az")
        bundle = metricBundles.MetricBundle(metric, altaz_slicer, sql, plot_funcs=plotFuncs_lam)
        bundleList.append(bundle)

        metric = metrics.MeanMetric(mjdCol, metric_name="Mean Visit Time")
        bundle = metricBundles.MetricBundle(metric, reg_slicer, sql)
        bundleList.append(bundle)
        metric = metrics.MeanMetric(mjdCol, metric_name="Mean Visit Time alt az")
        bundle = metricBundles.MetricBundle(metric, altaz_slicer, sql, plot_funcs=plotFuncs_lam)
        bundleList.append(bundle)

        metric = metrics.CountMetric(mjdCol, metric_name="N_visits")
        bundle = metricBundles.MetricBundle(metric, unislicer, sql)
        bundleList.append(bundle)

        # Need pairs in window to get a map of how well it gathered SS pairs.

    # Moon phase.

    metric = metrics.NChangesMetric(col="filter", metric_name="Filter Changes")
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
    metric = metrics.PairMetric(mjd_col=mjdCol)
    sql = 'night=%i and (filter ="r" or filter="g" or filter="i")' % night
    bundle = metricBundles.MetricBundle(metric, reg_slicer, sql)
    bundleList.append(bundle)

    metric = metrics.PairMetric(mjd_col=mjdCol, metric_name="z Pairs")
    sql = 'night=%i and filter="z"' % night
    bundle = metricBundles.MetricBundle(metric, reg_slicer, sql)
    bundleList.append(bundle)

    # Plot up each visit
    metric = metrics.NightPointingMetric(mjd_col=mjdCol)
    slicer = slicers.UniSlicer()
    sql = "night=%i" % night
    plot_funcs = [plots.NightPointingPlotter()]
    bundle = metricBundles.MetricBundle(metric, slicer, sql, plot_funcs=plot_funcs)
    bundleList.append(bundle)

    # stats from the note column
    if notes:
        display_dict = {"group": "Basic Stats", "subgroup": "Percent stats"}
        metric = metrics.StringCountMetric(col="note", percent=True, metric_name="Percents")
        bundle = metricBundles.MetricBundle(metric, unislicer, sql, display_dict=display_dict)
        bundleList.append(bundle)
        display_dict["subgroup"] = "Count Stats"
        metric = metrics.StringCountMetric(col="note", metric_name="Counts")
        bundle = metricBundles.MetricBundle(metric, unislicer, sql, display_dict=display_dict)
        bundleList.append(bundle)

    return metricBundles.make_bundles_dict_from_list(bundleList)


def maf_night_report():
    """Generate a report on a single night."""

    parser = argparse.ArgumentParser(description="Python script to generate a report on a single night.")
    parser.add_argument(
        "db_file",
        type=str,
        default=None,
        help="full file path to the simulation sqlite file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="night_report",
        help="Output directory for MAF outputs." + ' Default "Out"',
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=64,
        help="Resolution to run Healpix grid at (must be 2^x). Default 64.",
    )
    parser.add_argument(
        "--lon_col",
        type=str,
        default="fieldRA",
        help="Column to use for RA values (can be a stacker dither column)." + " Default=fieldRA.",
    )
    parser.add_argument(
        "--lat_col",
        type=str,
        default="fieldDec",
        help="Column to use for Dec values (can be a stacker dither column)." + " Default=fieldDec.",
    )
    parser.add_argument("--night", type=int, default=1)
    parser.add_argument("--run_name", type=str, default="run_name")

    parser.set_defaults()
    args, extras = parser.parse_known_args()

    bundle_dict = make_bundle_list(
        args.db_file,
        nside=args.nside,
        lonCol=args.lon_col,
        latCol=args.lat_col,
        night=args.night,
    )

    for key in bundle_dict:
        bundle_dict[key].set_run_name(args.run_name)

    # Set up / connect to results_db.
    results_db = db.ResultsDb(out_dir=args.out_dir)

    # Set up metricBundleGroup.
    group = metricBundles.MetricBundleGroup(
        bundle_dict, args.db_file, out_dir=args.out_dir, results_db=results_db
    )
    group.run_all()
    group.plot_all()
