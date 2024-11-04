__all__ = ("run_selfcal_metric",)

import argparse
import os
import shutil

import healpy as hp
import matplotlib
import numpy as np

matplotlib.use("Agg")

from .db import ResultsDb
from .maf_contrib import PhotometricSelfCalUniformityMetric
from .metric_bundles import MetricBundle, MetricBundleGroup
from .metrics import IdentityMetric
from .plots import HealpixHistogram, HealpixSkyMap, PlotHandler
from .slicers import HealpixSlicer, UniSlicer


def run_selfcal_metric():
    """
    Run the self-calibration metric on one database.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument(
        "--no_clobber",
        dest="no_clobber",
        default=False,
        action="store_true",
        help="Do not remove existing directory outputs",
    )
    parser.add_argument("--filter", type=str, default="r")

    args = parser.parse_args()

    opsdb = args.db
    sim_name = os.path.basename(opsdb).replace(".db", "")

    out_dir = sim_name + "_selfcal"
    if not args.no_clobber:
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)

    # Set up the metric bundle.
    map_nside = 128
    selfcal_metric = PhotometricSelfCalUniformityMetric(
        nside_residual=map_nside, filter_name=args.filter, metric_name="selfcal_uniformity-%s" % args.filter
    )
    slicer = UniSlicer()
    # Exclude DDF visits
    sql = "scheduler_note not like '%DD%'"
    # And run on only year 1 (?)
    sql += " and night < 366"
    sql += " and filter='%s'" % args.filter

    bundle = MetricBundle(
        selfcal_metric, slicer, sql, run_name=sim_name, info_label="year 1 no-DD %s" % args.filter
    )

    # Set up the resultsDB
    results_db = ResultsDb(out_dir=out_dir)
    # Go and run it
    group = MetricBundleGroup(
        {"selfcal": bundle}, opsdb, out_dir=out_dir, results_db=results_db, save_early=True
    )
    group.run_all(clear_memory=False, plot_now=True)

    # Make plots of the residuals map
    map_bundle = MetricBundle(
        IdentityMetric(metric_name="PhotoCal Uniformity %s" % args.filter),
        HealpixSlicer(map_nside),
        sql,
        run_name=sim_name,
        info_label="year 1 no-DD",
    )
    tmp_vals = bundle.metric_values[0]["uniformity_map"]
    tmp_vals = np.where(tmp_vals == hp.UNSEEN, map_bundle.slicer.badval, tmp_vals)
    map_bundle.metric_values = np.ma.MaskedArray(
        data=tmp_vals,
        mask=np.zeros(map_bundle.slicer.shape, "bool"),
        fill_value=map_bundle.slicer.badval,
    )
    map_bundle.write(out_dir=out_dir, results_db=results_db)

    ph = PlotHandler(results_db=results_db, out_dir=out_dir)
    ph.set_metric_bundles([map_bundle])
    _ = ph.plot(HealpixSkyMap(), plot_dicts={"color_min": -0.02, "color_max": 0.02})
    _ = ph.plot(HealpixHistogram(), plot_dicts={"percentile_clip": 99})

    results_db.close()
