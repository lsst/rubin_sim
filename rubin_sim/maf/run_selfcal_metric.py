__all__ = ("run_selfcal_metric",)

import argparse
import os
import shutil

import matplotlib

matplotlib.use("Agg")

from .db import ResultsDb
from .metric_bundles import MetricBundle, MetricBundleGroup
from .slicers import UniSlicer, HealpixSlicer
from .maf_contrib import PhotometricSelfCalUniformityMetric
from .plots import PlotHandler, HealpixSkyMap, HealpixHistogram


def run_selfcal_metric():
    """
    Run the self-calibration metric on one database.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument(
        "--no_clobber",
        dest="no_clobber",
        action="store_false",
        help="Do not remove existing directory outputs",
    )
    args = parser.parse_args()

    opsdb = args.db
    sim_name = os.path.basename(opsdb).replace(".db", "")

    out_dir = sim_name + "_selfcal"
    if not args.no_clobber:
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)

    # Set up the metric bundle.
    map_nside = 128
    selfcal_metric = PhotometricSelfCalUniformityMetric(nside_residual=map_nside)
    slicer = UniSlicer()
    # Exclude DDF visits
    sql = "note not like '%DD%'"
    # And run on only year 1 (?)
    sql += " and night < 366"
    bundle = MetricBundle(selfcal_metric, slicer, sql, run_name=sim_name, info_label="year 1 no-DD")

    # Set up the resultsDB
    results_db = ResultsDb(out_dir=out_dir)
    # Go and run it
    group = MetricBundleGroup(
        {"selfcal": bundle}, opsdb, out_dir=out_dir, results_db=results_db, save_early=False
    )
    group.run_all(clear_memory=False, plot_now=True)

    # Make plots of the residuals map
    resid_map_key = [k for k in group.bundle_dict.keys() if "uniformity_map" in k][0]
    map_bundle = group.bundle_dict[resid_map_key]
    map_bundle.slicer = HealpixSlicer(nside=map_nside)
    ph = PlotHandler(results_db=results_db, out_dir=out_dir)
    ph.set_metric_bundles([map_bundle])
    ph.plot(HealpixSkyMap, plot_dicts={"percentile_clip": 98})
    ph.plot(HealpixHistogram)

    results_db.close()
