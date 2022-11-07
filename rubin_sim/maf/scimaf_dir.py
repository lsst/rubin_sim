#!/usr/bin/env python
import os
import shutil
import glob
import argparse
import matplotlib

matplotlib.use("Agg")

from . import batches as batches
from . import db as db
from . import metricBundles as mb


def scimaf_dir():
    """Run the science batch on all .db files in a directory."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument(
        "--no_clobber",
        dest="no_clobber",
        action="store_false",
        help="Do not remove existing directory outputs",
    )
    parser.set_defaults(no_long_micro=False)
    args = parser.parse_args()

    if args.db is None:
        db_files = glob.glob("*.db")
        db_files = [filename for filename in db_files if "trackingDb" not in filename]
    else:
        db_files = [args.db]
    run_names = [os.path.basename(name).replace(".db", "") for name in db_files]

    for filename, name in zip(db_files, run_names):
        out_dir = name + "_sci"
        # Clobber output directory if it exists
        if not args.no_clobber:
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
        colmap = batches.ColMapDict()
        results_db = db.ResultsDb(out_dir=out_dir)
        # Set up the metricBundles
        bdict = batches.science_radar_batch(
            runName=name,
        )
        # Run them, including generating plots
        group = mb.MetricBundleGroup(
            bdict, filename, out_dir=out_dir, results_db=results_db, save_early=False
        )
        group.run_all(clear_memory=True, plot_now=True)
        results_db.close()
        db.add_run_to_database(
            out_dir,
            "trackingDb_sqlite.db",
            run_group=None,
            opsim_run=name,
            opsim_comment=None,
            maf_comment="ScienceRadar",
            db_file=name + ".db",
        )
