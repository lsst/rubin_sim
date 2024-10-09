__all__ = ("scimaf_dir",)

import argparse
import glob
import os
import shutil
import sqlite3
import warnings

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from . import batches as batches
from . import db as db
from . import metricBundles as mmB


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
    parser.add_argument("--limited", dest="limited", action="store_true")
    parser.set_defaults(limited=False)
    args = parser.parse_args()

    if args.db is None:
        db_files = glob.glob("*.db")
        db_files = [filename for filename in db_files if "trackingDb" not in filename]
    else:
        db_files = [args.db]
    run_names = [os.path.basename(name).replace(".db", "") for name in db_files]

    for filename, name in zip(db_files, run_names):
        out_dir = name + "_sci"

        # Grab the starting date for the Presto KNe metric
        try:
            con = sqlite3.connect(filename)
            mjd0_df = pd.read_sql("select min(observationStartMJD) from observations;", con)
            con.close()
            mjd0 = mjd0_df.values.min()
        # If this fails for any reason (aka schema change)
        except:  # noqa E722
            warnings.warn("Could not find survey start date for Presto KNe, setting mjd0=None.")
            mjd0 = None
        # Clobber output directory if it exists
        if not args.no_clobber:
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
        results_db = db.ResultsDb(out_dir=out_dir)
        # Set up the metricBundles
        if args.limited:
            bdict = batches.radar_limited(
                runName=name,
                mjd0=mjd0,
            )
        else:
            bdict = batches.science_radar_batch(
                runName=name,
                mjd0=mjd0,
            )
        # Run them, including generating plots
        group = mmB.MetricBundleGroup(
            bdict, filename, out_dir=out_dir, results_db=results_db, save_early=False
        )
        group.run_all(clear_memory=True, plot_now=True)
        results_db.close()
        db.add_run_to_database(
            out_dir,
            "trackingDb_sqlite.db",
            run_group=None,
            run_name=name,
            run_comment=None,
            maf_comment="ScienceRadar",
            db_file=name + ".db",
        )
