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
        opsdb = db.OpsimDatabase(filename)
        colmap = batches.ColMapDict()
        results_db = db.ResultsDb(outDir=out_dir)
        # Set up the metricBundles
        bdict = batches.scienceRadarBatch(
            runName=name,
        )
        # Run them, including generating plots
        group = mb.MetricBundleGroup(
            bdict, opsdb, out_dir=out_dir, results_db=results_db, saveEarly=False
        )
        group.runAll(clearMemory=True, plotNow=True)
        results_db.close()
        db.addRunToDatabase(
            out_dir,
            "trackingDb_sqlite.db",
            opsimGroup=None,
            opsimRun=name,
            opsimComment=None,
            mafComment="ScienceRadar",
            dbFile=name + ".db",
        )
