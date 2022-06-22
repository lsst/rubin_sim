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
    parser.add_argument("--long_micro", dest="long_micro", action="store_true")
    parser.set_defaults(long_micro=False)
    parser.add_argument(
        "--no_clobber",
        dest="no_clobber",
        action="store_false",
        help="Do not remove existing directory outputs",
    )
    args = parser.parse_args()

    if args.db is None:
        db_files = glob.glob("*.db")
        db_files = [filename for filename in db_files if "trackingDb" not in filename]
    else:
        db_files = [args.db]
    run_names = [os.path.basename(name).replace(".db", "") for name in db_files]

    for filename, name in zip(db_files, run_names):
        outDir = name + "_sci"
        # Clobber output directory if it exists
        if not args.no_clobber:
            if os.path.isdir(outDir):
                shutil.rmtree(outDir)
        opsdb = db.OpsimDatabase(filename)
        colmap = batches.ColMapDict()
        resultsDb = db.ResultsDb(outDir=outDir)
        # Set up the metricBundles
        bdict = batches.scienceRadarBatch(
            runName=name, long_microlensing=args.long_micro
        )
        # Run them, including generating plots
        group = mb.MetricBundleGroup(
            bdict, opsdb, outDir=outDir, resultsDb=resultsDb, saveEarly=False
        )
        group.runAll(clearMemory=True, plotNow=True)
        resultsDb.close()
        db.addRunToDatabase(
            outDir,
            "trackingDb_sqlite.db",
            opsimGroup=None,
            opsimRun=name,
            opsimComment=None,
            mafComment="ScienceRadar",
            dbFile=name + ".db",
        )
