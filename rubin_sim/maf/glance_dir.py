#!/usr/bin/env python

import matplotlib

matplotlib.use("Agg")
import os
import glob
import shutil
import rubin_sim.maf.batches as batches
import rubin_sim.maf.db as db
import rubin_sim.maf.metricBundles as mb
import argparse


def glance_dir():
    """
    Run the glance batch on all .db files in a directory.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default=None)
    args = parser.parse_args()

    if args.db is None:
        db_files = glob.glob("*.db")
        db_files = [filename for filename in db_files if "trackingDb" not in filename]
    else:
        db_files = [args.db]
    run_names = [os.path.basename(name).replace(".db", "") for name in db_files]

    for filename, name in zip(db_files, run_names):
        if os.path.isdir(name + "_glance"):
            shutil.rmtree(name + "_glance")
        opsdb = db.OpsimDatabase(filename)
        colmap = batches.ColMapDict()

        bdict = {}
        bdict.update(batches.glanceBatch(colmap, name))
        results_db = db.ResultsDb(outDir=name + "_glance")
        group = mb.MetricBundleGroup(
            bdict, opsdb, outDir=name + "_glance", results_db=results_db, saveEarly=False
        )
        group.runAll(clearMemory=True, plotNow=True)
        results_db.close()
        db.addRunToDatabase(
            name + "_glance", "trackingDb_sqlite.db", None, name, "", "", name + ".db"
        )
