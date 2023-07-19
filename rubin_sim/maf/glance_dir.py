__all__ = ("glance_dir",)

import matplotlib

matplotlib.use("Agg")
import argparse
import glob
import os
import shutil

import rubin_sim.maf.batches as batches
import rubin_sim.maf.db as db
import rubin_sim.maf.metric_bundles as mb


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
        colmap = batches.col_map_dict()

        bdict = {}
        bdict.update(batches.glanceBatch(colmap, name))
        results_db = db.ResultsDb(out_dir=name + "_glance")
        group = mb.MetricBundleGroup(
            bdict,
            filename,
            out_dir=name + "_glance",
            results_db=results_db,
            save_early=False,
        )
        group.run_all(clear_memory=True, plot_now=True)
        results_db.close()
        db.add_run_to_database(
            maf_dir=name + "_glance",
            tracking_db_file="trackingDb_sqlite.db",
            run_group=None,
            run_name=name,
            run_comment="",
            maf_comment="Glance",
            db_file=name + ".db",
        )
