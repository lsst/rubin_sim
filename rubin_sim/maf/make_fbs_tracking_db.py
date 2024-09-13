__all__ = ("make_fbs_tracking_db",)

import argparse
import os
import sqlite3

import pandas as pd

from rubin_sim.maf.db import ResultsDb, TrackingDb, VersionRow


def make_fbs_tracking_db():
    """Make a database to track lots of opsim outputs,
    adding in comments and identifiers based on the metric subsets.
    """
    parser = argparse.ArgumentParser(
        description="Create a tracking database for many opsim maf outputs,"
        "with comments and dates. "
        "Assumes metrics are in a subdirectory called 'maf'."
    )
    parser.add_argument(
        "--opsim_list",
        type=str,
        default="opsim_list",
        help="File containing a list of all of the opsim runs to add.",
    )
    args = parser.parse_args()

    batches = {
        "meta": "General Info Metrics",
        "glance": "Quick Look Metrics",
        "sci": "Science Metrics",
        "ss": "Solar System Metrics",
        "ddf": "Deep Drilling Metrics",
    }

    tracking_db = TrackingDb()
    print(f"Tracking database in directory {tracking_db.tracking_db_dir}")

    with open(args.opsim_list, "r") as runlist:
        for run in runlist:
            db_file = run.replace("\n", "")
            vals = db_file.split("/")
            family = vals[-2]
            db_name = vals[-1]
            run_name = db_name.replace(".db", "")
            run_version = run_name.split("_10yrs")[0][-4:]
            run_group = run_version + family

            # Try to build a comment on the run based on the run_name
            run_comment = run_name.replace("_10yrs", "")[0:-4]
            run_comment = run_comment.replace("_", "")

            print(run_name, db_file)
            conn = sqlite3.connect(db_file)
            query = "select Value from info where Parameter == 'Date, ymd'"
            result = pd.read_sql(query, conn)
            sched_date = result.iloc[0, 0]
            query = "select Value from info where Parameter like 'rubin_%.__version__'"
            result = pd.read_sql(query, conn)
            sched_version = result.iloc[0, 0]

            # Look for metrics in any of the above sets
            for k in batches:
                maf_dir = os.path.join("maf", run_name + "_" + k)
                maf_comment = batches[k]
                # Get maf run date and version
                if os.path.isfile(os.path.join(maf_dir, "resultsDb_sqlite.db")):
                    resdb = ResultsDb(maf_dir)
                    resdb.open()
                    query = resdb.session.query(VersionRow).all()
                    for v in query:
                        maf_version = v.version
                        maf_date = v.run_date
                    resdb.close()

                    maf_dir = os.path.relpath(maf_dir, start=os.path.dirname(tracking_db.tracking_db_dir))
                    runId = tracking_db.add_run(
                        run_group=run_group,
                        run_name=run_name,
                        run_comment=run_comment,
                        run_version=sched_version,
                        run_date=sched_date,
                        maf_comment=maf_comment,
                        maf_version=maf_version,
                        maf_date=maf_date,
                        maf_dir=maf_dir,
                        db_file=db_file,
                    )
                    print("Used MAF RunID %d" % (runId))
