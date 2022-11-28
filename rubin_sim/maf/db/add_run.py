#!/usr/bin/env python

import argparse
from . import add_run_to_database


def add_run():

    parser = argparse.ArgumentParser(
        description="Add a MAF run to the tracking database."
    )
    parser.add_argument("maf_dir", type=str, help="Directory containing MAF outputs.")
    parser.add_argument(
        "-c", "--maf_comment", type=str, default=None, help="Comment on MAF analysis."
    )
    parser.add_argument("--group", type=str, default=None, help="Opsim Group name.")
    parser.add_argument("--run_name", type=str, default=None, help="Run Name.")
    parser.add_argument(
        "--run_comment", type=str, default=None, help="Comment on OpSim run."
    )
    parser.add_argument(
        "--db_file", type=str, default="None", help="Opsim Sqlite filename"
    )
    defaultdb = "trackingDb_sqlite.db"
    parser.add_argument(
        "-t",
        "--tracking_db",
        type=str,
        default=defaultdb,
        help="Tracking database filename. Default is %s, in the current directory."
        % defaultdb,
    )
    args = parser.parse_args()

    add_run_to_database(
        args.maf_dir,
        args.tracking_db,
        args.group,
        args.run_name,
        args.run_comment,
        args.maf_comment,
        args.db_file,
    )
