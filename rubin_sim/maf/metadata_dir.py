#!/usr/bin/env python

import os
import argparse
import glob
import numpy as np
import shutil
import matplotlib

matplotlib.use("Agg")

from . import batches as batches
from .metrics import CountExplimMetric
from .slicers import HealpixSlicer, HealpixSubsetSlicer
from .metricBundles import MetricBundle, MetricBundleGroup
from .utils import getDateVersion
from .db import TrackingDb, ResultsDb


def metadata_dir():
    """
    Run the metadata batch on all .db files in a directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument(
        "--nside",
        type=float,
        default=64,
        help="nside to use for the healpix slicer and subsetslicer for metrics.",
    )
    parser.add_argument(
        "--wfd_threshold",
        type=float,
        default=750,
        help="Threshold number of visits per pointing to use to define the WFD footprint."
        "Default value of 750 corresponds to the minimum median value per pointing from the SRD.",
    )
    parser.add_argument(
        "--no_clobber",
        dest="no_clobber",
        action="store_false",
        help="Do not remove existing directory outputs",
    )
    args = parser.parse_args()

    # If runNames not given, scan for sim_name databases in current directory and use those
    # Note that 'runNames' can be full path to directories

    if args.db is None:
        # Just look for any .db files in this directory
        db_files = glob.glob("*.db")
        # But remove trackingDb and results_db if they're there
        try:
            db_files.remove("trackingDb_sqlite.db")
        except ValueError:
            pass
        try:
            db_files.remove("resultsDb_sqlite.db")
        except ValueError:
            pass
    elif isinstance(args.db, str):
        db_files = [args.db]
    else:
        db_files = args.db

    sim_names = [os.path.basename(name).replace(".db", "") for name in db_files]

    for filename, sim_name in zip(db_files, sim_names):
        # Connect to the database
        colmap = batches.ColMapDict()

        # Set and create if needed the output directory
        out_dir = sim_name + "_meta"
        if not args.no_clobber:
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)

        # Find the 'wfd' footprint
        m = CountExplimMetric(col="observationStartMJD")
        allsky_slicer = HealpixSlicer(nside=args.nside)
        constraint = 'note not like "%DD%"'
        bundle = MetricBundle(m, allsky_slicer, constraint, runName=sim_name)
        g = MetricBundleGroup(
            {f"{sim_name} footprint": bundle}, filename, outDir=out_dir
        )
        g.run_all()
        wfd_footprint = bundle.metricValues.filled(0)
        wfd_footprint = np.where(wfd_footprint > args.wfd_threshold, 1, 0)
        wfd_hpix = np.where(wfd_footprint == 1)[0]
        wfd_slicer = HealpixSubsetSlicer(nside=args.nside, hpid=wfd_hpix)

        bdict = batches.metadata_bundle_dicts(
            allsky_slicer, wfd_slicer, sim_name, colmap
        )

        # Set up the resultsDB
        results_db = ResultsDb(out_dir=out_dir)
        # Go and run it
        group = MetricBundleGroup(
            bdict, opsdb, outDir=out_dir, resultsDb=results_db, saveEarly=False
        )
        group.run_all(clear_memory=True, plot_now=True)
        results_db.close()
