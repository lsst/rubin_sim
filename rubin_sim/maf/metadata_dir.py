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

    # If runNames not given, scan for opsim databases in current directory and use those
    # Note that 'runNames' can be full path to directories

    if args.db is None:
        # Just look for any .db files in this directory
        dbFiles = glob.glob("*.db")
        # But remove trackingDb and resultsDb if they're there
        try:
            dbFiles.remove("trackingDb_sqlite.db")
        except ValueError:
            pass
        try:
            dbFiles.remove("resultsDb_sqlite.db")
        except ValueError:
            pass
    elif isinstance(args.db, str):
        dbFiles = [args.db]
    else:
        dbFiles = args.db

    sim_names = [os.path.basename(name).replace(".db", "") for name in dbFiles]

    trackingDb = TrackingDb(database=None)
    mafDate, mafVersion = getDateVersion()
    mafVersion = mafVersion["__version__"]

    for filename, opsim in zip(dbFiles, sim_names):
        # Connect to the database
        opsdb = filename
        colmap = batches.ColMapDict()

        # Set and create if needed the output directory
        outDir = opsim + "_meta"
        if not args.no_clobber:
            if os.path.isdir(outDir):
                shutil.rmtree(outDir)

        # Find the 'wfd' footprint
        m = CountExplimMetric(col="observationStartMJD")
        allsky_slicer = HealpixSlicer(nside=args.nside)
        constraint = 'note not like "%DD%"'
        bundle = MetricBundle(m, allsky_slicer, constraint, runName=opsim)
        g = MetricBundleGroup({f"{opsim} footprint": bundle}, opsdb, outDir=outDir)
        g.runAll()
        wfd_footprint = bundle.metricValues.filled(0)
        wfd_footprint = np.where(wfd_footprint > args.wfd_threshold, 1, 0)
        wfd_hpix = np.where(wfd_footprint == 1)[0]
        wfd_slicer = HealpixSubsetSlicer(nside=args.nside, hpid=wfd_hpix)

        bdict = batches.metadata_bundle_dicts(allsky_slicer, wfd_slicer, opsim, colmap)

        # Set up the resultsDB
        resultsDb = ResultsDb(outDir=outDir)
        # Go and run it
        group = MetricBundleGroup(
            bdict, opsdb, outDir=outDir, resultsDb=resultsDb, saveEarly=False
        )
        group.runAll(clearMemory=True, plotNow=True)
        resultsDb.close()

        # Add outputs to tracking database. -- note possible race condition if running in parallel.
        trackingDb.addRun(
            opsimRun=opsim,
            opsimVersion=None,
            opsimDate=None,
            mafComment="Simple",
            mafVersion=mafVersion,
            mafDate=mafDate,
            mafDir=outDir,
            dbFile=filename,
            mafRunId=None,
            opsimGroup=None,
            opsimComment=None,
        )
    # Close trackingDB
    trackingDb.close()
