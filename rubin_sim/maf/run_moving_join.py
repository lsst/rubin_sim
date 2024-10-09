__all__ = ("run_moving_join",)

import argparse
import glob
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")

from . import batches as batches


def run_moving_join():
    """Join split metric outputs into a single metric output file."""
    parser = argparse.ArgumentParser(
        description="Join moving object metrics (from splits) for a particular "
        "scheduler run.  Assumes split metric files are in "
        "<orbitRoot_split> subdirectories of base_dir. "
    )
    parser.add_argument("--orbit_file", type=str, help="File containing the moving object orbits.")
    parser.add_argument(
        "--base_dir",
        type=str,
        default=".",
        help="Root directory containing split (or single) metric outputs.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for moving object metrics. Default [orbitRoot]",
    )
    args = parser.parse_args()

    if args.orbit_file is None:
        print("Must specify an orbit_file")
        exit()

    # Outputs from the metrics are generally like so:
    #  <base_dir>/<splitDir>/<metricFileName>
    # - base_dir tends to be <opsimName_orbitRoot> (
    # but is set by user when starting to generate obs.)
    # - splitDir tends to be <orbitRoot_split#>
    # (and is set by observation generation script)
    # - metricFile is <opsimName_metricName_metadata
    # (NEO/L7/etc + metadata from metric script)_MOOB.npz
    #  (the metricFileName is set by the metric generation script
    #  - run_moving_calc.py).
    #  (note that split# does not show up in the metricFileName,
    #  and is not used in run_moving_calc.py).
    #  ... this lets run_moving_calc.py easily run in parallel on
    #  multiple splits.

    # Assume splits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    splits = np.arange(0, 10, 1)
    orbit_root = args.orbit_file.replace(".txt", "").replace(".des", "").replace(".s3m", "")

    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = f"{orbit_root}"

    # Scan first splitDir for all metric files.
    tempdir = os.path.join(args.base_dir, f"{orbit_root}_{splits[0]}")
    print(f"# Joining files from {orbit_root}_[0-9]; will use {tempdir} to find metric names.")

    metricfiles = glob.glob(os.path.join(tempdir, "*MOOB.npz"))
    # Identify metric names that we want to join.
    metricNames = []
    for m in metricfiles:
        mname = os.path.split(m)[-1]
        # Hack out raw Discovery outputs.
        # We don't want to join the raw discovery files.
        # This is a hack because currently we're just pulling out
        # _Time and _N_Chances to join but there are other options.
        if "Discovery" in mname:
            if "DiscoveryTime" in mname:
                metricNames.append(mname)
            elif "DiscoveryNChances" in mname:
                metricNames.append(mname)
            elif "Magic" in mname:
                metricNames.append(mname)
            elif "HighVelocity" in mname:
                metricNames.append(mname)
            else:
                pass
        else:
            metricNames.append(mname)

    if len(metricNames) == 0:
        print(f"Could not read any metric files from {tempdir}")
        exit()

    # Create the output directory.
    if not (os.path.isdir(out_dir)):
        os.makedirs(out_dir)

    # Read and combine the metric files.
    for m in metricNames:
        b = batches.read_and_combine(orbit_root, args.base_dir, splits, m)
        b.write(out_dir=out_dir)
