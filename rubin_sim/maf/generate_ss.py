__all__ = ("generate_ss_commands", "generate_ss")

import argparse
import glob
import os
import sqlite3

import numpy as np
import pandas as pd

import rubin_sim.maf.db as db
from rubin_sim.data import get_data_dir


def generate_ss_commands(
    dbfiles=None,
    pops=None,
    split=False,
):
    if dbfiles is None:
        dbfiles = glob.glob("*.db")
        dbfiles.sort()

    data_dir = os.path.join(get_data_dir(), "orbits")
    orbit_files = {
        "granvik_5k": os.path.join(data_dir, "granvik_5k.txt"),
        "granvik_pha_5k": os.path.join(data_dir, "granvik_pha_5k.txt"),
        "vatiras_granvik_10k": os.path.join(data_dir, "vatiras_granvik_10k.txt"),
        "l7_5k": os.path.join(data_dir, "l7_5k.txt"),
        "mba_5k": os.path.join(data_dir, "mba_5k.txt"),
        "occ_rmax5_5k": os.path.join(data_dir, "occ_rmax5_5k.txt"),
        "occ_rmax20_5k": os.path.join(data_dir, "occ_rmax20_5k.txt"),
        "trojan_5k": os.path.join(data_dir, "trojan_5k.txt"),
    }

    pos_dir = os.path.join(get_data_dir(), "orbits_precompute")
    position_files = {
        "granvik_5k": os.path.join(pos_dir, "granvik_5k.npz"),
        "granvik_pha_5k": os.path.join(pos_dir, "granvik_pha_5k.npz"),
        "vatiras_granvik_10k": os.path.join(pos_dir, "vatiras_granvik_10k.npz"),
        "l7_5k": os.path.join(pos_dir, "l7_5k.npz"),
        "mba_5k": os.path.join(pos_dir, "mba_5k.npz"),
        "occ_rmax5_5k": os.path.join(pos_dir, "occ_rmax5_5k.npz"),
        "occ_rmax20_5k": os.path.join(pos_dir, "occ_rmax20_5k.npz"),
        "trojan_5k": os.path.join(pos_dir, "trojan_5k.npz"),
    }

    objtypes = {
        "granvik_5k": "NEO",
        "granvik_pha_5k": "PHA",
        "vatiras_granvik_10k": "Vatira",
        "l7_5k": "TNO",
        "mba_5k": "MBA",
        "occ_rmax5_5k": "OCC_r5",
        "occ_rmax20_5k": "OCC_r20",
        "sdo_5k": "SDO",
        "trojan_5k": "Trojan",
    }

    if pops is None:
        # put in order so longer runtime ones are first
        pops = [
            "trojan_5k",
            "l7_5k",
            "mba_5k",
            "granvik_5k",
            "granvik_pha_5k",
            "vatiras_granvik_10k",
            "occ_rmax5_5k",
            "occ_rmax20_5k",
        ]

    elif pops is not None:
        pp = [p for p in orbit_files.keys() if p == pops]
        if len(pp) == 0:
            print(f"Did not find population {pops} in expected types ({list(orbit_files.keys())}")
        pops = [pops]

    runs = [os.path.split(file)[-1].replace(".db", "") for file in dbfiles]
    runs = [run for run in runs if "tracking" not in run]

    # Find mjd_start for each run:
    mjd_starts = {}
    for filename in dbfiles:
        con = sqlite3.connect(filename)
        value = pd.read_sql("select min(observationStartMJD) from observations;", con)
        con.close()
        mjd_starts[filename] = np.floor(value.values.min())

    if not split:
        output_file = open("ss_script.sh", "w")
        for run, filename in zip(runs, dbfiles):
            out_dir = f"{run}_ss"
            try:
                os.mkdir(out_dir)
            except FileExistsError:
                pass
            # Create the results DB so multiple threads
            # don't try to create it later
            # This isn't used in this script, but needs to exist on-disk.
            results_db = db.ResultsDb(out_dir=out_dir)
        for pop in pops:
            for run, filename in zip(runs, dbfiles):
                objtype = objtypes[pop]

                s1 = (
                    f"make_lsst_obs --simulation_db {filename} --orbit_file {orbit_files[pop]}"
                    f" --positions_file {position_files[pop]}"
                )
                s2 = (
                    f"run_moving_calc --obs_file {run}__{pop}_obs.npz"
                    f" --simulation_db {filename} --orbit_file {orbit_files[pop]}"
                    f" --out_dir {run}_ss"
                    f" --objtype {objtype}"
                    f" --start_time {mjd_starts[filename]}"
                )
                s3 = (
                    f"run_moving_fractions --work_dir {run}_ss"
                    f" --metadata {objtype}"
                    f" --start_time {mjd_starts[filename]}"
                )
                print(s1 + " ; " + s2 + " ; " + s3, file=output_file)
    else:
        for run, filename in zip(runs, dbfiles):
            out_dir = f"{run}_ss"
            try:
                os.mkdir(out_dir)
            except FileExistsError:
                pass
            # Create the results DB so multiple threads
            # don't try to create it later
            results_db = db.ResultsDb(out_dir=out_dir)  # noqa F841
            outfile = f"{run}_ss_script.sh"
            if split:
                output_file = open(outfile, "w")
            for pop in pops:
                objtype = objtypes[pop]
                if split:
                    splitfiles = glob.glob(os.path.join(data_dir, "split") + f"/*{pop}*")
                    outfile_split = outfile.replace(".sh", f"_{pop}_split.sh")
                    # If the output split file already exists, remove it
                    # (as we append, not write)
                    if os.path.isfile(outfile_split):
                        os.remove(outfile_split)
                    for i, splitfile in enumerate(splitfiles):
                        split = os.path.split(splitfile)[-1]
                        split = split.replace(".des", "").replace(".s3m", "").replace(".txt", "")
                        with open(outfile_split, "a") as wi:
                            s1 = (
                                f"make_lsst_obs --simulation_db {filename} --orbit_file {splitfile}"
                                f" --positions_file {position_files[pop]}"
                                f" --out_dir {out_dir}"
                            )
                            s2 = (
                                f"run_moving_calc --obs_file {out_dir}/{run}__{split}_obs.npz"
                                f" --simulation_db {filename} --orbit_file {orbit_files[pop]}"
                                f" --out_dir {out_dir}/{split}"
                                f" --run_name {run}"
                                f" --objtype {objtype}"
                                f" --start_time {mjd_starts[filename]}"
                            )
                            print(s1 + " ; " + s2, file=wi)
                    s3 = (
                        f"run_moving_join --orbit_file {pop}"
                        f" --base_dir {out_dir}"
                        f" --out_dir {out_dir}/sso"
                    )
                    s4 = (
                        f"run_moving_fractions --work_dir {out_dir}/sso"
                        f" --metadata {objtype}"
                        f" --start_time {mjd_starts[filename]}"
                    )
                    print(
                        f"cat {outfile_split} | parallel -j 10 ; {s3}  ; {s4}",
                        file=output_file,
                    )
    output_file.close()


def generate_ss():
    """Generate solar system processing commands."""

    parser = argparse.ArgumentParser(description="Generate solar system processing commands")
    parser.add_argument("--db", type=str, default=None, help="database to process")
    parser.set_defaults(vatiras=False)
    parser.add_argument("--pop", type=str, default=None, help="identify one population to run")
    parser.add_argument("--start_mjd", type=float, default=None, help="start of the sim")
    parser.add_argument(
        "--split",
        dest="split",
        default=False,
        action="store_true",
        help="Split up population files; rejoin during processing",
    )
    args = parser.parse_args()

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

    generate_ss_commands(
        split=args.split,
        dbfiles=db_files,
        pops=args.pop,
    )
