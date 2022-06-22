#!/usr/bin/env python

import argparse
from .runComparison import RunComparison


def resultdb_combine():

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--outfile", type=str, default="combined_stats.pkl")
    args = parser.parse_args()

    stats = RunComparison(baseDir=args.base_dir)()
    stats.to_pickle(args.outfile)
