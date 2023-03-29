#!/usr/bin/env python
import os
import warnings
import requests
import argparse

from shutil import unpack_archive, rmtree

from .data_sets import get_data_dir, data_versions


def data_dict():
    """The data directories needed and what tar file they map to.

    to create tar files and follow any sym links
    tar -chvzf maf_may_2021.tgz maf
    """
    file_dict = {
        "maf": "maf_2022_08_26.tgz",
        "maps": "maps_2022_2_28.tgz",
        "movingObjects": "movingObjects_oct_2021.tgz",
        "orbits": "orbits_2022_3_1.tgz",
        "scheduler": "scheduler_2023_03_09.tgz",
        "sim_baseline": "sim_baseline_2023_01_18.tgz",
        "site_models": "site_models_2023_03_28.tgz",
        "skybrightness": "skybrightness_may_2021.tgz",
        "skybrightness_pre": "skybrightness_pre_2022_5_18.tgz",
        "throughputs": "throughputs_aug_2021.tgz",
        "tests": "tests_2022_10_18.tgz",
    }
    return file_dict


def rs_download_data():
    """Download data."""

    files = data_dict()
    parser = argparse.ArgumentParser(
        description="Download data files for rubin_sim package"
    )
    parser.add_argument(
        "--versions",
        dest="versions",
        default=False,
        action="store_true",
        help="Report expected versions, then quit",
    )
    parser.add_argument(
        "-d",
        "--dirs",
        type=str,
        default=None,
        help="Comma-separated list of directories to download",
    )
    parser.add_argument(
        "-f",
        "--force",
        dest="force",
        default=False,
        action="store_true",
        help="Force re-download of data directory(ies)",
    )
    parser.add_argument(
        "--url_base",
        type=str,
        default="https://s3df.slac.stanford.edu/data/rubin/sim-data/rubin_sim_data/",
        help="Root URL of download location",
    )
    args = parser.parse_args()

    dirs = args.dirs
    if dirs is None:
        dirs = files.keys()
    else:
        dirs = dirs.split(",")
    url_base = args.url_base
    data_dir = get_data_dir()
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    version_file = os.path.join(data_dir, "versions.txt")
    versions = data_versions()
    if versions is None:
        versions = {}

    if args.versions:
        print("Versions on disk currently // versions expected for this release:")
        match = True
        for k in files:
            print(f"{k} : {versions.get(k, '')} // {files[k]}")
            if versions.get(k, "") != files[k]:
                match = False
        if match:
            print("Versions are in sync")
            return 0
        else:
            print("Versions do not match")
            return 1

    for key in dirs:
        filename = files[key]
        path = os.path.join(data_dir, key)
        if os.path.isdir(path) and not args.force:
            warnings.warn("Directory %s already exists, skipping download" % path)
        else:
            if os.path.isdir(path) and args.force:
                rmtree(path)
                warnings.warn(
                    "Removed existing directory %s, downloading new copy" % path
                )
            # Download file
            url = url_base + filename
            print("Downloading file: %s" % url)
            r = requests.get(url)
            with open(os.path.join(data_dir, filename), "wb") as f:
                f.write(r.content)
            # untar in place
            unpack_archive(os.path.join(data_dir, filename), data_dir)
            os.remove(os.path.join(data_dir, filename))
            versions[key] = files[key]

    # Write out the new version info
    with open(version_file, "w") as f:
        for key in versions:
            print(key + "," + versions[key], file=f)

    # Write a little table to stdout
    new_versions = data_versions()
    print("Current/updated data versions:")
    for k in new_versions:
        if len(k) <= 10:
            sep = "\t\t"
        else:
            sep = "\t"
        print(f"{k}{sep}{new_versions[k]}")
