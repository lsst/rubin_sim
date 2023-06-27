#!/usr/bin/env python
import os
import warnings
import requests
from requests.exceptions import ConnectionError
import argparse
from tqdm.auto import tqdm
from shutil import unpack_archive, rmtree

from .data_sets import get_data_dir, data_versions

DEFAULT_DATA_URL = "https://s3df.slac.stanford.edu/data/rubin/sim-data/rubin_sim_data/"
BACKUP_DATA_URL = (
    "https://epyc.astro.washington.edu/~lynnej/opsim_downloads/rubin_sim_data/"
)


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
        "orbits_precompute": "orbits_precompute_2023_05_23.tgz",
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
        default=DEFAULT_DATA_URL,
        help="Root URL of download location",
    )
    parser.add_argument(
        "--orbits_pre",
        dest="orbits",
        default=False,
        action="store_true",
        help="Include pre-computed orbit files.",
    )
    args = parser.parse_args()

    dirs = args.dirs
    if dirs is None:
        dirs = files.keys()
    else:
        dirs = dirs.split(",")

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

    if not args.orbits:
        dirs = [key for key in dirs if "orbits_precompute" not in key]

    # See if base URL is alive
    s = requests.Session()
    url_base = args.url_base
    try:
        r = requests.get(url_base)
        if r.status_code != requests.codes.ok:
            url_base = BACKUP_DATA_URL
    except ConnectionError:
        url_base = BACKUP_DATA_URL
    print(f"Could not connect to {args.url_base}; trying {url_base}")
    try:
        r = requests.get(url_base)
        fail_message = (
            f"Could not connect to {args.url_base} or {url_base}. Check sites are up?"
        )
    except ConnectionError:
        print(fail_message)
        exit()
    if r.status_code != requests.codes.ok:
        print(fail_message)
        exit()

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
            # stream and write in chunks (avoid large memory usage)
            r = requests.get(url, stream=True)
            file_size = int(r.headers.get("Content-Length", 0))
            block_size = (
                1024 * 1024
            )  # download this size chunk at a time; reasonable guess
            progress_bar = tqdm(total=file_size, unit="iB", unit_scale=True)
            with open(os.path.join(data_dir, filename), "wb") as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            progress_bar.close()
            # untar in place
            unpack_archive(os.path.join(data_dir, filename), data_dir)
            os.remove(os.path.join(data_dir, filename))
            versions[key] = files[key]

    # Write out the new version info to the data directory
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
