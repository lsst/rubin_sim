__all__ = ("get_data_dir", "get_baseline", "data_dict", "rs_download_testing", "rs_download_data")

import argparse
import glob
import os

from rubin_scheduler.data import DEFAULT_DATA_URL, download_rubin_data
from rubin_scheduler.data import get_data_dir as gdd


def get_data_dir() -> str:
    """Wraps rubin_scheduler.data.get_data_dir().
    Provided here for backwards compatibility.

    Returns
    -------
    $RUBIN_SIM_DATA_DIR : `str`
        Directory containing the necessary data for rubin_sim_data.
    """
    return gdd()


def get_baseline() -> str:
    """Get the path to the baseline cadence simulation sqlite file.

    Returns
    -------
    file : `str`
        Path to the baseline cadence simulation sqlite file.
    """
    dd = get_data_dir()
    path = os.path.join(dd, "sim_baseline")
    file = glob.glob(path + "/*10yrs.db")[0]
    return file


def data_dict() -> dict:
    """
    Dictionary containing expected version information for rubin_sim_data
    data sets, for this version of rubin_sim.

    Returns
    -------
    file_dict : `dict`
        Data bucket filenames dictionary with keys:
        ``"name"``
        Data bucket name (`str`).
        ``"version"``
        Versioned file name (`str`).
    """
    # Note for developers:
    # to create tar files and follow any sym links, run: e.g.
    #  ``tar -chvzf maf_may_2021.tgz maf``
    file_dict = {
        "maf": "maf_2024_06_13.tgz",
        "maps": "maps_2022_2_28.tgz",
        "movingObjects": "movingObjects_oct_2021.tgz",
        "orbits": "orbits_2022_3_1.tgz",
        "orbits_precompute": "orbits_precompute_2023_05_23.tgz",
        "sim_baseline": "sim_baseline_2024_10_08.tgz",
        "skybrightness": "skybrightness_2023_09_11.tgz",
        "throughputs": "throughputs_2023_09_22.tgz",
        "tests": "tests_2024_04_23.tgz",
    }
    return file_dict


def rs_download_testing() -> None:
    """Convenience function for github actions, to download only a subset
    of data.

    Contains maf,maps,movingObjects,skybrightness,throughputs and tests.
    For users who don't care about moving objects, this is likely sufficient.
    """
    all_files = data_dict()
    keys = ["maf", "maps", "movingObjects", "skybrightness", "throughputs", "tests"]
    subset = {}
    for k in keys:
        subset[k] = all_files[k]

    download_rubin_data(
        subset,
        dirs=None,
        print_versions_only=False,
        update=True,
        force=False,
        url_base=DEFAULT_DATA_URL,
        tdqm_disable=True,
    )


def rs_download_data() -> None:
    """Utility to download necessary data for rubin_sim.

    Wrapper around rubin_scheduler.scheduler_download_data,
    but downloading the data files specified by rubin_sim.
    """

    files = data_dict()
    parser = argparse.ArgumentParser(description="Download data files for rubin_sim package")
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
        "--tdqm_disable",
        dest="tdqm_disable",
        default=False,
        action="store_true",
        help="Turn off tdqm progress bar",
    )
    parser.add_argument(
        "--update",
        dest="update",
        default=False,
        action="store_true",
        help="Update versions of data on disk to match current",
    )

    args = parser.parse_args()

    download_rubin_data(
        files,
        dirs=args.dirs,
        print_versions_only=args.versions,
        update=args.update,
        force=args.force,
        url_base=args.url_base,
        tdqm_disable=args.tdqm_disable,
    )
