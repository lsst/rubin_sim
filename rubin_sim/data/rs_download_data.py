__all__ = ("data_dict", "rs_download_data", "get_data_dir")

import argparse

from rubin_scheduler.data import DEFAULT_DATA_URL, download_rubin_data
from rubin_scheduler.data import get_data_dir as gdd


def get_data_dir():
    """For backwards compatibility since this got moved over to the scheduler."""
    return gdd()


def data_dict():
    """Creates a `dict` for all data buckets and the tar file they map to.
    To create tar files and follow any sym links, run:
        ``tar -chvzf maf_may_2021.tgz maf``

    Returns
    -------
    result : `dict`
        Data bucket filenames dictionary with keys:
        ``"name"``
            Data bucket name (`str`).
        ``"version"``
            Versioned file name (`str`).
    """
    file_dict = {
        "maf": "maf_2022_08_26.tgz",
        "maps": "maps_2022_2_28.tgz",
        "movingObjects": "movingObjects_oct_2021.tgz",
        "orbits": "orbits_2022_3_1.tgz",
        "orbits_precompute": "orbits_precompute_2023_05_23.tgz",
        "sim_baseline": "sim_baseline_2023_09_22.tgz",
        "skybrightness": "skybrightness_2023_09_11.tgz",
        "throughputs": "throughputs_2023_09_22.tgz",
        "tests": "tests_2022_10_18.tgz",
    }
    return file_dict


def rs_download_data():
    """Download data."""

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
        force=args.force,
        url_base=args.url_base,
        tdqm_disable=args.tdqm_disable,
        update=args.update,
    )
