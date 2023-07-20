__all__ = ("get_data_dir", "data_versions", "get_baseline")

import glob
import os


def get_data_dir():
    """Get the location of the rubin_sim data directory.

    Returns
    -------
    data_dir : `str`
        Path to the rubin_sim data directory.
    """
    # See if there is an environment variable with the path
    data_dir = os.getenv("RUBIN_SIM_DATA_DIR")

    # Set the root data directory
    if data_dir is None:
        data_dir = os.path.join(os.getenv("HOME"), "rubin_sim_data")
    return data_dir


def get_baseline():
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


def data_versions():
    """Get the dictionary of source filenames in the rubin_sim data directory.

    Returns
    -------
    result : `dict`
        Data directory filenames dictionary with keys:
        ``"name"``
            Data bucket name (`str`).
        ``"version"``
            Versioned file name (`str`).
    """
    data_dir = get_data_dir()
    result = None
    version_file = os.path.join(data_dir, "versions.txt")
    if os.path.isfile(version_file):
        with open(version_file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        result = {}
        for line in content:
            ack = line.split(",")
            result[ack[0]] = ack[1]

    return result
