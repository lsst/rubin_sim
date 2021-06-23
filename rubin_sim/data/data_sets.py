import os
import warnings
import subprocess


__all__ = ['get_data_dir', 'data_versions', 'get_baseline']


def get_data_dir():
    """Get the location of the rubin_sim data directory.

    Returns
    -------
    string that is the path to the root data directory
    """

    # See if there is an environment variable with the path
    data_dir = os.getenv('RUBIN_SIM_DATA_DIR')

    # Set the root data directory
    if data_dir is None:
        data_dir = os.path.join(os.getenv('HOME'), 'rubin_sim_data')
    return data_dir


def get_baseline():
    """Get the path to the baseline cadence simulation and the run name
    """
    dd = get_data_dir()
    path = os.path.join(dd, 'sim_baseline', 'baseline.db')
    link = os.readlink(path)
    final_path = os.path.join(dd, 'sim_baseline', link)
    return final_path


def data_versions():
    """return a dictionary of the source filenames in the data directory
    """

    data_dir = get_data_dir()
    result = None
    version_file = os.path.join(data_dir, 'versions.txt')
    if os.path.isfile(version_file):
        with open(version_file) as f:
            content = f.readlines()
        content = [x.strip() for x in content] 
        result = {}
        for line in content:
            ack = line.split(',')
            result[ack[0]] = ack[1]

    return result
