import os
import warnings
import subprocess


__all__ = ['get_data_dir']


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
