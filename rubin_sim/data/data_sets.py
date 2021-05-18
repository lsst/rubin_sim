import os
import warnings
import subprocess


__all__ = ['get_data_dir', 'check_and_load', 'rsync_sims_data']


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


def check_and_load(subdir, **kwargs):
    """Check if data files exist, download if needed
    """
    data_root = get_data_dir()
    path = os.path.join(data_root, subdir)
    if os.path.isdir(path):
        nfiles = len(os.listdir(path))
    else:
        nfiles = 0

    if nfiles == 0:
        rsync_sims_data(subdirs=[subdir], **kwargs)


def rsync_sims_data(data_dir=None, remote_location="lsst-rsync.ncsa.illinois.edu::sim/"):
    """Download data files from NCSA.
    Parameters
    ----------
    data_dir : str (None)
        The path to where the rubin_sim data is stored. If None, checks if it's defined by environ variable.
    remote_location : str (lsst-rsync.ncsa.illinois.edu::sim/)
        The path the download data from
    subdirs = list of str
        The names of the subdirectories to syncronize. Options are XXX
    """

    if data_dir is None:
        data_dir = get_data_dir()

    # Check that the directory exists
    if not os.path.isdir(data_dir):
        warnings.warn('Directory %s does not exist, attempting to create' % data_dir)
        os.mkdir(data_dir)

    base_call = ["rsync", "-avz", "--progress", "--delete"]

    for subdir in subdirs:
        # Check that subdir exists
        data_path = os.path.join(data_dir, subdir)
        if not os.path.isdir(data_path):
            os.mkdir(data_path)

        call = base_call + [os.path.join(remote_location, subdir)] + [data_path]
        subprocess.call(call)
