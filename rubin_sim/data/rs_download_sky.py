#!/usr/bin/env python
import os
import subprocess

from . import get_data_dir


def rs_download_sky():
    """Download sky files."""

    source = "lsst-rsync.ncsa.illinois.edu::sim/sims_skybrightness_pre/h5/*"
    data_dir = get_data_dir()
    destination = os.path.join(data_dir, "skybrightness_pre")
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
        os.mkdir(destination)

    subprocess.Popen(["rsync", "-av", "--progress", source, destination + "/"]).wait()
