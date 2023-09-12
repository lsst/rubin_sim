import glob
import os

import h5py
import numpy as np

from rubin_sim.data import get_data_dir

if __name__ == "__main__":
    # Gererate an sky map for each filters that is an estimate of how faint that part of sky
    # can get

    data_dir = get_data_dir()

    sky_files = glob.glob(os.path.join(data_dir, "skybrightness_pre/*.h5"))

    maximum_maps = {}
    filternames = "ugrizy"
    for filtername in filternames:
        maximum_maps[filtername] = []

    for skyfile in sky_files:
        h5 = h5py.File(skyfile, "r")
        for filtername in filternames:
            maximum_maps[filtername].append(np.nanmax(h5["sky_mags"][filtername], axis=0))
        h5.close()

    for filtername in filternames:
        maximum_maps[filtername] = np.nanmax(maximum_maps[filtername], axis=0)
    # convert dict to array
    dark_maps = np.empty(np.size(maximum_maps["r"]), dtype=list(zip(filternames, [float] * 6)))
    for filtername in filternames:
        dark_maps[filtername] = maximum_maps[filtername]
    np.savez(os.path.join(data_dir, "skybrightness_pre", "dark_maps.npz"), dark_maps=dark_maps)
