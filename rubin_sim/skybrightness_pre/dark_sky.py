import numpy as np
import os
from rubin_sim.data import get_data_dir


__all__ = ["dark_sky"]


def dark_sky():
    """Load an array of HEALpix maps that have the darkest expected sky backgrounds per filter"""
    if not hasattr(dark_sky, "data"):
        # Load up the data
        data_dir = get_data_dir()
        data = np.load(os.path.join(data_dir, "skybrightness_pre", "dark_maps.npz"))
        dark_sky.data = data["dark_maps"].copy()
        data.close()

    return dark_sky.data
