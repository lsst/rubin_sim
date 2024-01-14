import glob
import os

import numpy as np

from rubin_sim.data import get_data_dir
from rubin_sim.moving_objects import DirectObs, Orbits

if __name__ == "__main__":
    """Pre-generate a series of nightly ephemerides with a 1-night timestep."""
    mjd_start = 60676.0
    length = 365.25 * 12  # How long to pre-compute for
    dtime = 1
    mjds = np.arange(mjd_start, mjd_start + length, dtime)

    orbit_files = glob.glob(os.path.join(get_data_dir(), "orbits/") + "*.txt")
    output_dir = os.path.join(os.path.join(get_data_dir(), "orbits_precompute/"))

    names = ["ra", "dec"]
    types = [float] * 2
    dt = list(zip(names, types))

    for filename in orbit_files:
        print("working on %s" % filename)
        orbits = Orbits()
        orbits.read_orbits(filename)
        # Array to hold results
        results = np.zeros((len(orbits.orbits), np.size(mjds)), dt)
        do = DirectObs()
        _temp_positions = do.generate_ephemerides(orbits, mjds, eph_mode="nbody", eph_type="basic")
        results["ra"] += _temp_positions["ra"]
        results["dec"] += _temp_positions["dec"]
        np.savez(
            os.path.join(output_dir, os.path.basename(filename).replace(".txt", ".npz")),
            positions=results,
            mjds=mjds,
        )
