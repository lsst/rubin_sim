__all__ = ("wrap_ra", "robust_rms", "recalc_mags")

import glob
import os

import numpy as np
from rubin_scheduler.data import get_data_dir

from rubin_sim.phot_utils import Bandpass, Sed


def wrap_ra(ra):
    """
    Wrap only RA values into 0-2pi (using mod).
    """
    ra = ra % (2.0 * np.pi)
    return ra


def robust_rms(array, missing=0.0):
    """
    Use the interquartile range to compute a robust approximation of the RMS.
    if passed an array smaller than 2 elements, return missing value
    """
    if np.size(array) < 2:
        rms = missing
    else:
        iqr = np.percentile(array, 75) - np.percentile(array, 25)
        rms = iqr / 1.349  # approximation
    return rms


def spec2mags(spectra_list, wave):
    """Convert sky spectra to magnitudes"""
    # Load LSST filters
    through_path = os.path.join(get_data_dir(), "throughputs/baseline")
    keys = ["u", "g", "r", "i", "z", "y"]

    dtype = [("mags", "float", (6))]
    result = np.zeros(len(spectra_list), dtype=dtype)

    filters = {}
    for filtername in keys:
        bp = np.loadtxt(
            os.path.join(through_path, "total_" + filtername + ".dat"),
            dtype=list(zip(["wave", "trans"], [float] * 2)),
        )
        temp_b = Bandpass()
        temp_b.set_bandpass(bp["wave"], bp["trans"])
        filters[filtername] = temp_b

    filterwave = np.array([filters[f].calc_eff_wavelen()[0] for f in keys])

    for i, spectrum in enumerate(spectra_list):
        tempSed = Sed()
        tempSed.set_sed(wave, flambda=spectrum)
        for j, filtName in enumerate(keys):
            try:
                result["mags"][i][j] = tempSed.calc_mag(filters[filtName])
            except ValueError:
                pass
    return result, filterwave


def recalc_mags(data_dir=None):
    """Recalculate the magnitudes for sky brightness components.

    DANGER:  Overwrites data files in place. The rubin_sim_data/skybrightness
    folder will need to be packaged and updated after running this to propagate
    changes to other users.
    """
    dirs = ["Airglow", "MergedSpec", "ScatteredStarLight", "Zodiacal", "LowerAtm", "Moon", "UpperAtm"]

    if data_dir is None:
        data_dir = get_data_dir()

    full_paths = [os.path.join(data_dir, "skybrightness/ESO_Spectra", dirname) for dirname in dirs]
    for path in full_paths:
        files = glob.glob(os.path.join(path, "*.npz"))
        for filename in files:
            data = np.load(filename)

            spec = data["spec"].copy()
            wave = data["wave"].copy()
            data.close()
            new_mags, filterwave = spec2mags(spec["spectra"], wave)
            spec["mags"] = new_mags["mags"]

            np.savez(filename, wave=wave, spec=spec, filterWave=filterwave)

    pass
