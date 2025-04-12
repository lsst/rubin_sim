__all__ = ("t_eff",)

from collections import defaultdict

import numpy as np

# From reference von Karman 500nm zenith seeing of 0.69"
# median zenith dark seeing from sims_skybrightness_pre
# airmass = 1
# 2 "snaps" of 15 seconds each
# m5_flat_sed sysEngVals from rubin_sim
#   commit 6d03bd49550972e48648503ed60784a4e6775b82 (2021-05-18)
# These include constants from:
#   https://github.com/lsst-pst/syseng_throughputs/blob/master/notebooks/generate_sims_values.ipynb
#   commit 7abb90951fcbc70d9c4d0c805c55a67224f9069f (2021-05-05)
# See https://github.com/lsst-sims/smtn-002/blob/master/notebooks/teff_fiducial.ipynb

# If we do not recognize the filter, return a nan. The np.array(np.nan).item
# is a hack to define a function that retruns a nan without defining a useless
# new function or violating the style guide by using a lambda function.
TEFF_FIDUCIAL_DEPTH = defaultdict(
    np.array(np.nan).item,
    {
        "u": 23.71,
        "g": 24.67,
        "r": 24.24,
        "i": 23.82,
        "z": 23.21,
        "y": 22.40,
    }.items(),
)

TEFF_FIDUCIAL_EXPTIME = 30.0


def t_eff(m5_depth, filter_name, exptime=None, fiducial_depth=None, teff_base=None, normed=False):
    """Compute the effective exposure time for a limiting magnitude.

    Parameters
    ----------
    m5_depth : `float` or `numpy.ndarray`, (N,)
        The five sigma point source limiting magintude.
    filter_name : `str` or `numpy.ndarray`, (N,)
        The name of the filter.
    exptime : `float` or `numpy.ndarray`, (N,)
        The expsore time (seconds), defaults to TEFF_FIDUCIAL_EXPTIME.
    fiducial_depth: `dict` [`str`, `float`]
        A mapping of filter to fiducial depth.
        Defaults to TEFF_FIDUCIAL_DEPTH.
    teff_base : `float`
        The exposure time (in seconds) corresponding to the exposure depth.
    normed : `bool`
        Normalize against the exposure time, such that a value of 1 corresponds
        to the exposure having been taken at fiducial conditions. Defaults
        to False.

    Returns
    -------
    t_eff : `float`
        Effective expsore time, in seconds (if normed is False) or unitless
        (if normed is true).
    """
    if fiducial_depth is None:
        fiducial_depth = TEFF_FIDUCIAL_DEPTH

    if teff_base is None:
        teff_base = TEFF_FIDUCIAL_EXPTIME

    if exptime is None:
        exptime = TEFF_FIDUCIAL_EXPTIME

    try:
        fiducial_m5_depth = fiducial_depth[filter_name]
    except TypeError:
        # If filter_name is not one value, but an iterable, map it according
        # to fiducial_depth.
        if len(filter_name) != len(m5_depth):
            raise ValueError()

        fiducial_m5_depth = np.array([fiducial_depth[b] for b in filter_name])

    t_eff = teff_base * 10.0 ** (0.8 * (m5_depth - fiducial_m5_depth))

    if normed:
        tau = t_eff / exptime
        return tau
    else:
        return t_eff
