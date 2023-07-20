__all__ = ("get_imsim_flux_norm",)

import numpy as np

from .bandpass import Bandpass


def get_imsim_flux_norm(sed, magmatch):
    """
    Calculate the flux normalization of an SED in the imsim bandpass.

    Parameters
    -----------
    sed is the SED to be normalized

    magmatch is the desired magnitude in the imsim bandpass

    Returns
    --------
    The factor by which the flux of sed needs to be multiplied to achieve
    the desired magnitude.
    """

    # This method works based on the assumption that the imsim bandpass
    # is a delta function.  If that ever ceases to be true, the unit test
    # testSedUtils.py, which checks that the results of this method are
    # identical to calling Sed.calcFluxNorm and passing in the imsim bandpass,
    # will fail and we will know to modify this method.

    if not hasattr(get_imsim_flux_norm, "imsim_wavelen"):
        bp = Bandpass()
        bp.imsim_bandpass()
        non_zero_dex = np.where(bp.sb > 0.0)[0][0]
        get_imsim_flux_norm.imsim_wavelen = bp.wavelen[non_zero_dex]

    if sed.fnu is None:
        sed.flambda_tofnu()

    if (
        get_imsim_flux_norm.imsim_wavelen < sed.wavelen.min()
        or get_imsim_flux_norm.imsim_wavelen > sed.wavelen.max()
    ):
        raise RuntimeError(
            "Cannot normalize sed "
            "at wavelength of %e nm\n" % get_imsim_flux_norm.imsim_wavelen
            + "The SED does not cover that wavelength\n"
            + "(Covers %e < lambda %e)" % (sed.wavelen.min(), sed.wavelen.max())
        )

    mag = -2.5 * np.log10(np.interp(get_imsim_flux_norm.imsim_wavelen, sed.wavelen, sed.fnu)) - sed.zp
    dmag = magmatch - mag
    return np.power(10, (-0.4 * dmag))
