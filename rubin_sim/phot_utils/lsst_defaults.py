__all__ = ("LSSTdefaults",)


class LSSTdefaults:
    """
    This class exists to store default values of seeing, m5, and gamma taken from the over
    view paper (arXiv 0805.2366, Table 2, 29 August 2014 version)
    """

    def __init__(self):
        # Standard fwhm_effective in arcseconds
        self._fwhm_eff = {
            "u": 0.92,
            "g": 0.87,
            "r": 0.83,
            "i": 0.80,
            "z": 0.78,
            "y": 0.76,
        }
        # Expected effective wavelength for throughput curves, in nanometers
        self._effwavelen = {
            "u": 367.0,
            "g": 482.5,
            "r": 622.2,
            "i": 754.5,
            "z": 869.1,
            "y": 971.0,
        }
        # Expected m5 depths (using fwhm_effective + dark sky + X=1.2 atmosphere + throughput curves)
        self._m5 = {
            "u": 23.68,
            "g": 24.89,
            "r": 24.43,
            "i": 24.00,
            "z": 24.45,
            "y": 22.60,
        }
        self._gamma = {
            "u": 0.037,
            "g": 0.038,
            "r": 0.039,
            "i": 0.039,
            "z": 0.040,
            "y": 0.040,
        }

    def m5(self, tag):
        """
        From arXiv 0805.2366  (Table 2):

        Typical 5-sigma depth for point sources at zenith, assuming
        exposure time of 2 x 15 seconds and observing conditions as listed.
        Calculated using $SYSENG_THROUGHPUT curves as of 11/25/2015, using
        $SYSENG_THROUGHPUT/python/calcM5.py

        Parameters
        ----------
        tag : `str`
            Name of a filter i.e. 'u', 'g', 'r', 'i', 'z', or 'y'

        Returns
        -------
        m5 : `float`
            Corresponding m5 value for this filter.
        """
        return self._m5[tag]

    def fwhm_eff(self, tag):
        """
        From arXiv 0805.2366 XXX version (Table 2):

        The expected fwhm_eff in arcseconds. This is the width of a single gaussian
        which produces the appropriate number of effective pixels in the PSF (thus 'fwhm_eff').
        This is the value to use for calculating Neffective, when Neffective assumes a single gaussian.
        It can be converted to a geometric FWHM (equivalent to the approximate value which would
        be measured across a van Karmen PSF profile) using SignalToNoise.fwhm_eff2FWHMgeom.

        Parameters
        ----------
        tag : `str`
            Name of a filter i.e. 'u', 'g', 'r', 'i', 'z', or 'y'

        Returns
        ----------
        fwhm_eff : `float`
            corresponding fwhm_eff
        """

        return self._fwhm_eff[tag]

    def effwavelen(self, tag):
        """
        From the throughput curves in syseng_throughputs, calculated by
        $SYSENG_THROUGHPUTS/python/effectiveWavelen.py
        as of 11/25/2015.
        """
        return self._effwavelen[tag]

    def gamma(self, tag):
        """
        See Table 2 and Equaiton 5 of arXiv 0805.2366 29 August 2014 version.

        Parameters
        ----------
        tag : `str`
            Name of a filter i.e. 'u', 'g', 'r', 'i', 'z', or 'y'

        Returns
        ----------
        gamma : `float`
            Corresponding value of gamma for this filter.
        """
        return self._gamma[tag]
