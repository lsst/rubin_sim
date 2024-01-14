__all__ = ("PhysicalParameters",)


class PhysicalParameters:
    """
    Stores physical constants and other immutable parameters
    used by the sims_phot_utils code.
    """

    def __init__(self):
        self._lightspeed = 299792458.0  # speed of light, m/s
        self._planck = 6.626068e-27  # planck's constant, ergs*seconds
        self._nm2m = 1.00e-9  # nanometers to meters conversion m/nm
        self._ergsetc2jansky = 1.00e23  # erg/cm2/s/Hz to Jansky units (fnu)

    @property
    def lightspeed(self):
        """Speed of light in meters per second."""
        return self._lightspeed

    @lightspeed.setter
    def lightspeed(self, value):
        raise RuntimeError("Cannot change the value of lightspeed " + "(Einstein does not approve)")

    @property
    def nm2m(self):
        """Conversion factor to go from nm to m."""
        return self._nm2m

    @nm2m.setter
    def nm2m(self, value):
        raise RuntimeError("Cannot change the value of nm2m")

    @property
    def ergsetc2jansky(self):
        """Conversion factor to go from ergs/sec/cm^2 to Janskys."""
        return self._ergsetc2jansky

    @ergsetc2jansky.setter
    def ergsetc2jansky(self, value):
        raise RuntimeError("Cannot change the value of ergsetc2Jansky")

    @property
    def planck(self):
        """Planck's constant in ergs*seconds."""
        return self._planck

    @planck.setter
    def planck(self, value):
        raise RuntimeError("Cannot change the value of planck")
