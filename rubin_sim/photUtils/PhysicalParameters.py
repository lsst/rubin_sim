
__all__ = ["PhysicalParameters"]

class PhysicalParameters(object):
    """
    A class to store physical constants and other immutable parameters
    used by the sims_photUtils code
    """

    def __init__(self):
        #the quantities below are in nanometers
        self._minwavelen = 300.0
        self._maxwavelen = 1150.0
        self._wavelenstep = 0.1

        self._lightspeed = 299792458.0      # speed of light, = 2.9979e8 m/s
        self._planck = 6.626068e-27        # planck's constant, = 6.626068e-27 ergs*seconds
        self._nm2m = 1.00e-9               # nanometers to meters conversion = 1e-9 m/nm
        self._ergsetc2jansky = 1.00e23     # erg/cm2/s/Hz to Jansky units (fnu)

    @property
    def minwavelen(self):
        """
        minimum wavelength in nanometers
        """
        return self._minwavelen

    @minwavelen.setter
    def minwavelen(self, value):
        raise RuntimeError('Cannot change the value of minwavelen')


    @property
    def maxwavelen(self):
        """
        maximum wavelength in nanometers
        """
        return self._maxwavelen

    @maxwavelen.setter
    def maxwavelen(self, value):
        raise RuntimeError('Cannot change the value of maxwavelen')


    @property
    def wavelenstep(self):
        """
        wavelength step in nanometers
        """
        return self._wavelenstep

    @wavelenstep.setter
    def wavelenstep(self, value):
        raise RuntimeError('Cannot change the value of wavelenstep')


    @property
    def lightspeed(self):
        """
        speed of light in meters per second
        """
        return self._lightspeed

    @lightspeed.setter
    def lightspeed(self, value):
        raise RuntimeError('Cannot change the value of lightspeed ' +
                           '(Einstein does not approve)')


    @property
    def nm2m(self):
        """
        conversion factor to go from nm to m
        """
        return self._nm2m

    @nm2m.setter
    def nm2m(self, value):
        raise RuntimeError('Cannot change the value of nm2m')


    @property
    def ergsetc2jansky(self):
        """
        conversion factor to go from ergs/sec/cm^2 to Janskys
        """
        return self._ergsetc2jansky

    @ergsetc2jansky.setter
    def ergsetc2jansky(self, value):
        raise RuntimeError('Cannot change the value of ergsetc2Jansky')


    @property
    def planck(self):
        """
        Planck's constant in ergs*seconds
        """
        return self._planck

    @planck.setter
    def planck(self, value):
        raise RuntimeError('Cannot change the value of planck')
