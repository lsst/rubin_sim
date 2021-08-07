"""
This file defines some test catalog and DBObject classes for use with unit tests.

To date (30 October 2014) testPhotometry.py and testCosmology.py import from this module
"""

import numpy
from rubin_sim.photUtils import calcSkyCountsPerPixelForM5, Sed

__all__ = ["setM5",
           "comovingDistanceIntegrand", "cosmologicalOmega"]


def setM5(m5target, skysed, totalBandpass, hardware,
          photParams,
          FWHMeff=None):
    """
    Take an SED representing the sky and normalize it so that
    m5 (the magnitude at which an object is detected in this
    bandpass at 5-sigma) is set to some specified value.

    The 5-sigma limiting magnitude (m5) for an observation is
    determined by a combination of the telescope and camera parameters
    (such as diameter of the mirrors and the readnoise) together with the
    sky background. This method (setM5) scales a provided sky background
    Sed so that an observation would have a target m5 value, for the
    provided hardware parameters. Using the resulting Sed in the
    'calcM5' method will return this target value for m5.

    @param [in] the desired value of m5

    @param [in] skysed is an instantiation of the Sed class representing
    sky emission

    @param [in] totalBandpass is an instantiation of the Bandpass class
    representing the total throughput of the telescope (instrumentation
    plus atmosphere)

    @param [in] hardware is an instantiation of the Bandpass class representing
    the throughput due solely to instrumentation.

    @param [in] photParams is an instantiation of the
    PhotometricParameters class that carries details about the
    photometric response of the telescope.

    @param [in] FWHMeff in arcseconds

    @param [out] returns an instantiation of the Sed class that is the skysed renormalized
    so that m5 has the desired value.

    Note that the returned SED will be renormalized such that calling the method
    self.calcADU(hardwareBandpass) on it will yield the number of counts per square
    arcsecond in a given bandpass.
    """

    #This is based on the LSST SNR document (v1.2, May 2010)
    #www.astro.washington.edu/users/ivezic/Astr511/LSST_SNRdoc.pdf

    if FWHMeff is None:
        FWHMeff = LSSTdefaults().FWHMeff('r')

    skyCountsTarget = calcSkyCountsPerPixelForM5(m5target, totalBandpass, FWHMeff=FWHMeff,
                                             photParams=photParams)

    skySedOut = Sed(wavelen=numpy.copy(skysed.wavelen),
                    flambda=numpy.copy(skysed.flambda))

    skyCounts = skySedOut.calcADU(hardware, photParams=photParams) \
                    * photParams.platescale * photParams.platescale
    skySedOut.multiplyFluxNorm(skyCountsTarget/skyCounts)

    return skySedOut


def cosmologicalOmega(redshift, H0, Om0, Ode0 = None, Og0=0.0, Onu0=0.0, w0=-1.0, wa=0.0):
    """
    A method to compute the evolution of the Hubble and density parameters
    with redshift (as a baseline against which to test the cosmology unittest)

    @param [in] redshift is the redshift at which the output is desired

    @param [in] H0 is the Hubble parameter at the present epoch in km/s/Mpc

    @param [in] Om0 is the density parameter (fraction of critical) for matter at the
    present epoch

    @param [in] Ode0 is the density parameter for Dark Energy at the present epoch.
    If left as None, will be set to 1.0-Om0-Og0-Onu0 (i.e. a flat universe)

    @param [in] Og0 is the density parameter for photons at the present epoch

    @param [in] Onu0 is the density parameter for neutrinos at the present epoch
    (assume massless neutrinos)

    @param [in] w0 is a parameter for calculating the equation of state for Dark Energy
    w = w0 + wa * z/(1 + z)

    @param [in] wa is the other parameter for calculating the equation of state for Dark
    Energy

    @returns Hubble parameter at desired redshift (in km/s/Mpc)

    @returns matter density Parameter at desired redshift

    @returns Dark Energy density parameter at desired redshift

    @returns photon density parameter at desired redshift

    @returns neutrino density parameter at desired redshift

    @returns curvature density parameter at desired redshift
    """

    if Ode0 is None:
        Ode0 = 1.0 - Om0 - Og0 - Onu0

    Ok0 = 1.0 - Om0 - Ode0 - Og0 - Onu0

    aa = 1.0/(1.0+redshift)
    Omz = Om0 * numpy.power(1.0+redshift, 3)
    Ogz = Og0 * numpy.power(1.0+redshift, 4)
    Onuz = Onu0 * numpy.power(1.0+redshift, 4)
    Okz = Ok0 * numpy.power(1.0+redshift, 2)
    Odez = Ode0 * numpy.exp(-3.0*(numpy.log(aa)*(w0 + wa +1.0) - wa*(aa - 1.0)))

    Ototal = Omz + Ogz + Onuz + Odez + Okz

    return H0*numpy.sqrt(Ototal), Omz/Ototal, Odez/Ototal, Ogz/Ototal, Onuz/Ototal, Okz/Ototal

def comovingDistanceIntegrand(redshift, H0, Om0, Ode0, Og0, Onu0, w0, wa):
    """
    The integrand of comoving distance (as a baseline for cosmology unittest)

    @param [in] redshift is the redshift at which to evaluate the integrand

    @param [in] H0 is the Hubble parameter at the present epoch in km/s/Mpc

    @param [in] Om0 is the density parameter (fraction of critical) for matter at the
    present epoch

    @param [in] Ode0 is the density parameter for Dark Energy at the present epoch.

    @param [in] Og0 is the density parameter for photons at the present epoch

    @param [in] Onu0 is the density parameter for neutrinos at the present epoch
    (assume massless neutrinos)

    @param [in] w0 is a parameter for calculating the equation of state for Dark Energy
    w = w0 + wa * z/(1 + z)

    @param [in] wa is the other parameter for calculating the equation of state for Dark
    Energy

    @returns 1/(Hubble parameter at desired redshift in km/s/Mpc)

    """
    hh, mm, de, gg, nn, kk = cosmologicalOmega(redshift, H0, Om0, Ode0=Ode0,
                                          Og0=Og0, Onu0=Onu0, w0=w0, wa=wa)
    return 1.0/hh
