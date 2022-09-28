"""
This file defines some test catalog and DBObject classes for use with unit tests.

To date (30 October 2014) testPhotometry.py and testCosmology.py import from this module
"""

import numpy
from rubin_sim.phot_utils import calc_sky_counts_per_pixel_for_m5, Sed

__all__ = ["set_m5", "comoving_distance_integrand", "cosmological_omega"]


def set_m5(m5target, skysed, total_bandpass, hardware, phot_params, fwhm_eff=None):
    """
    Take an SED representing the sky and normalize it so that
    m5 (the magnitude at which an object is detected in this
    bandpass at 5-sigma) is set to some specified value.

    The 5-sigma limiting magnitude (m5) for an observation is
    determined by a combination of the telescope and camera parameters
    (such as diameter of the mirrors and the readnoise) together with the
    sky background. This method (set_m5) scales a provided sky background
    Sed so that an observation would have a target m5 value, for the
    provided hardware parameters. Using the resulting Sed in the
    'calcM5' method will return this target value for m5.

    @param [in] the desired value of m5

    @param [in] skysed is an instantiation of the Sed class representing
    sky emission

    @param [in] total_bandpass is an instantiation of the Bandpass class
    representing the total throughput of the telescope (instrumentation
    plus atmosphere)

    @param [in] hardware is an instantiation of the Bandpass class representing
    the throughput due solely to instrumentation.

    @param [in] phot_params is an instantiation of the
    PhotometricParameters class that carries details about the
    photometric response of the telescope.

    @param [in] fwhm_eff in arcseconds

    @param [out] returns an instantiation of the Sed class that is the skysed renormalized
    so that m5 has the desired value.

    Note that the returned SED will be renormalized such that calling the method
    self.calcADU(hardwareBandpass) on it will yield the number of counts per square
    arcsecond in a given bandpass.
    """

    # This is based on the LSST SNR document (v1.2, May 2010)
    # www.astro.washington.edu/users/ivezic/Astr511/LSST_SNRdoc.pdf

    if fwhm_eff is None:
        fwhm_eff = LSSTdefaults().FWHMeff("r")

    sky_counts_target = calc_sky_counts_per_pixel_for_m5(
        m5target, total_bandpass, fwhm_eff=fwhm_eff, phot_params=phot_params
    )

    sky_sed_out = Sed(
        wavelen=numpy.copy(skysed.wavelen), flambda=numpy.copy(skysed.flambda)
    )

    sky_counts = (
        sky_sed_out.calc_adu(hardware, phot_params=phot_params)
        * phot_params.platescale
        * phot_params.platescale
    )
    sky_sed_out.multiply_flux_norm(sky_counts_target / sky_counts)

    return sky_sed_out


def cosmological_omega(
    redshift, h0, om0, ode0=None, og0=0.0, onu0=0.0, w0=-1.0, wa=0.0
):
    """
    A method to compute the evolution of the Hubble and density parameters
    with redshift (as a baseline against which to test the cosmology unittest)

    @param [in] redshift is the redshift at which the output is desired

    @param [in] h0 is the Hubble parameter at the present epoch in km/s/Mpc

    @param [in] om0 is the density parameter (fraction of critical) for matter at the
    present epoch

    @param [in] ode0 is the density parameter for Dark Energy at the present epoch.
    If left as None, will be set to 1.0-om0-og0-onu0 (i.e. a flat universe)

    @param [in] og0 is the density parameter for photons at the present epoch

    @param [in] onu0 is the density parameter for neutrinos at the present epoch
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

    if ode0 is None:
        ode0 = 1.0 - om0 - og0 - onu0

    ok0 = 1.0 - om0 - ode0 - og0 - onu0

    aa = 1.0 / (1.0 + redshift)
    omz = om0 * numpy.power(1.0 + redshift, 3)
    ogz = og0 * numpy.power(1.0 + redshift, 4)
    onuz = onu0 * numpy.power(1.0 + redshift, 4)
    okz = ok0 * numpy.power(1.0 + redshift, 2)
    odez = ode0 * numpy.exp(-3.0 * (numpy.log(aa) * (w0 + wa + 1.0) - wa * (aa - 1.0)))

    ototal = omz + ogz + onuz + odez + okz

    return (
        h0 * numpy.sqrt(ototal),
        omz / ototal,
        odez / ototal,
        ogz / ototal,
        onuz / ototal,
        okz / ototal,
    )


def comoving_distance_integrand(redshift, h0, om0, ode0, og0, onu0, w0, wa):
    """
    The integrand of comoving distance (as a baseline for cosmology unittest)

    @param [in] redshift is the redshift at which to evaluate the integrand

    @param [in] h0 is the Hubble parameter at the present epoch in km/s/Mpc

    @param [in] om0 is the density parameter (fraction of critical) for matter at the
    present epoch

    @param [in] ode0 is the density parameter for Dark Energy at the present epoch.

    @param [in] og0 is the density parameter for photons at the present epoch

    @param [in] onu0 is the density parameter for neutrinos at the present epoch
    (assume massless neutrinos)

    @param [in] w0 is a parameter for calculating the equation of state for Dark Energy
    w = w0 + wa * z/(1 + z)

    @param [in] wa is the other parameter for calculating the equation of state for Dark
    Energy

    @returns 1/(Hubble parameter at desired redshift in km/s/Mpc)

    """
    hh, mm, de, gg, nn, kk = cosmological_omega(
        redshift, h0, om0, ode0=ode0, og0=og0, onu0=onu0, w0=w0, wa=wa
    )
    return 1.0 / hh
