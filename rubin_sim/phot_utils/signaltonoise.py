import numpy
from .sed import Sed
from .photometric_parameters import PhotometricParameters
from . import lsst_defaults


__all__ = [
    "fwhm_eff2_fwhm_geom",
    "fwhm_geom2_fwhm_eff",
    "calc_neff",
    "calc_instr_noise_sq",
    "calc_total_non_source_noise_sq",
    "calc_snr_sed",
    "calc_m5",
    "calc_sky_counts_per_pixel_for_m5",
    "calc_gamma",
    "calc_snr_m5",
    "calc_astrometric_error",
    "mag_error_from_snr",
    "calc_mag_error_m5",
    "calc_mag_error_sed",
]


def fwhm_eff2_fwhm_geom(fwhm_eff):
    """
    Convert fwhm_eff to fwhm_geom.
    This conversion was calculated by Bo Xin and Zeljko Ivezic
    (and will be in an update on the LSE-40 and overview papers).

    Parameters
    ----------
    fwhm_eff: `float`
        the single-gaussian equivalent FWHM value, appropriate for calc_neff, in arcseconds

    Returns
    -------
    fwhm_geom : `float`
        FWHM geom, the geometric FWHM value as measured from a typical PSF profile in arcseconds.
    """
    fwhm_geom = 0.822 * fwhm_eff + 0.052
    return fwhm_geom


def fwhm_geom2_fwhm_eff(fwhm_geom):
    """
    Convert fwhm_geom to fwhm_eff.
    This conversion was calculated by Bo Xin and Zeljko Ivezic
    (and will be in an update on the LSE-40 and overview papers).

    Parameters
    ----------
    fwhm_geom: `float`
        The geometric FWHM value, as measured from a typical PSF profile, in arcseconds.

    Returns
    -------
    fwhm_eff: `float`
        FWHM effective, the single-gaussian equivalent FWHM value, appropriate for calc_neff, in arcseconds.
    """
    fwhm_eff = (fwhm_geom - 0.052) / 0.822
    return fwhm_eff


def calc_neff(fwhm_eff, platescale):
    """
    Calculate the effective number of pixels in a single gaussian PSF.
    This equation comes from LSE-40, equation 27.
    https://docushare.lsstcorp.org/docushare/dsweb/ImageStoreViewer/LSE-40

    Parameters
    ----------
    fwhm_eff: `float`
        The width of a single-gaussian that produces correct Neff for typical PSF profile.
    platescale: `float`
        The platescale in arcseconds per pixel (0.2 for LSST)

    Returns
    -------
    nEff : `float`
        The effective number of pixels contained in the PSF

    The fwhm_eff is a way to represent the equivalent seeing value, if the
    atmosphere could be simply represented as a single gaussian (instead of a more
    complicated von Karman profile for the atmosphere, convolved properly with the
    telescope hardware additional blurring of 0.4").
    A translation from the geometric FWHM to the fwhm_eff is provided in fwhm_geom2_fwhm_eff.
    """
    return 2.266 * (fwhm_eff / platescale) ** 2


def calc_instr_noise_sq(phot_params):
    """
    Combine all of the noise due to intrumentation into one value

    Parameters
    ----------
    phot_params : `PhotometricParameters`
        A PhotometricParameters object that carries details about the
        photometric response of the telescope.

    Returns
    -------
    inst_noise_sq : `float`
        The noise due to all of these sources added in quadrature in ADU counts
    """

    # instrumental squared noise in electrons
    inst_noise_sq = (
        phot_params.nexp * phot_params.readnoise**2
        + phot_params.darkcurrent * phot_params.exptime * phot_params.nexp
        + phot_params.nexp * phot_params.othernoise**2
    )

    # convert to ADU counts
    inst_noise_sq = inst_noise_sq / (phot_params.gain * phot_params.gain)

    return inst_noise_sq


def calc_total_non_source_noise_sq(sky_sed, hardwarebandpass, phot_params, fwhm_eff):
    """
    Calculate the noise due to things that are not the source being observed
    (i.e. intrumentation and sky background)

    Parameters
    ----------
    sky_sed : `Sed`
        A Sed object representing the sky (normalized so that sky_sed.calc_mag() gives the sky brightness
        in magnitudes per square arcsecond)
    hardwarebandpass : `Bandpass`
        A Bandpass object containing just the instrumentation throughputs (no atmosphere)
    phot_params : `PhotometricParameters`
        A PhotometricParameters object containing information about the photometric
        properties of the telescope.
    fwhm_eff : `float`
        fwhm_eff in arcseconds

    Returns
    -------
    total_noise_sq : `float`
        total non-source noise squared (in ADU counts)
        (this is simga^2_tot * neff in equation 41 of the SNR document
        https://docushare.lsstcorp.org/docushare/dsweb/ImageStoreViewer/LSE-40 )
    """

    # Calculate the effective number of pixels for double-Gaussian PSF
    neff = calc_neff(fwhm_eff, phot_params.platescale)

    # Calculate the counts from the sky.
    # We multiply by two factors of the platescale because we expect the
    # sky_sed to be normalized such that calc_adu gives counts per
    # square arc second, and we need to convert to counts per pixel.

    skycounts = (
        sky_sed.calc_adu(hardwarebandpass, phot_params=phot_params)
        * phot_params.platescale
        * phot_params.platescale
    )

    # Calculate the square of the noise due to instrumental effects.
    # Include the readout noise as many times as there are exposures

    noise_instr_sq = calc_instr_noise_sq(phot_params=phot_params)

    # Calculate the square of the noise due to sky background poisson noise
    noise_sky_sq = skycounts / phot_params.gain

    # Discount error in sky measurement for now
    noise_skymeasurement_sq = 0

    total_noise_sq = neff * (noise_sky_sq + noise_instr_sq + noise_skymeasurement_sq)

    return total_noise_sq


def calc_sky_counts_per_pixel_for_m5(
    m5target, total_bandpass, phot_params, fwhm_eff=None
):
    """
    Calculate the number of sky counts per pixel expected for a given
    value of the 5-sigma limiting magnitude (m5)

    The 5-sigma limiting magnitude (m5) for an observation is
    determined by a combination of the telescope and camera parameters
    (such as diameter of the mirrors and the readnoise) together with the
    sky background.

    Parameters
    ----------
    m5target : `float`
        the desired value of m5
    total_bandpass : `Bandpass`
        A bandpass object representing the total throughput of the telescope
        (instrumentation plus atmosphere)
    phot_params : `PhotometricParameters`
        A photometric parameters object containing the photometric response information for Rubin
    fwhm_eff : `float`
        fwhm_eff in arcseconds

    Returns
    -------
    sky_counts_target : `float`
        the expected number of sky counts per pixel
    """

    if fwhm_eff is None:
        fwhm_eff = LSSTdefaults().fwhm_eff("r")

    # instantiate a flat SED
    flat_sed = Sed()
    flat_sed.set_flat_sed()

    # normalize the SED so that it has a magnitude equal to the desired m5
    f_norm = flat_sed.calc_flux_norm(m5target, total_bandpass)
    flat_sed.multiply_flux_norm(f_norm)
    source_counts = flat_sed.calc_adu(total_bandpass, phot_params=phot_params)

    # calculate the effective number of pixels for a double-Gaussian PSF
    neff = calc_neff(fwhm_eff, phot_params.platescale)

    # calculate the square of the noise due to the instrument
    noise_instr_sq = calc_instr_noise_sq(phot_params=phot_params)

    # now solve equation 41 of the SNR document for the neff * sigma_total^2 term
    # given snr=5 and counts as calculated above
    # SNR document can be found at
    # https://docushare.lsstcorp.org/docushare/dsweb/ImageStoreViewer/LSE-40

    n_sigma_sq = (
        source_counts * source_counts
    ) / 25.0 - source_counts / phot_params.gain

    sky_noise_target = n_sigma_sq / neff - noise_instr_sq
    sky_counts_target = sky_noise_target * phot_params.gain

    # TODO:
    # This method should throw an error if sky_counts_target is negative
    # unfortunately, that currently happens for default values of
    # m5 as taken from arXiv:0805.2366, table 2.  Adding the error
    # should probably wait for a later issue in which we hash out what
    # the units are for all of the parameters stored in PhotometricDefaults.

    return sky_counts_target


def calc_m5(skysed, total_bandpass, hardware, phot_params, fwhm_eff=None):
    """
    Calculate the AB magnitude of a 5-sigma above sky background source.

    The 5-sigma limiting magnitude (m5) for an observation is determined by
    a combination of the telescope and camera parameters (such as diameter
    of the mirrors and the readnoise) together with the sky background. This
    method (calc_m5) calculates the expected m5 value for an observation given
    a sky background Sed and hardware parameters.

    @param [in] skysed is an instantiation of the Sed class representing
    sky emission, normalized so that skysed.calc_mag gives the sky brightness
    in magnitudes per square arcsecond.

    @param [in] total_bandpass is an instantiation of the Bandpass class
    representing the total throughput of the telescope (instrumentation
    plus atmosphere)

    @param [in] hardware is an instantiation of the Bandpass class representing
    the throughput due solely to instrumentation.

    @param [in] phot_params is an instantiation of the
    PhotometricParameters class that carries details about the
    photometric response of the telescope.

    @param [in] fwhm_eff in arcseconds

    @param [out] returns the value of m5 for the given bandpass and sky SED
    """
    # This comes from equation 45 of the SNR document (v1.2, May 2010)
    # https://docushare.lsstcorp.org/docushare/dsweb/ImageStoreViewer/LSE-40

    if fwhm_eff is None:
        fwhm_eff = LSSTdefaults().fwhm_eff("r")

    # create a flat fnu source
    flatsource = Sed()
    flatsource.set_flat_sed()
    snr = 5.0
    v_n = calc_total_non_source_noise_sq(skysed, hardware, phot_params, fwhm_eff)

    counts_5sigma = (snr**2) / 2.0 / phot_params.gain + numpy.sqrt(
        (snr**4) / 4.0 / phot_params.gain + (snr**2) * v_n
    )

    # renormalize flatsource so that it has the required counts to be a 5-sigma detection
    # given the specified background
    counts_flat = flatsource.calc_adu(total_bandpass, phot_params=phot_params)
    flatsource.multiply_flux_norm(counts_5sigma / counts_flat)

    # Calculate the AB magnitude of this source.
    mag_5sigma = flatsource.calc_mag(total_bandpass)
    return mag_5sigma


def mag_error_from_snr(snr):
    """
    convert flux signal to noise ratio to an error in magnitude

    @param [in] snr is the signal to noise ratio in flux

    @param [out] the resulting error in magnitude
    """

    # see www.ucolick.org/~bolte/AY257/s_n.pdf section 3.1
    return 2.5 * numpy.log10(1.0 + 1.0 / snr)


def calc_gamma(bandpass, m5, phot_params):

    """
    Calculate the gamma parameter used for determining photometric
    signal to noise in equation 5 of the LSST overview paper
    (arXiv:0805.2366)

    @param [in] bandpass is an instantiation of the Bandpass class
    representing the bandpass for which you desire to calculate the
    gamma parameter

    @param [in] m5 is the magnitude at which a 5-sigma detection occurs
    in this Bandpass

    @param [in] phot_params is an instantiation of the
    PhotometricParameters class that carries details about the
    photometric response of the telescope.

    @param [out] gamma
    """
    # This is based on the LSST SNR document (v1.2, May 2010)
    # https://docushare.lsstcorp.org/docushare/dsweb/ImageStoreViewer/LSE-40
    # as well as equations 4-6 of the overview paper (arXiv:0805.2366)

    # instantiate a flat SED
    flat_sed = Sed()
    flat_sed.set_flat_sed()

    # normalize the SED so that it has a magnitude equal to the desired m5
    f_norm = flat_sed.calc_flux_norm(m5, bandpass)
    flat_sed.multiply_flux_norm(f_norm)
    counts = flat_sed.calc_adu(bandpass, phot_params=phot_params)

    # The expression for gamma below comes from:
    #
    # 1) Take the approximation N^2 = N0^2 + alpha S from footnote 88 in the overview paper
    # where N is the noise in flux of a source, N0 is the noise in flux due to sky brightness
    # and instrumentation, S is the number of counts registered from the source and alpha
    # is some constant
    #
    # 2) Divide by S^2 and demand that N/S = 0.2 for a source detected at m5. Solve
    # the resulting equation for alpha in terms of N0 and S5 (the number of counts from
    # a source at m5)
    #
    # 3) Substitute this expression for alpha back into the equation for (N/S)^2
    # for a general source.  Re-factor the equation so that it looks like equation
    # 5 of the overview paper (note that x = S5/S).  This should give you gamma = (N0/S5)^2
    #
    # 4) Solve equation 41 of the SNR document for the neff * sigma_total^2 term
    # given snr=5 and counts as calculated above.  Note that neff * sigma_total^2
    # is N0^2 in the equation above
    #
    # This should give you

    gamma = 0.04 - 1.0 / (counts * phot_params.gain)

    return gamma


def calc_snr_m5(magnitude, bandpass, m5, phot_params, gamma=None):
    """
    Calculate signal to noise in flux using the model from equation (5) of arXiv:0805.2366

    @param [in] magnitude of the sources whose signal to noise you are calculating
    (can be a numpy array)

    @param [in] bandpass (an instantiation of the class Bandpass) in which the magnitude
    was calculated

    @param [in] m5 is the 5-sigma limiting magnitude for the bandpass

    @param [in] phot_params is an instantiation of the
    PhotometricParameters class that carries details about the
    photometric response of the telescope.

    @param [in] gamma (optional) is the gamma parameter from equation(5) of
    arXiv:0805.2366.  If not provided, this method will calculate it.

    @param [out] snr is the signal to noise ratio corresponding to
    the input magnitude.

    @param [out] gamma is  the calculated gamma parameter for the
    bandpass used here (in case the user wants to call this method again).

    Note: You can also pass in a numpy array of magnitudes calculated
    in the same bandpass with the same m5 and get a numpy array of SNR out.
    """

    if gamma is None:
        gamma = calc_gamma(bandpass, m5, phot_params=phot_params)

    dummy_sed = Sed()
    m5_flux = dummy_sed.flux_from_mag(m5)
    source_flux = dummy_sed.flux_from_mag(magnitude)

    flux_ratio = m5_flux / source_flux

    noise = numpy.sqrt((0.04 - gamma) * flux_ratio + gamma * flux_ratio * flux_ratio)

    return 1.0 / noise, gamma


def calc_mag_error_m5(magnitude, bandpass, m5, phot_params, gamma=None):
    """
    Calculate magnitude error using the model from equation (5) of arXiv:0805.2366

    @param [in] magnitude of the source whose error you want
    to calculate (can be a numpy array)

    @param [in] bandpass (an instantiation of the Bandpass class) in question

    @param [in] m5 is the 5-sigma limiting magnitude in that bandpass

    @param [in] phot_params is an instantiation of the
    PhotometricParameters class that carries details about the
    photometric response of the telescope.

    @param [in] gamma (optional) is the gamma parameter from equation(5) of
    arXiv:0805.2366.  If not provided, this method will calculate it.

    @param [out] the error associated with the magnitude

    @param [out] gamma is  the calculated gamma parameter for the
    bandpass used here (in case the user wants to call this method again).

    Note: you can also pass in a numpy of array of magnitudes calculated in
    the same Bandpass with the same m5 and get a numpy array of errors out.
    """

    snr, gamma = calc_snr_m5(magnitude, bandpass, m5, phot_params, gamma=gamma)

    if phot_params.sigma_sys is not None:
        return (
            numpy.sqrt(
                numpy.power(mag_error_from_snr(snr), 2)
                + numpy.power(phot_params.sigma_sys, 2)
            ),
            gamma,
        )
    else:
        return mag_error_from_snr(snr), gamma


def calc_snr_sed(
    source_sed,
    totalbandpass,
    skysed,
    hardwarebandpass,
    phot_params,
    fwhm_eff,
    verbose=False,
):
    """
    Calculate the signal to noise ratio for a source, given the bandpass(es) and sky SED.

    For a given source, sky sed, total bandpass and hardware bandpass, as well as
    fwhm_eff / exptime, calculates the SNR with optimal PSF extraction
    assuming a double-gaussian PSF.

    @param [in] source_sed is an instantiation of the Sed class containing the SED of
    the object whose signal to noise ratio is being calculated

    @param [in] totalbandpass is an instantiation of the Bandpass class
    representing the total throughput (system + atmosphere)

    @param [in] skysed is an instantiation of the Sed class representing
    the sky emission per square arcsecond.

    @param [in] hardwarebandpass is an instantiation of the Bandpass class
    representing just the throughput of the system hardware.

    @param [in] phot_params is an instantiation of the
    PhotometricParameters class that carries details about the
    photometric response of the telescope.

    @param [in] fwhm_eff in arcseconds

    @param [in] verbose is a `bool`

    @param [out] signal to noise ratio
    """

    # Calculate the counts from the source.
    sourcecounts = source_sed.calc_adu(totalbandpass, phot_params=phot_params)

    # Calculate the (square of the) noise due to signal poisson noise.
    noise_source_sq = sourcecounts / phot_params.gain

    non_source_noise_sq = calc_total_non_source_noise_sq(
        skysed, hardwarebandpass, phot_params, fwhm_eff
    )

    # Calculate total noise
    noise = numpy.sqrt(noise_source_sq + non_source_noise_sq)
    # Calculate the signal to noise ratio.
    snr = sourcecounts / noise
    if verbose:
        skycounts = skysed.calc_adu(hardwarebandpass, phot_params) * (
            phot_params.platescale**2
        )
        noise_sky_sq = skycounts / phot_params.gain
        neff = calc_neff(fwhm_eff, phot_params.platescale)
        noise_instr_sq = calc_instr_noise_sq(phot_params)

        print("For Nexp %.1f of time %.1f: " % (phot_params.nexp, phot_params.exptime))
        print(
            "Counts from source: %.2f  Counts from sky: %.2f"
            % (sourcecounts, skycounts)
        )
        print("fwhm_eff: %.2f('')  Neff pixels: %.3f(pix)" % (fwhm_eff, neff))
        print(
            "Noise from sky: %.2f Noise from instrument: %.2f"
            % (numpy.sqrt(noise_sky_sq), numpy.sqrt(noise_instr_sq))
        )
        print("Noise from source: %.2f" % (numpy.sqrt(noise_source_sq)))
        print(
            " Total Signal: %.2f   Total Noise: %.2f    SNR: %.2f"
            % (sourcecounts, noise, snr)
        )
        # Return the signal to noise value.
    return snr


def calc_mag_error_sed(
    source_sed,
    totalbandpass,
    skysed,
    hardwarebandpass,
    phot_params,
    fwhm_eff,
    verbose=False,
):
    """
    Calculate the magnitudeError for a source, given the bandpass(es) and sky SED.

    For a given source, sky sed, total bandpass and hardware bandpass, as well as
    fwhm_eff / exptime, calculates the SNR with optimal PSF extraction
    assuming a double-gaussian PSF.

    @param [in] source_sed is an instantiation of the Sed class containing the SED of
    the object whose signal to noise ratio is being calculated

    @param [in] totalbandpass is an instantiation of the Bandpass class
    representing the total throughput (system + atmosphere)

    @param [in] skysed is an instantiation of the Sed class representing
    the sky emission per square arcsecond.

    @param [in] hardwarebandpass is an instantiation of the Bandpass class
    representing just the throughput of the system hardware.

    @param [in] phot_params is an instantiation of the
    PhotometricParameters class that carries details about the
    photometric response of the telescope.

    @param [in] fwhm_eff in arcseconds

    @param [in] verbose is a `bool`

    @param [out] magnitude error
    """

    snr = calc_snr_sed(
        source_sed,
        totalbandpass,
        skysed,
        hardwarebandpass,
        phot_params,
        fwhm_eff,
        verbose=verbose,
    )

    if phot_params.sigma_sys is not None:
        return numpy.sqrt(
            numpy.power(mag_error_from_snr(snr), 2)
            + numpy.power(phot_params.sigma_sys, 2)
        )
    else:
        return mag_error_from_snr(snr)


def calc_astrometric_error(mag, m5, fwhm_geom=0.7, nvisit=1, systematic_floor=10):
    """
    Calculate an expected astrometric error.
    Can be used to estimate this for general catalog purposes (use typical FWHM and n_visit=Number of visit).
    Or can be used for a single visit, use actual FWHM and n_visit=1.

    Parameters
    ----------
    mag: `float`
        Magnitude of the source
    m5: `float`
        Point source five sigma limiting magnitude of the image (or typical depth per image).
    fwhm_geom: `float`, optional
        The geometric (physical) FWHM of the image, in arcseconds. Default 0.7.
    nvisit: `int`, optional
        The number of visits/measurement. Default 1.
        If this is >1, the random error contribution is reduced by sqrt(nvisits).
    systematic_floor: `float`, optional
        The systematic noise floor for the astrometric measurements, in mas. Default 10mas.

    Returns
    -------
    astrom_err : `float`
        Astrometric error for a given SNR, in mas.
    """
    # The astrometric error can be applied to parallax or proper motion (for n_visit>1).
    # If applying to proper motion, should also divide by the # of years of the survey.
    # This is also referenced in the astroph/0805.2366 paper.
    # D. Monet suggests sqrt(Nvisit/2) for first 3 years, sqrt(N) for longer, in reduction of error
    # because of the astrometric measurement method, the systematic and random error are both reduced.
    # Zeljko says 'be conservative', so removing this reduction for now.
    rgamma = 0.039
    xval = numpy.power(10, 0.4 * (mag - m5))
    # The average fwhm_eff is 0.7" (or 700 mas), but user can specify. Convert to mas.
    seeing = fwhm_geom * 1000.0
    error_rand = seeing * numpy.sqrt((0.04 - rgamma) * xval + rgamma * xval * xval)
    error_rand = error_rand / numpy.sqrt(nvisit)
    # The systematic error floor in astrometry (mas).
    error_sys = systematic_floor
    astrom_error = numpy.sqrt(error_sys * error_sys + error_rand * error_rand)
    return astrom_error
