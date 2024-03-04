__all__ = (
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
    "scale_sky_m5",
)

import numpy

from .sed import Sed


def fwhm_eff2_fwhm_geom(fwhm_eff):
    """Convert fwhm_eff to fwhm_geom.

    This conversion was calculated by Bo Xin and Zeljko Ivezic
    (and will be in an update on the LSE-40 and overview papers).

    Parameters
    ----------
    fwhm_eff: `float`
        the single-gaussian equivalent FWHM value, appropriate for calc_neff,
        in arcseconds

    Returns
    -------
    fwhm_geom : `float`
        FWHM geom, the geometric FWHM value as measured from a typical
        PSF profile in arcseconds.
    """
    fwhm_geom = 0.822 * fwhm_eff + 0.052
    return fwhm_geom


def fwhm_geom2_fwhm_eff(fwhm_geom):
    """Convert fwhm_geom to fwhm_eff.

    This conversion was calculated by Bo Xin and Zeljko Ivezic
    (and will be in an update on the LSE-40 and overview papers).

    Parameters
    ----------
    fwhm_geom: `float`
        The geometric FWHM value, as measured from a typical PSF profile,
         in arcseconds.

    Returns
    -------
    fwhm_eff: `float`
        FWHM effective, the single-gaussian equivalent FWHM value,
        appropriate for calc_neff, in arcseconds.
    """
    fwhm_eff = (fwhm_geom - 0.052) / 0.822
    return fwhm_eff


def calc_neff(fwhm_eff, platescale):
    """Calculate the effective number of pixels in a single gaussian PSF.

    This equation comes from LSE-40, equation 27.
    https://docushare.lsstcorp.org/docushare/dsweb/ImageStoreViewer/LSE-40

    Parameters
    ----------
    fwhm_eff: `float`
        The width of a single-gaussian that produces correct
        Neff for typical PSF profile.
    platescale: `float`
        The platescale in arcseconds per pixel (0.2 for LSST)

    Returns
    -------
    nEff : `float`
        The effective number of pixels contained in the PSF

    Notes
    -----
    The fwhm_eff is a way to represent the equivalent seeing value, if the
    atmosphere could be simply represented as a single gaussian (instead of
    a more complicated von Karman profile for the atmosphere, convolved
    properly with the telescope hardware additional blurring of 0.4").
    A translation from the geometric FWHM to the fwhm_eff is provided
    in fwhm_geom2_fwhm_eff.
    """
    return 2.266 * (fwhm_eff / platescale) ** 2


def calc_instr_noise_sq(phot_params):
    """Combine all of the noise due to intrumentation into one value

    Parameters
    ----------
    phot_params : `rubin_sim.phot_utils.PhotometricParameters`
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
    """Calculate the noise due to instrumentation and sky background.

    Parameters
    ----------
    sky_sed : `rubin_sim.phot_utils.Sed`
        A Sed object representing the sky (normalized so that
        sky_sed.calc_mag() gives the sky brightness in
        magnitudes per square arcsecond)
    hardwarebandpass : `rubin_sim.phot_utils.Bandpass`
        A Bandpass object containing just the instrumentation
        throughputs (no atmosphere)
    phot_params : `rubin_sim.phot_utils.PhotometricParameters`
        A PhotometricParameters object containing information
        about the photometric properties of the telescope.
    fwhm_eff : `float`
        fwhm_eff in arcseconds

    Returns
    -------
    total_noise_sq : `float`
        total non-source noise squared (in ADU counts)
        (this is simga^2_tot * neff in equation 41 of the SNR document
        https://ls.st/LSE-40 )
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


def calc_sky_counts_per_pixel_for_m5(m5target, total_bandpass, phot_params, fwhm_eff=0.83):
    """Calculate the skycounts per pixel.

    Calculate the number of sky counts per pixel expected for a given
    value of the 5-sigma limiting magnitude (m5)

    Parameters
    ----------
    m5target : `float`
        The desired value of m5.
    total_bandpass : `rubin_sim.phot_utils.Bandpass`
        A bandpass object representing the total throughput of the telescope
        (instrumentation plus atmosphere).
    phot_params : `rubin_sim.phot_utils.PhotometricParameters`
        A photometric parameters object containing the photometric response
        information for Rubin.
    fwhm_eff : `float`
        fwhm_eff in arcseconds. Default 0.83

    Returns
    -------
    sky_counts_target : `float`
        The expected number of sky counts per pixel (ADU/pixel).

    Notes
    -----
    The 5-sigma limiting magnitude (m5) for an observation is
    determined by a combination of the telescope and camera parameters
    (such as diameter of the mirrors and the readnoise) together with the
    sky background.
    """

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

    # now solve equation 41 of the SNR document for the neff * sigma_total^2
    # term given snr=5 and counts as calculated above
    # SNR document can be found at
    # https://docushare.lsstcorp.org/docushare/dsweb/ImageStoreViewer/LSE-40

    n_sigma_sq = (source_counts * source_counts) / 25.0 - source_counts / phot_params.gain

    sky_noise_target = n_sigma_sq / neff - noise_instr_sq
    sky_counts_target = sky_noise_target * phot_params.gain

    # TODO:
    # This method should throw an error if sky_counts_target is negative
    # unfortunately, that currently happens for default values of
    # m5 as taken from arXiv:0805.2366, table 2.  Adding the error
    # should probably wait for a later issue in which we hash out what
    # the units are for all of the parameters stored in PhotometricDefaults.

    return sky_counts_target


def calc_m5(skysed, total_bandpass, hardware, phot_params, fwhm_eff=0.83):
    """Calculate the AB magnitude of a 5-sigma source above sky background.

    Parameters
    ----------
    skysed : `rubin_sim.phot_utils.Sed`
        An SED representing the sky background emission, normalized such that
        skysed.calc_mag(Bandpass) returns the expected sky brightness in
        magnitudes per sq arcsecond.
    total_bandpass : `rubin_sim.phot_utils.Bandpass`
        The Bandpass representing the total throughput of the telescope
        (instrument plus atmosphere).
    hardware : `rubin_sim.phot_utils.Bandpass`
        The Bandpass representing the throughput of the telescope instrument
        only (no atmosphere).
    phot_params : `rubin_sim.phot_utils.PhotometricParameters`
        The PhotometricParameters class that carries details about the
        photometric response of the telescope.
    fwhm_eff : `float`
        FWHM in arcseconds.

    Returns
    -------
    mag_5sigma : `float`
        The value of m5 for the given bandpass and sky SED

    Notes
    -----
    The 5-sigma limiting magnitude (m5) for an observation is determined by
    a combination of the telescope and camera parameters (such as diameter
    of the mirrors and the readnoise) together with the sky background. This
    method (calc_m5) calculates the expected m5 value for an observation given
    a sky background Sed and hardware parameters.

    This comes from equation 45 of the SNR document (v1.2, May 2010)
    https://docushare.lsstcorp.org/docushare/dsweb/ImageStoreViewer/LSE-40
    """

    # create a flat fnu source
    flatsource = Sed()
    flatsource.set_flat_sed()
    snr = 5.0
    v_n = calc_total_non_source_noise_sq(skysed, hardware, phot_params, fwhm_eff)

    counts_5sigma = (snr**2) / 2.0 / phot_params.gain + numpy.sqrt(
        (snr**4) / 4.0 / phot_params.gain + (snr**2) * v_n
    )

    # renormalize flatsource so that it has the required counts to be a
    # 5-sigma detection given the specified background
    counts_flat = flatsource.calc_adu(total_bandpass, phot_params=phot_params)
    flatsource.multiply_flux_norm(counts_5sigma / counts_flat)

    # Calculate the AB magnitude of this source.
    mag_5sigma = flatsource.calc_mag(total_bandpass)
    return mag_5sigma


def mag_error_from_snr(snr):
    """Convert flux signal to noise ratio to an error in magnitude.

    Parameters
    ----------
    snr : `float`
        The signal to noise ratio (a flux-related measurement).

    Returns
    -------
    mag_error : `float`
        Corresponding error in magnitude.
    """

    # see www.ucolick.org/~bolte/AY257/s_n.pdf section 3.1
    return 2.5 * numpy.log10(1.0 + 1.0 / snr)


def calc_gamma(bandpass, m5, phot_params):
    """Calculate gamma parameter.

    Calculate the gamma parameter used for determining photometric
    signal to noise in equation 5 of the LSST overview paper
    (arXiv:0805.2366)

    Parameters
    ----------
    bandpass : `Bandpass`
        Bandpass for which you desire to calculate the gamma parameter.
    m5 : `float`
        The magnitude of a 5-sigma point source detection.
    phot_params : `PhotometricParameters`
        The PhotometricParameters class that carries details about the
        photometric response of the telescope.

    Returns
    -------
    gamma : `float`
        The gamma value for this bandpass.
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
    # 1) Take the approximation N^2 = N0^2 + alpha S from footnote 88
    # in the overview paper where N is the noise in flux of a source,
    # N0 is the noise in flux due to sky brightness and instrumentation,
    # S is the number of counts registered from the source and
    # alpha is some constant
    #
    # 2) Divide by S^2 and demand that N/S = 0.2 for a source detected at m5.
    # Solve the resulting equation for alpha in terms of N0 and S5
    # (the number of counts from a source at m5)
    #
    # 3) Substitute this expression for alpha into the equation for (N/S)^2
    # for a general source.  Re-factor the equation so that it looks like
    # equation 5 of the overview paper (note that x = S5/S).
    # This should give you gamma = (N0/S5)^2
    #
    # 4) Solve equation 41 of the SNR document for neff * sigma_total^2 term
    # given snr=5 and counts as calculated above.
    # Note that neff * sigma_total^2 is N0^2 in the equation above
    #
    # This should give you

    gamma = 0.04 - 1.0 / (counts * phot_params.gain)

    return gamma


def calc_snr_m5(magnitude, bandpass, m5, phot_params, gamma=None):
    """Calculate the SNR of a source based on the 5-sigma limit for an
    observation.

    Calculate signal to noise in flux using the model from equation (5)
    of arXiv:0805.2366

    Parameters
    ----------
    magnitude : `float` or `np.ndarray`, (N,)
        Magnitudes of the sources whose signal to noise you are calculating.
    bandpass : `rubin_sim.phot_utils.Bandpass`
        The Bandpass in which the magnitude was calculated
        (total instrument + atmosphere).
    m5 : `float`
        The 5-sigma point source limiting magnitude of the exposure.
    phot_params : `rubin_sim.phot_utils.PhotometricParameters`
        The PhotometricParameters class that carries details about the
        photometric response of the telescope.
    gamma : `float`, opt
        The gamma parameter from equation(5) of arXiv:0805.2366.
        If not provided, this method will calculate it.

    Returns
    -------
    snr : `float` or `np.ndarray`, (N,)
        The SNR of the input magnitude.
    gamma : `float`
        The gamma parameter for the Bandpass.
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
    """Calculate magnitude error using the model from equation (5)
    of arXiv:0805.2366

    Parameters
    ----------
    magnitude : `float`
        Magnitude of the source.
    bandpass : `rubin_sim.phot_utils.Bandpass`
        The Bandpass in which to calculate the magnitude error
        (total instrument + atmosphere).
    m5 : `float`
        The 5-sigma point source limiting magnitude.
    phot_params : `rubin_sim.phot_utils.PhotometricParameters`
        The PhotometricParameters class that carries details about the
        photometric response of the telescope.
    gamma : `float`, optional
        The gamma parameter from equation(5) of arXiv:0805.2366.
        If not provided, this method will calculate it.

    Returns
    -------
    mag_error : `float` or `np.ndarray`, (N,)
        The magnitude error of the input magnitude.
    gamma : `float`
        The gamma parameter for the Bandpass.
    """

    snr, gamma = calc_snr_m5(magnitude, bandpass, m5, phot_params, gamma=gamma)

    if phot_params.sigma_sys is not None:
        return (
            numpy.sqrt(numpy.power(mag_error_from_snr(snr), 2) + numpy.power(phot_params.sigma_sys, 2)),
            gamma,
        )
    else:
        return mag_error_from_snr(snr), gamma


def calc_snr_sed(
    source_sed,
    total_bandpass,
    sky_sed,
    hardwarebandpass,
    phot_params,
    fwhm_eff,
    verbose=False,
):
    """Calculate the signal to noise ratio for a source, based on the SED.

    For a given source, sky sed, total bandpass and hardware bandpass,
    as well af fwhm_eff / exptime, calculates the SNR with
    optimal PSF extraction assuming a double-gaussian PSF.

    Parameters
    ----------
    source_sed : `rubin_sim.phot_utils.Sed`
        A SED representing the source, normalized such that source_sed.calc_mag
        gives the desired magnitude.
    total_bandpass : `rubin_sim.phot_utils.Bandpass`
        The Bandpass representing the total throughput of the telescope
        (instrument plus atmosphere).
    sky_sed : `rubin_sim.phot_utils.Sed`
        A SED representing the sky background emission, normalized such that
        skysed.calc_mag(Bandpass) returns the expected sky brightness in
        magnitudes per sq arcsecond.
    hardware : `rubin_sim.phot_utils.Bandpass`
        The Bandpass representing the throughput of the telescope instrument
        only (no atmosphere).
    phot_params : `rubin_sim.phot_utils.PhotometricParameters`
        The PhotometricParameters class that carries details about the
        photometric response of the telescope.
    fwhm_eff : `float`
        FWHM in arcseconds.
    verbose : `bool`
        Flag as to whether to print output about SNR.

    Returns
    -------
    snr : `float`
        Calculated SNR.
    """

    # Calculate the counts from the source.
    sourcecounts = source_sed.calc_adu(total_bandpass, phot_params=phot_params)

    # Calculate the (square of the) noise due to signal poisson noise.
    noise_source_sq = sourcecounts / phot_params.gain

    non_source_noise_sq = calc_total_non_source_noise_sq(sky_sed, hardwarebandpass, phot_params, fwhm_eff)

    # Calculate total noise
    noise = numpy.sqrt(noise_source_sq + non_source_noise_sq)
    # Calculate the signal to noise ratio.
    snr = sourcecounts / noise
    if verbose:
        skycounts = sky_sed.calc_adu(hardwarebandpass, phot_params) * (phot_params.platescale**2)
        noise_sky_sq = skycounts / phot_params.gain
        neff = calc_neff(fwhm_eff, phot_params.platescale)
        noise_instr_sq = calc_instr_noise_sq(phot_params)

        print("For Nexp %.1f of time %.1f: " % (phot_params.nexp, phot_params.exptime))
        print("Counts from source: %.2f  Counts from sky: %.2f" % (sourcecounts, skycounts))
        print("fwhm_eff: %.2f('')  Neff pixels: %.3f(pix)" % (fwhm_eff, neff))
        print(
            "Noise from sky: %.2f Noise from instrument: %.2f"
            % (numpy.sqrt(noise_sky_sq), numpy.sqrt(noise_instr_sq))
        )
        print("Noise from source: %.2f" % (numpy.sqrt(noise_source_sq)))
        print(" Total Signal: %.2f   Total Noise: %.2f    SNR: %.2f" % (sourcecounts, noise, snr))
        # Return the signal to noise value.
    return snr


def calc_mag_error_sed(
    source_sed,
    total_bandpass,
    sky_sed,
    hardware_bandpass,
    phot_params,
    fwhm_eff,
    verbose=False,
):
    """Calculate the magnitudeError for a source, given the bandpass(es)
    and sky SED.

    For a given source, sky sed, total bandpass and hardware bandpass, and
    fwhm_eff / exptime, calculates the SNR with optimal PSF extraction
    assuming a double-gaussian PSF.

    Parameters
    ----------
    source_sed : `rubin_sim.phot_utils.Sed`
        A SED representing the source, normalized such that source_sed.calc_mag
        gives the desired magnitude.
    total_bandpass : `rubin_sim.phot_utils.Bandpass`
        The Bandpass representing the total throughput of the telescope
        (instrument plus atmosphere).
    sky_sed : `rubin_sim.phot_utils.Sed`
        A SED representing the sky background emission, normalized such that
        skysed.calc_mag(Bandpass) returns the expected sky brightness in
        magnitudes per sq arcsecond.
    hardware_bandpass : `rubin_sim.phot_utils.Bandpass`
        The Bandpass representing the throughput of the telescope instrument
        only (no atmosphere).
    phot_params : `rubin_sim.phot_utils.PhotometricParameters`
        The PhotometricParameters class that carries details about the
        photometric response of the telescope.
    fwhm_eff : `float`
        FWHM in arcseconds.
    verbose : `bool`, optional
        Flag as to whether to print output about SNR.

    Returns
    -------
    mag_err : `float`
        Magnitude error in expected magnitude.
    """

    snr = calc_snr_sed(
        source_sed,
        total_bandpass,
        sky_sed,
        hardware_bandpass,
        phot_params,
        fwhm_eff,
        verbose=verbose,
    )

    if phot_params.sigma_sys is not None:
        return numpy.sqrt(numpy.power(mag_error_from_snr(snr), 2) + numpy.power(phot_params.sigma_sys, 2))
    else:
        return mag_error_from_snr(snr)


def calc_astrometric_error(mag, m5, fwhm_geom=0.7, nvisit=1, systematic_floor=10):
    """Calculate an expected astrometric error.

    The astrometric error can be estimated for catalog purposes by using the
    typical FWHM and n_visit = total number of visits, or can be used for a
    single visit by using the actual FWHM and n_visit =1.


    Parameters
    ----------
    mag: `float`
        Magnitude of the source
    m5: `float`
        Point source five sigma limiting magnitude of the image
        (or typical depth per image).
    fwhm_geom: `float`, optional
        The geometric (physical) FWHM of the image, in arcseconds.
    nvisit: `int`, optional
        The number of visits/measurement. Default 1.
        If this is >1, the random error contribution is reduced by
        sqrt(nvisits).
    systematic_floor: `float`, optional
        The systematic noise floor for the astrometric measurements,
        in mas. Default 10mas.

    Returns
    -------
    astrom_err : `float`
        Astrometric error for a given SNR, in mas.
    """
    # The astrometric error can be applied to parallax or proper motion
    # (for n_visit>1).
    # If applying to proper motion, should also divide by the # of years
    # of the survey.
    # This is also referenced in the astroph/0805.2366 paper.
    # D. Monet suggests sqrt(Nvisit/2) for first 3 years, sqrt(N) for longer,
    # in reduction of error.
    # Zeljko says 'be conservative', so removing this reduction for now.
    rgamma = 0.039
    xval = numpy.power(10, 0.4 * (mag - m5))
    # The average fwhm_eff is 0.7" (or 700 mas), but user can specify.
    # Convert to mas.
    seeing = fwhm_geom * 1000.0
    error_rand = seeing * numpy.sqrt((0.04 - rgamma) * xval + rgamma * xval * xval)
    error_rand = error_rand / numpy.sqrt(nvisit)
    # The systematic error floor in astrometry (mas).
    error_sys = systematic_floor
    astrom_error = numpy.sqrt(error_sys * error_sys + error_rand * error_rand)
    return astrom_error


def scale_sky_m5(m5target, skysed, total_bandpass, hardware, phot_params, fwhm_eff=0.83):
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

    Note that the returned SED will be renormalized such that calling the
    method self.calcADU(hardwareBandpass) on it will yield the number of
    counts per square arcsecond in a given bandpass.
    """

    # This is based on the LSST SNR document (v1.2, May 2010)
    # www.astro.washington.edu/users/ivezic/Astr511/LSST_SNRdoc.pdf

    sky_counts_target = calc_sky_counts_per_pixel_for_m5(
        m5target, total_bandpass, fwhm_eff=fwhm_eff, phot_params=phot_params
    )

    sky_sed_out = Sed(wavelen=numpy.copy(skysed.wavelen), flambda=numpy.copy(skysed.flambda))

    sky_counts = (
        sky_sed_out.calc_adu(hardware, phot_params=phot_params)
        * phot_params.platescale
        * phot_params.platescale
    )
    sky_sed_out.multiply_flux_norm(sky_counts_target / sky_counts)

    return sky_sed_out
