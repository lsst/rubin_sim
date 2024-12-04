__all__ = [
    "predicted_zeropoint",
    "predicted_zeropoint_hardware",
    "predicted_zeropoint_itl",
    "predicted_zeropoint_hardware_itl",
    "predicted_zeropoint_e2v",
    "predicted_zeropoint_hardware_e2v",
]

import numpy as np


def predicted_zeropoint(band: str, airmass: float, exptime: float = 1) -> float:
    """General zeropoint values derived from v1.9 throughputs.

    Extinction coefficients and zeropoint intercepts calculated in
    https://github.com/lsst-pst/syseng_throughputs/blob/main/notebooks/InterpolateZeropoint.ipynb

    Parameters
    ----------
    band : `str`
        The bandpass name.
    airmass : `float`
        The airmass at which to evaluate the zeropoint.
    exptime : `float`, optional
        The exposure time to calculate zeropoint.

    Returns
    -------
    zeropoint : `float`
        The zeropoint for the `band` at airmass `airmass` for an exposure
        of time `exptime`.

    Notes
    -----
    Useful for comparing to DM pipeline zeropoints or for calculating counts
    in an image (`np.power(10, (zp - mag)/2.5)) = counts`).
    """
    extinction_coeff = {
        "u": -0.45815823969467745,
        "g": -0.20789273881603035,
        "r": -0.12233514157672552,
        "i": -0.07387773563152214,
        "z": -0.0573260392897174,
        "y": -0.09549137502152871,
    }
    # Interestingly, because these come from a fit over X, these
    # values are not identical to the zp values found for one second,
    # without considering the fit. They are within 0.005 magnitudes though.
    zeropoint_X1 = {
        "u": 26.52229760811932,
        "g": 28.50754554409417,
        "r": 28.360365503331952,
        "i": 28.170373076693625,
        "z": 27.781368851776005,
        "y": 26.813708013594344,
    }

    return zeropoint_X1[band] + extinction_coeff[band] * (airmass - 1) + 2.5 * np.log10(exptime)


def predicted_zeropoint_hardware(band: str, exptime: float = 1) -> float:
    """Zeropoint values for the hardware throughput curves only,
    without atmospheric contributions.

    Parameters
    ----------
    band : `str`
        The bandpass name.
    exptime : `float`, optional
        The exposure time to calculate zeropoint.

    Returns
    -------
    zeropoint : `float`
        The zeropoint for the hardware only component of `band` for an
        exposure of `exptime`.

    Notes
    -----
    This hardware-only zeropoint is primarily useful for converting sky
    background magnitudes into counts.
    """
    zeropoint = {
        "u": 26.99435242519598,
        "g": 28.72132437054738,
        "r": 28.487206668180864,
        "i": 28.267160381353793,
        "z": 27.850681356053688,
        "y": 26.988827459758397,
    }
    return zeropoint[band] + 2.5 * np.log10(exptime)


def predicted_zeropoint_itl(band: str, airmass: float, exptime: float = 1) -> float:
    """Average ITL zeropoint values derived from v1.9 throughputs.

    Parameters
    ----------
    band : `str`
        The bandpass name.
    airmass : `float`
        The airmass at which to evaluate the zeropoint.
    exptime : `float`, optional
        The exposure time to calculate zeropoint.

    Returns
    -------
    zeropoint : `float`
        The zeropoint for the `band` at airmass `airmass` for an exposure
        of time `exptime`.

    Notes
    -----
    Useful for comparing to DM pipeline zeropoints or for calculating counts
    in an image (`np.power(10, (zp - mag)/2.5)) = counts`).
    """
    extinction_coeff = {
        "u": -0.45815223255080606,
        "g": -0.20789245761381037,
        "r": -0.12233512060667238,
        "i": -0.07387767800064417,
        "z": -0.05739100372986528,
        "y": -0.09474605376660676,
    }
    zeropoint_X1 = {
        "u": 26.52231025834397,
        "g": 28.507547620761827,
        "r": 28.360365812523426,
        "i": 28.170380673108845,
        "z": 27.796728189989665,
        "y": 26.870922441512732,
    }

    return zeropoint_X1[band] + extinction_coeff[band] * (airmass - 1) + 2.5 * np.log10(exptime)


def predicted_zeropoint_hardware_itl(band: str, exptime: float = 1) -> float:
    """Zeropoint values for the ITL hardware throughput curves only,
    without atmospheric contributions.

    Parameters
    ----------
    band : `str`
        The bandpass name.
    exptime : `float`, optional
        The exposure time to calculate zeropoint.

    Returns
    -------
    zeropoint : `float`
        The zeropoint for the hardware only component of `band` for an
        exposure of `exptime`.

    Notes
    -----
    This hardware-only zeropoint is primarily useful for converting sky
    background magnitudes into counts.
    """
    zeropoint = {
        "u": 26.994361876151476,
        "g": 28.721326283280213,
        "r": 28.487206961551088,
        "i": 28.26716792192087,
        "z": 27.86618994188104,
        "y": 27.04446387553851,
    }
    return zeropoint[band] + 2.5 * np.log10(exptime)


def predicted_zeropoint_e2v(band: str, airmass: float, exptime: float = 1) -> float:
    """Average E2V zeropoint values derived from v1.9 throughputs.

    Parameters
    ----------
    band : `str`
        The bandpass name.
    airmass : `float`
        The airmass at which to evaluate the zeropoint.
    exptime : `float`, optional
        The exposure time to calculate zeropoint.

    Returns
    -------
    zeropoint : `float`
        The zeropoint for the `band` at airmass `airmass` for an exposure
        of time `exptime`.

    Notes
    -----
    Useful for comparing to DM pipeline zeropoints or for calculating counts
    in an image (`np.power(10, (zp - mag)/2.5)) = counts`).
    """
    extinction_coeff = {
        "u": -0.4600735940453953,
        "g": -0.20651340321330425,
        "r": -0.12276192263131014,
        "i": -0.07398443681400534,
        "z": -0.057334002964289726,
        "y": -0.095496483868828,
    }
    zeropoint_X1 = {
        "u": 26.58989678516041,
        "g": 28.567959743207357,
        "r": 28.44712188941494,
        "i": 28.19470048013101,
        "z": 27.7817595301949,
        "y": 26.813791858927964,
    }

    return zeropoint_X1[band] + extinction_coeff[band] * (airmass - 1) + 2.5 * np.log10(exptime)


def predicted_zeropoint_hardware_e2v(band: str, exptime: float = 1) -> float:
    """Zeropoint values for the E2V hardware throughput curves only,
    without atmospheric contributions.

    Parameters
    ----------
    band : `str`
        The bandpass name.
    exptime : `float`, optional
        The exposure time to calculate zeropoint.

    Returns
    -------
    zeropoint : `float`
        The zeropoint for the hardware only component of `band` for an
        exposure of `exptime`.

    Notes
    -----
    This hardware-only zeropoint is primarily useful for converting sky
    background magnitudes into counts.
    """
    zeropoint = {
        "u": 27.063967445283826,
        "g": 28.78030646345493,
        "r": 28.574328242939043,
        "i": 28.291563456601306,
        "z": 27.85108207854988,
        "y": 26.988912028019346,
    }

    return zeropoint[band] + 2.5 * np.log10(exptime)
