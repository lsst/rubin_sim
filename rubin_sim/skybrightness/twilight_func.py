import numpy as np

__all__ = ["twilight_func", "zenith_twilight", "simple_twi"]


def simple_twi(xdata, *args):
    """
    Fit a simple slope and constant to many healpixels

    xdata should have keys:
    sunAlt
    hpid

    args:
    0: slope
    1:hpid: magnitudes
    hpid+1:2*hpid: constant offsets
    """

    args = np.array(args)
    hpmax = np.max(xdata["hpid"])
    result = (
        args[xdata["hpid"] + 1] * np.exp(xdata["sunAlt"] * args[0])
        + args[xdata["hpid"] + 2 + hpmax]
    )
    return result


def twilight_func(xdata, *args, amCut=1.0):
    """
    xdata: numpy array with columns 'alt', 'az', 'sunAlt' all in radians.
    az should be relative to the sun (i.e., sun is at az zero.

    based on what I've seen, here's my guess for how to fit the twilight:
    args[0] = ratio of (zenith twilight flux at sun_alt = -12) and dark sky zenith flux
    args[1] = decay slope for all pixels (mags/radian)
    args[2] = airmass term for hemisphere away from the sun. (factor to multiply max brightness at zenith by)
    args[3] = az term for hemisphere towards sun
    args[4] = zenith dark sky flux
    args[5:] = zenith dark sky times constant (optionall)

    amCut : float (1.0)
        The airmass cut to apply to use only the away from sun fit. Was set to 1.1
        previously for not very clear reasons.

    """

    args = np.array(args)
    az = xdata["azRelSun"]
    airmass = xdata["airmass"]
    sun_alt = xdata["sunAlt"]
    flux = np.zeros(az.size, dtype=float)
    away = np.where((airmass <= amCut) | ((az >= np.pi / 2) & (az <= 3.0 * np.pi / 2)))
    towards = np.where((airmass > amCut) & ((az < np.pi / 2) | (az > 3.0 * np.pi / 2)))

    flux = (
        args[0]
        * args[4]
        * 10.0 ** (args[1] * (sun_alt + np.radians(12.0)) + args[2] * (airmass - 1.0))
    )
    flux[towards] *= 10.0 ** (args[3] * np.cos(az[towards]) * (airmass[towards] - 1.0))

    # This let's one fit the dark sky background simultaneously.
    # It assumes the dark sky is a function of airmass only. Forced to be args[4] at zenith.
    if np.size(args) >= 6:
        flux[away] += args[4] * np.exp(
            args[5:][xdata["hpid"][away]] * (airmass[away] - 1.0)
        )
        flux[towards] += args[4] * np.exp(
            args[5:][xdata["hpid"][towards]] * (airmass[towards] - 1.0)
        )

    return flux


def zenith_twilight(alpha, *args):
    """
    The flux at zenith as a linear combination of a twilight component and a constant:
    alpha = sun altitude (radians)
    args[0] = ratio of (zenith twilight flux at sunAlt = -12) and dark sky zenith flux
    args[1] = decay slope for all pixels (mags/radian)
    args[2] = airmass term for hemisphere away from the sun. (factor to multiply max brightness at zenith by)
    args[3] = az term for hemisphere towards sun
    args[4] = zenith dark sky flux
    """

    flux = args[0] * args[4] * 10.0 ** (args[1] * (alpha + np.radians(12.0))) + args[4]
    return flux
