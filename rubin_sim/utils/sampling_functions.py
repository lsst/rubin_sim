__all__ = ("spatially_sample_obsmetadata", "sample_patch_on_sphere", "uniform_sphere")

import warnings

import numpy as np


def spatially_sample_obsmetadata(obsmetadata, size=1, seed=1):
    """
    Sample a square patch on the sphere overlapping obsmetadata
    field of view by picking the area enclosed in
    obsmetadata.pointing_ra pm obsmetadata.bound_length
    obsmetadata.pointing_dec pm obsmetadata.bound_length

    Parameters
    ----------
    obsmetadata : rubin_sim.utils.ObservationMetaData`
    size : `int`, optional, defaults to 1
        number of samples
    seed : `int`, optional, defaults to 1
        Random Seed used in generating random values

    Returns
    -------
    ravals, thetavals : `np.ndarray`, `np.ndarray`
        tuple of ravals, decvalues in radians
    """

    phi = obsmetadata.pointing_ra
    theta = obsmetadata.pointing_dec

    if obsmetadata.bound_type != "box":
        warnings.warn(
            "Warning: sampling obsmetata with provided boundLen and"
            'bound_type="box", despite diff bound_type specified\n'
        )
    equalrange = obsmetadata.bound_length
    ravals, thetavals = sample_patch_on_sphere(phi=phi, theta=theta, delta=equalrange, size=size, seed=seed)
    return ravals, thetavals


def uniform_sphere(npoints, seed=42):
    """
    Just make RA, dec points on a sphere
    """
    np.random.seed(seed)
    u = np.random.uniform(size=npoints)
    v = np.random.uniform(size=npoints)

    ra = 2.0 * np.pi * u
    dec = np.arccos(2.0 * v - 1.0)
    # astro convention of -90 to 90
    dec -= np.pi / 2.0
    return np.degrees(ra), np.degrees(dec)


def sample_patch_on_sphere(phi, theta, delta, size, seed=1):
    """
    Uniformly distributes samples on a patch on a sphere between phi pm delta,
    and theta pm delta on a sphere. Uniform distribution implies that the
    number of points in a patch of sphere is proportional to the area of the
    patch. Here, the coordinate system is the usual
    spherical coordinate system but with the azimuthal angle theta going from
    90 degrees at the North Pole, to -90 degrees at the South Pole, through
    0. at the equator.

    This function is not equipped to handle wrap-around the ranges of theta
    phi and therefore does not work at the poles.

    Parameters
    ----------
    phi : `float`,
        center of the spherical patch in ra with range, degrees
    theta : `float`
        degrees
    delta : `float`
        degrees
    size : `int`
        number of samples
    seed : `int`, optional, defaults to 1
        random Seed used for generating values

    Returns
    -------
    phivals, thetavals : `np.ndarray`, `np.ndarray`
        tuple of (phivals, thetavals) where phivals and thetavals are arrays of
        size size in degrees.
    """
    np.random.seed(seed)
    u = np.random.uniform(size=size)
    v = np.random.uniform(size=size)

    phi = np.radians(phi)
    theta = np.radians(theta)
    delta = np.radians(delta)

    phivals = 2.0 * delta * u + (phi - delta)
    phivals = np.where(phivals >= 0.0, phivals, phivals + 2.0 * np.pi)

    # use conventions in spherical coordinates
    theta = np.pi / 2.0 - theta

    thetamax = theta + delta
    thetamin = theta - delta

    if thetamax > np.pi or thetamin < 0.0:
        raise ValueError("Function not implemented to cover wrap around poles")

    # Cumulative Density Function is cos(thetamin) - cos(theta) /
    # cos(thetamin) - cos(thetamax)
    a = np.cos(thetamin) - np.cos(thetamax)
    thetavals = np.arccos(-v * a + np.cos(thetamin))

    # Get back to -pi/2 to pi/2 range of decs
    thetavals = np.pi / 2.0 - thetavals
    return np.degrees(phivals), np.degrees(thetavals)
