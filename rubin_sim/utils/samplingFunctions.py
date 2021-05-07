import numpy as np
import warnings

__all__ = ['spatiallySample_obsmetadata', 'samplePatchOnSphere', 'uniformSphere']


def spatiallySample_obsmetadata(obsmetadata, size=1, seed=1):
    """
    Sample a square patch on the sphere overlapping obsmetadata
    field of view by picking the area enclosed in
    obsmetadata.pointingRA pm obsmetadata.boundLength
    obsmetadata.pointingDec pm obsmetadata.boundLength

    Parameters
    ----------
    obsmetadata: instance of
        `sims.catalogs.generation.db.ObservationMetaData`
    size: integer, optional, defaults to 1
        number of samples

    seed: integer, optional, defaults to 1
        Random Seed used in generating random values
    Returns
    -------
    tuple of ravals, decvalues in radians
    """

    phi = obsmetadata.pointingRA
    theta = obsmetadata.pointingDec

    if obsmetadata.boundType != 'box':
        warnings.warn('Warning: sampling obsmetata with provided boundLen and'
                      'boundType="box", despite diff boundType specified\n')
    equalrange = obsmetadata.boundLength
    ravals, thetavals = samplePatchOnSphere(phi=phi,
                                            theta=theta,
                                            delta=equalrange,
                                            size=size,
                                            seed=seed)
    return ravals, thetavals


def uniformSphere(npoints, seed=42):
    """
    Just make RA, dec points on a sphere
    """
    np.random.seed(seed)
    u = np.random.uniform(size=npoints)
    v = np.random.uniform(size=npoints)

    ra = 2.*np.pi * u
    dec = np.arccos(2.*v - 1.)
    # astro convention of -90 to 90
    dec -= np.pi/2.
    return np.degrees(ra), np.degrees(dec)


def samplePatchOnSphere(phi, theta, delta, size, seed=1):
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
    phi: float, mandatory, degrees
        center of the spherical patch in ra with range
    theta: float, mandatory, degrees
    delta: float, mandatory, degrees
    size: int, mandatory
        number of samples
    seed : int, optional, defaults to 1
        random Seed used for generating values
    Returns
    -------
    tuple of (phivals, thetavals) where phivals and thetavals are arrays of
        size size in degrees.
    """
    np.random.seed(seed)
    u = np.random.uniform(size=size)
    v = np.random.uniform(size=size)

    phi = np.radians(phi)
    theta = np.radians(theta)
    delta = np.radians(delta)

    phivals = 2. * delta * u + (phi - delta)
    phivals = np.where(phivals >= 0., phivals, phivals + 2. * np.pi)

    # use conventions in spherical coordinates
    theta = np.pi / 2.0 - theta

    thetamax = theta + delta
    thetamin = theta - delta

    if thetamax > np.pi or thetamin < 0.:
        raise ValueError('Function not implemented to cover wrap around poles')

    # Cumulative Density Function is cos(thetamin) - cos(theta) /
    # cos(thetamin) - cos(thetamax)
    a = np.cos(thetamin) - np.cos(thetamax)
    thetavals = np.arccos(-v * a + np.cos(thetamin))

    # Get back to -pi/2 to pi/2 range of decs
    thetavals = np.pi / 2.0 - thetavals
    return np.degrees(phivals), np.degrees(thetavals)
