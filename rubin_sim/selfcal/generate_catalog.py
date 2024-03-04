__all__ = ("generate_catalog",)

import sys
import warnings

import numpy as np
import numpy.lib.recfunctions as rfn
from scipy.spatial import cKDTree as kdtree  # noqa: N813

from .offsets import OffsetSNR
from .star_tools import assign_patches, stars_project


def wrap_ra(ra):
    """Wraps RA into 0-360 degrees."""
    ra = ra % 360.0
    return ra


def cap_dec(dec):
    """Terminates declination at +/- 90 degrees."""
    dec = np.where(dec > 90, 90, dec)
    dec = np.where(dec < -90, -90, dec)
    return dec


def treexyz(ra, dec):
    """Calculate x/y/z values for ra/dec points, ra/dec in radians."""
    # Note ra/dec can be arrays.
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return x, y, z


def build_tree(ra, dec, leafsize=100):
    """Build KD tree on RA/dec and set radius (via setRad) for matching.

    Parameters
    ----------
    ra : `nd.ndarray`, (N,)
        RA values of the tree (in radians)
    dec : `nd.ndarray`, (N,)
        Dec values of the tree (in radians).
    leafsize : `float`, opt
        The number of RA/Dec pointings in each leafnode.
        Default 100.
    """
    if np.any(np.abs(ra) > np.pi * 2.0) or np.any(np.abs(dec) > np.pi * 2.0):
        raise ValueError("Expecting RA and Dec values to be in radians.")
    x, y, z = treexyz(ra, dec)
    data = list(zip(x, y, z))
    if np.size(data) > 0:
        star_tree = kdtree(data, leafsize=leafsize)
    else:
        raise ValueError("ra and dec should have length greater than 0.")
    return star_tree


def generate_catalog(
    visits,
    stars_array,
    offsets=None,
    lsst_filter="r",
    n_patches=16,
    radius_fov=1.8,
    seed=42,
    uncert_floor=0.005,
    verbose=True,
):
    """
    Generate a catalog of observed stellar magnitudes.

    Parameters
    ----------
    visits : `np.array`, (N,)
        A numpy array with the properties of the visits.
        Expected columns of fiveSigmaDepth, ra, dec, rotSkyPos (all degrees)
    offsets : `list` of rubin_sim.selfcal.Offset classes
        A list of instatiated classes that will apply offsets to the stars
    lsst_filter :  `str`
        Which filter to use for the observed stars.
    n_patches : `int`
        Number of patches to divide the FoV into.  Must be an integer squared
    radius_fov : `float`
        Radius of the telescope field of view in degrees
    seed : `float`
        Random number seed
    uncert_floor : `float`
        Value to add in quadrature to magnitude uncertainties (mags)
    verbose : `bool`
        Should we be verbose
    """

    if offsets is None:
        # Maybe change this to just run with a default SNR offset
        warnings.warn("Warning, no offsets set, returning without running")
        return

    # For computing what the 'expected' uncertainty on the observation will be
    mag_uncert = OffsetSNR(lsst_filter=lsst_filter)

    # set the radius for the kdtree
    x0, y0, z0 = (1, 0, 0)
    x1, y1, z1 = treexyz(np.radians(radius_fov), 0)
    tree_radius = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)

    newcols = [
        "x",
        "y",
        "radius",
        "patch_id",
        "sub_patch",
        "observed_mag",
        "mag_uncert",
    ]
    newtypes = [float, float, float, int, int, float, float]
    stars_new = np.zeros(stars_array.size, dtype=list(zip(newcols, newtypes)))

    # only need to output these columns (saving rmag+ for now for convienence)
    output_cols = [
        "id",
        "patch_id",
        "observed_mag",
        "mag_uncert",
        "%smag" % lsst_filter,
        "ra",
        "decl",
    ]
    output_dtypes = [int, int, float, float, float, float, float]

    stars = rfn.merge_arrays([stars_array, stars_new], flatten=True)

    # Build a KDTree for the stars
    star_tree = build_tree(np.radians(stars["ra"]), np.radians(stars["decl"]))

    # XXX--maybe update the way seeding is going on
    np.random.seed(seed)

    list_of_observed_arrays = []

    n_visits = np.size(visits)

    for i, visit in enumerate(visits):
        dmags = {}
        # Calc x,y, radius for each star, crop off stars outside the FoV
        # could replace with code to see where each star falls and get chipID.
        vx, vy, vz = treexyz(np.radians(visit["ra"]), np.radians(visit["dec"]))
        indices = star_tree.query_ball_point((vx, vy, vz), tree_radius)
        stars_in = stars[indices]
        stars_in = stars_project(stars_in, visit)

        # Assign patchIDs
        stars_in = assign_patches(stars_in, visit, n_patches=n_patches)

        # Apply the offsets that have been configured
        for offset in offsets:
            dmags[offset.newkey] = offset(stars_in, visit, dmags=dmags)

        # Total up all the dmag's to make the observed magnitude
        keys = list(dmags.keys())
        obs_mag = stars_in["%smag" % lsst_filter].copy()
        for key in keys:
            obs_mag += dmags[key]

        # Calculate the uncertainty in the observed mag:
        mag_err = (
            mag_uncert.calc_mag_errors(obs_mag, err_only=True, m5=visit["fiveSigmaDepth"]) ** 2
            + uncert_floor**2
        ) ** 0.5

        # put values into the right columns
        stars_in["observed_mag"] = obs_mag
        stars_in["mag_uncert"] = mag_err
        # Should shrink this down so we only return the needed columns
        # observed_mag, mag_uncert, patchid, star_id
        sub_cols = np.empty(stars_in.size, dtype=list(zip(output_cols, output_dtypes)))
        for key in output_cols:
            sub_cols[key] = stars_in[key]
        list_of_observed_arrays.append(sub_cols)
        if verbose:
            progress = i / n_visits * 100
            text = "\rprogress = %.2f%%" % progress
            sys.stdout.write(text)
            sys.stdout.flush()

    result = np.concatenate(list_of_observed_arrays)
    return result
