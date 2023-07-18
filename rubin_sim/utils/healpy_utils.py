__all__ = (
    "hpid2_ra_dec",
    "ra_dec2_hpid",
    "healbin",
    "_hpid2_ra_dec",
    "_ra_dec2_hpid",
    "_healbin",
    "moc2array",
    "hp_grow_argsort",
)

import warnings

import healpy as hp
import numpy as np


def _hpid2_ra_dec(nside, hpids, **kwargs):
    """
    Correct for healpy being silly and running dec from 0-180.

    Parameters
    ----------
    nside : int
        Must be a value of 2^N.
    hpids : np.array
        Array (or single value) of healpixel IDs.

    Returns
    -------
    ra_ret : float (or np.array)
        RA positions of the input healpixel IDs. In radians.
    dec_ret : float (or np.array)
        Dec positions of the input healpixel IDs. In radians.
    """

    lat, lon = hp.pix2ang(nside, hpids, **kwargs)
    dec_ret = np.pi / 2.0 - lat
    ra_ret = lon

    return ra_ret, dec_ret


def hpid2_ra_dec(nside, hpids, **kwargs):
    """
    Correct for healpy being silly and running dec from 0-180.

    Parameters
    ----------
    nside : int
        Must be a value of 2^N.
    hpids : np.array
        Array (or single value) of healpixel IDs.

    Returns
    -------
    raRet : float (or np.array)
        RA positions of the input healpixel IDs. In degrees.
    decRet : float (or np.array)
        Dec positions of the input healpixel IDs. In degrees.
    """
    ra, dec = _hpid2_ra_dec(nside, hpids, **kwargs)
    return np.degrees(ra), np.degrees(dec)


def _ra_dec2_hpid(nside, ra, dec, **kwargs):
    """
    Assign ra,dec points to the correct healpixel.

    Parameters
    ----------
    nside : int
        Must be a value of 2^N.
    ra : np.array
        RA values to assign to healpixels. Radians.
    dec : np.array
        Dec values to assign to healpixels. Radians.

    Returns
    -------
    hpids : np.array
        Healpixel IDs for the input positions.
    """
    lat = np.pi / 2.0 - dec
    hpids = hp.ang2pix(nside, lat, ra, **kwargs)
    return hpids


def ra_dec2_hpid(nside, ra, dec, **kwargs):
    """
    Assign ra,dec points to the correct healpixel.

    Parameters
    ----------
    nside : int
        Must be a value of 2^N.
    ra : np.array
        RA values to assign to healpixels. Degrees.
    dec : np.array
        Dec values to assign to healpixels. Degrees.

    Returns
    -------
    hpids : np.array
        Healpixel IDs for the input positions.
    """
    return _ra_dec2_hpid(nside, np.radians(ra), np.radians(dec), **kwargs)


def _healbin(ra, dec, values, nside=128, reduce_func=np.mean, dtype=float, fill_val=hp.UNSEEN):
    """
    Take arrays of ra's, dec's, and value and bin into healpixels. Like numpy.hexbin but for
    bins on a sphere.

    Parameters
    ----------
    ra : np.array
        RA positions of the data points. Radians.
    dec : np.array
        Dec positions of the data points. Radians
    values : np.array
        The values at each ra,dec position.
    nside : int
        Healpixel nside resolution. Must be a value of 2^N.
    reduce_func : function (numpy.mean)
        A function that will return a single value given a subset of `values`.
    dtype : dtype ('float')
        Data type of the resulting mask
    fill_val : float (hp.UNSEEN)
        Value to fill in with healpixels that have no value.
        Default is the healpy mask value.

    Returns
    -------
    map_vals : np.array
        A numpy array that is a valid Healpixel map.
    """

    hpids = _ra_dec2_hpid(nside, ra, dec)

    order = np.argsort(hpids)
    hpids = hpids[order]
    values = values[order]
    pixids = np.unique(hpids)

    left = np.searchsorted(hpids, pixids)
    right = np.searchsorted(hpids, pixids, side="right")

    map_vals = np.zeros(hp.nside2npix(nside), dtype=dtype) + fill_val

    # Wow, I thought histogram would be faster than the loop, but this has been faster!
    for i, idx in enumerate(pixids):
        map_vals[idx] = reduce_func(values[left[i] : right[i]])

    # Change any NaNs to fill value
    map_vals[np.isnan(map_vals)] = fill_val

    return map_vals


def healbin(ra, dec, values, nside=128, reduce_func=np.mean, dtype=float, fill_val=hp.UNSEEN):
    """
    Take arrays of ra's, dec's, and value and bin into healpixels. Like numpy.hexbin but for
    bins on a sphere.

    Parameters
    ----------
    ra : np.array
        RA positions of the data points. Degrees.
    dec : np.array
        Dec positions of the data points. Degrees.
    values : np.array
        The values at each ra,dec position.
    nside : int
        Healpixel nside resolution. Must be a value of 2^N.
    reduce_func : function (numpy.mean)
        A function that will return a single value given a subset of `values`.
    dtype : dtype ('float')
        Data type of the resulting mask
    fill_val : float (hp.UNSEEN)
        Value to fill in with healpixels that have no value.
        Default is the healpy mask value.

    Returns
    -------
    mapVals : np.array
        A numpy array that is a valid Healpixel map.
    """
    return _healbin(
        np.radians(ra),
        np.radians(dec),
        values,
        nside=nside,
        reduce_func=reduce_func,
        dtype=dtype,
        fill_val=fill_val,
    )


def moc2array(data, uniq, nside=128, reduce_func=np.sum, density=True, fill_val=0.0):
    """Convert a Multi-Order Coverage Map to a single nside HEALPix array. Useful
    for converting maps output by LIGO alerts. Expect that future versions of
    healpy or astropy will be able to replace this functionality. Note that this is
    a convienence function that will probably degrade portions of the MOC that are
    sampled at high resolution.

    Details of HEALPix Mulit-Order Coverage map: http://ivoa.net/documents/MOC/20190404/PR-MOC-1.1-20190404.pdf

    Parameters
    ----------
    data : np.array
        Data values for the MOC map
    uniq : np.array
        The UNIQ values for the MOC map
    nside : int (128)
        The output map nside
    reduce_func : function (np.sum)
        The function to use to combine data into single healpixels.
    density : bool (True)
        If True, multiplies data values by pixel area before applying reduce_func, and divides
        the final array by the output pixel area. Should be True if working on a probability density MOC.
    fill_val : float (0.)
        Value to fill empty HEALPixels with. Good choices include 0 (default), hp.UNSEEN, and np.nan.

    Returns
    -------
    np.array : HEALpy array of nside. Units should be the same as the input map as processed by reduce_func.
    """

    # NUNIQ packing, from page 12 of http://ivoa.net/documents/MOC/20190404/PR-MOC-1.1-20190404.pdf
    orders = np.floor(np.log2(uniq / 4) / 2).astype(int)
    npixs = (uniq - 4 * 4**orders).astype(int)

    nsides = 2**orders
    names = ["ra", "dec", "area"]
    types = [float] * len(names)
    data_points = np.zeros(data.size, dtype=list(zip(names, types)))
    for order in np.unique(orders):
        good = np.where(orders == order)
        ra, dec = _hpid2_ra_dec(nsides[good][0], npixs[good], nest=True)
        data_points["ra"][good] = ra
        data_points["dec"][good] = dec
        data_points["area"][good] = hp.nside2pixarea(nsides[good][0])

    if density:
        tobin_data = data * data_points["area"]
    else:
        tobin_data = data

    result = _healbin(
        data_points["ra"],
        data_points["dec"],
        tobin_data,
        nside=nside,
        reduce_func=reduce_func,
        fill_val=fill_val,
    )

    if density:
        good = np.where(result != fill_val)
        result[good] = result[good] / hp.nside2pixarea(nside)

    return result


def hp_grow_argsort(in_map, ignore_nan=True):
    """Find the maximum of a healpix map, then orders healpixels by selecting the maximum bordering the selected area.

    Parameters
    ----------
    in_map : np.array
        A valid HEALpix array
    ignore_nan : bool (True)
        If true, ignores values that are NaN

    Returns
    -------
    ordered_hp : int array
        The indices that put in_map in the correct order
    """
    nside = hp.npix2nside(np.size(in_map))
    npix = np.size(in_map)
    pix_indx = np.arange(npix)

    if ignore_nan:
        not_nan_pix = ~np.isnan(in_map)
        npix = np.size(in_map[not_nan_pix])

    # Check if we have already run with this nside
    if hasattr(hp_grow_argsort, "nside"):
        nside_match = nside == hp_grow_argsort.nside
    else:
        nside_match = False

    # If we already have neighbors chached, just use it
    if nside_match:
        neighbors = hp_grow_argsort.neighbors_cache
    else:
        # Running a new nside, or for the first time, compute neighbors and set attributes
        # Make a `bool` area to keep track of which pixels still need to be sorted
        neighbors = hp.get_all_neighbours(nside, pix_indx).T
        hp_grow_argsort.neighbors_cache = neighbors
        hp_grow_argsort.nside = nside

    valid_neighbors_mask = np.ones(neighbors.shape, dtype=bool)

    # Sometimes there can be no neighbors in some directions
    valid_neighbors_mask[np.where(neighbors == -1)] = False

    ordered_hp = np.zeros(npix, dtype=int)
    current_max = np.where(in_map == np.nanmax(in_map))[0].min()

    ordered_hp[0] = current_max

    # Remove max from valid_neighbors. Can be clever with indexing
    # so we don't have to do a brute force search of the entire
    # neghbors array to mask it.
    # valid_neighbors_mask[np.where(neighbors == current_max)] = False
    current_neighbors = neighbors[current_max][valid_neighbors_mask[current_max]]
    sub_indx = np.where(neighbors[current_neighbors] == current_max)
    valid_neighbors_mask[(current_neighbors[sub_indx[0]], sub_indx[1])] = False

    for i in np.arange(1, npix):
        current_neighbors = neighbors[ordered_hp[0:i]][valid_neighbors_mask[ordered_hp[0:i]]]
        indx = np.where(in_map[current_neighbors] == np.nanmax(in_map[current_neighbors]))[0]
        if np.size(indx) == 0:
            # We can't connect to any more pixels
            warnings.warn("Can not connect to any more pixels.")
            return ordered_hp[0:i]
        else:
            indx = np.min(indx)
        current_max = current_neighbors[indx]
        ordered_hp[i] = current_max
        # current_max is no longer a valid neighbor to consider
        neighbors_of_current = neighbors[current_max]
        sub_indx = np.where(neighbors[neighbors_of_current] == current_max)
        valid_neighbors_mask[(neighbors_of_current[sub_indx[0]], sub_indx[1])] = False

    return ordered_hp
