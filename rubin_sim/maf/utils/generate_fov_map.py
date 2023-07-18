import numpy as np

# Use the main stack to make a rough array.


# Need to put these in sims_utils and remove from MAF and scheduler.


def gnomonic_project_toxy(ra1, dec1, r_acen, deccen):
    """Calculate x/y projection of ra1/dec1 in system with center at r_acen, deccen.
    Input radians. Grabbed from sims_selfcal"""
    # also used in Global Telescope Network website
    cosc = np.sin(deccen) * np.sin(dec1) + np.cos(deccen) * np.cos(dec1) * np.cos(ra1 - r_acen)
    x = np.cos(dec1) * np.sin(ra1 - r_acen) / cosc
    y = (np.cos(deccen) * np.sin(dec1) - np.sin(deccen) * np.cos(dec1) * np.cos(ra1 - r_acen)) / cosc
    return x, y


def gnomonic_project_tosky(x, y, r_acen, deccen):
    """Calculate RA/dec on sky of object with x/y and RA/Cen of field of view.
    Returns Ra/dec in radians."""
    denom = np.cos(deccen) - y * np.sin(deccen)
    RA = r_acen + np.arctan2(x, denom)
    dec = np.arctan2(np.sin(deccen) + y * np.cos(deccen), np.sqrt(x * x + denom * denom))
    return RA, dec


if __name__ == "__main__":
    import lsst.sims.utils as simsUtils
    from lsst.obs.lsst import LsstCamMapper
    from lsst.sims.coordUtils import _chipNameFromRaDec

    mapper = LsstCamMapper()
    camera = mapper.camera
    epoch = 2000.0

    ra = 0.0
    dec = 0.0
    rot_sky_pos = 0.0
    mjd = 5300.0

    obs_metadata = simsUtils.ObservationMetaData(
        pointing_ra=np.degrees(ra),
        pointing_dec=np.degrees(dec),
        rot_sky_pos=np.degrees(rot_sky_pos),
        mjd=mjd,
    )

    nside = int(1000)
    # 60k pixels, from 0 to 3.5 degrees
    x_one = np.linspace(-1.75, 1.75, int(nside))

    # make 2-d x,y arrays
    x_two = np.broadcast_to(x_one, (nside, nside))
    y_two = np.broadcast_to(x_one, (nside, nside)).T

    result = np.ones((nside, nside), dtype=bool)
    ra_two, dec_two = gnomonic_project_tosky(np.radians(x_two), np.radians(y_two), ra, dec)
    chip_names = _chipNameFromRaDec(
        ra_two.ravel(),
        dec_two.ravel(),
        epoch=epoch,
        camera=camera,
        obs_metadata=obs_metadata,
    )

    chip_names = chip_names.reshape(nside, nside)
    wavefront_names = [
        name
        for name in np.unique(chip_names[np.where(chip_names != None)])
        if ("SW" in name) | ("R44" in name) | ("R00" in name) | ("R04" in name) | ("R40" in name)
    ]
    # If it's on a waverfront sensor, that's false
    for name in wavefront_names:
        result[np.where(chip_names == name)] = False
    # No chipname, that's a false
    result[np.where(chip_names == None)] = False

    np.savez("fov_map.npz", x=x_one, image=result)
