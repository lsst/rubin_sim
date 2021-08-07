import numpy as np

# Use the main stack to make a rough array.


# Need to put these in sims_utils and remove from MAF and scheduler.

def gnomonic_project_toxy(RA1, Dec1, RAcen, Deccen):
    """Calculate x/y projection of RA1/Dec1 in system with center at RAcen, Deccen.
    Input radians. Grabbed from sims_selfcal"""
    # also used in Global Telescope Network website
    cosc = np.sin(Deccen) * np.sin(Dec1) + np.cos(Deccen) * np.cos(Dec1) * np.cos(RA1-RAcen)
    x = np.cos(Dec1) * np.sin(RA1-RAcen) / cosc
    y = (np.cos(Deccen)*np.sin(Dec1) - np.sin(Deccen)*np.cos(Dec1)*np.cos(RA1-RAcen)) / cosc
    return x, y


def gnomonic_project_tosky(x, y, RAcen, Deccen):
    """Calculate RA/Dec on sky of object with x/y and RA/Cen of field of view.
    Returns Ra/Dec in radians."""
    denom = np.cos(Deccen) - y * np.sin(Deccen)
    RA = RAcen + np.arctan2(x, denom)
    Dec = np.arctan2(np.sin(Deccen) + y * np.cos(Deccen), np.sqrt(x*x + denom*denom))
    return RA, Dec


if __name__ == '__main__':
    from lsst.sims.coordUtils import _chipNameFromRaDec
    from lsst.obs.lsst import LsstCamMapper
    import lsst.sims.utils as simsUtils

    mapper = LsstCamMapper()
    camera = mapper.camera
    epoch = 2000.0

    ra = 0.
    dec = 0.
    rotSkyPos = 0.
    mjd = 5300.

    obs_metadata = simsUtils.ObservationMetaData(pointingRA=np.degrees(ra),
                                                 pointingDec=np.degrees(dec),
                                                 rotSkyPos=np.degrees(rotSkyPos),
                                                 mjd=mjd)

    nside = int(1000)
    # 60k pixels, from 0 to 3.5 degrees
    x_one = np.linspace(-1.75, 1.75, int(nside))

    # make 2-d x,y arrays
    x_two = np.broadcast_to(x_one, (nside, nside))
    y_two = np.broadcast_to(x_one, (nside, nside)).T

    result = np.ones((nside, nside), dtype=bool)
    ra_two, dec_two = gnomonic_project_tosky(np.radians(x_two), np.radians(y_two), ra, dec)
    chipNames = _chipNameFromRaDec(ra_two.ravel(), dec_two.ravel(), epoch=epoch,
                                   camera=camera, obs_metadata=obs_metadata)

    chipNames = chipNames.reshape(nside, nside)
    wavefront_names = [name for name in np.unique(chipNames[np.where(chipNames != None)]) if ('SW' in name)
                       | ('R44' in name) | ('R00' in name) | ('R04' in name) | ('R40' in name)]
    # If it's on a waverfront sensor, that's false
    for name in wavefront_names:
        result[np.where(chipNames == name)] = False
    # No chipname, that's a false
    result[np.where(chipNames == None)] = False

    np.savez('fov_map.npz', x=x_one, image=result)
