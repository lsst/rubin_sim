import glob
import sys

# from rubin_sim.utils import ObservationMetaData
import healpy as hp
import numpy as np

# Modifying createStarDensityMap to use GAIA DR1 catalog

# Use the catsim framework to loop over a healpy map and generate a
# stellar density map

# Connect to fatboy with: ssh -L 51433:fatboy.phys.washington.edu:1433
# gateway.astro.washington.edu
# If non-astro user, use simsuser@gateway.astro.washington.edu

# NOTE: fatboy is no longer operative

if __name__ == "__main__":
    # Hide imports here so documentation builds
    from lsst.sims.catalogs.db import DBObject
    from lsst.sims.utils import angular_separation, halfSpaceFromRaDec

    # from lsst.sims.catalogs.generation.db import CatalogDBObject
    # Import the bits needed to get the catalog to work
    # from rubin_sim.catUtils.baseCatalogModels import *
    # from rubin_sim.catUtils.exampleCatalogDefinitions import *
    # connect to fatboy
    gaia_db = DBObject(
        database="LSSTCATSIM",
        host="fatboy.phys.washington.edu",
        port=1433,
        driver="mssql+pymssql",
    )

    # get all of the column names for the gaia table in a list
    gaia_columns = gaia_db.get_column_names(tableName="gaia_2016")

    # Set up healpy map and ra, dec centers
    nside = 64

    # Set the min to 15 since we saturate there. CatSim max is 28
    mag_max = 15.2
    bins = np.arange(0.0, mag_max, 0.2)
    star_density = np.zeros((hp.nside2npix(nside), np.size(bins) - 1), dtype=float)
    over_max_mask = np.zeros(hp.nside2npix(nside), dtype=bool)
    lat, ra = hp.pix2ang(nside, np.arange(0, hp.nside2npix(nside)))
    dec = np.pi / 2.0 - lat
    ra = np.degrees(ra)
    dec = np.degrees(dec)

    # Square root of pixel area.
    hpsize_deg = hp.nside2resol(nside, arcmin=True) / 60.0

    # Limit things to a 10 arcmin radius
    hpsize_deg = np.min([10.0 / 60.0, hpsize_deg])

    indx_min = 0

    restore_file = glob.glob("gaiaStarDensity_nside_%i.npz" % (nside))
    if len(restore_file) > 0:
        data = np.load(restore_file[0])
        star_density = data["star_density"].copy()
        indx_min = data["icheck"].copy()
        over_max_mask = data["over_max_mask"].copy()

    print("")
    # Look at a circular area the same area as the healpix it's centered on.
    bound_length = hpsize_deg / np.pi**0.5
    radius = bound_length

    block_area = hpsize_deg**2  # sq deg

    checksize = 1000
    printsize = 10
    npix = float(hp.nside2npix(nside))

    # If the area has more than this number of objects, flag it as a max
    break_limit = 1e6
    chunk_size = 10000
    for i in np.arange(indx_min, int(npix)):
        last_cp = ""
        # wonder what the units of bound_length are...degrees!
        # And it's a radius
        # The newer interface:
        # obs_metadata = ObservationMetaData(bound_type='circle',
        ##                                   pointing_ra=np.degrees(ra[i]),
        #                                   pointing_dec=np.degrees(dec[i]),
        #                                   bound_length=bound_length,
        #                                   mjd=5700)

        # t = dbobj.getCatalog('ref_catalog_star', obs_metadata=obs_metadata)
        hs = halfSpaceFromRaDec(ra[i], dec[i], radius)
        current_level = 7
        n_bits_off = 2 * (21 - current_level)

        tx_list = hs.findAllTrixels(current_level)

        # actually construct the query
        query = "SELECT ra, dec, phot_g_mean_mag "
        query += "FROM gaia_2016 "
        query += "WHERE "
        for i_pair, pair in enumerate(tx_list):
            min_tx = int(pair[0] << n_bits_off)
            max_tx = int((pair[1] + 1) << n_bits_off)
            query += "(htmid>=%d AND htmid<=%d)" % (min_tx, max_tx)
            if i_pair < len(tx_list) - 1:
                query += " OR "

        dtype = np.dtype([("ra", float), ("dec", float), ("mag", float)])

        results = gaia_db.get_arbitrary_chunk_iterator(query, dtype=dtype, chunk_size=10000)
        result = list(results)[0]

        distances = angular_separation(result["ra"], result["dec"], ra[i], dec[i])
        result = result[np.where(distances < radius)]

        # I could think of setting the chunksize to something really large,
        # then only doing one chunk?
        # Or maybe setting up a way to break out of the loop if
        # everything gets really dense?
        temp_hist = np.zeros(np.size(bins) - 1, dtype=float)
        counter = 0
        col_name = "phot_g_mean_mag"
        for chunk in results:
            chunk_hist, bins = np.histogram(chunk[col_name], bins)
            temp_hist += chunk_hist
            counter += chunk_size
            if counter >= break_limit:
                over_max_mask[i] = True
                break

        star_density[i] = np.add.accumulate(temp_hist) / block_area

        # Checkpoint
        if (i % checksize == 0) & (i != 0):
            np.savez(
                "gaiaStarDensity_nside_%i.npz" % (nside),
                starDensity=star_density,
                bins=bins,
                icheck=i,
                overMaxMask=over_max_mask,
            )
            last_cp = "Checkpointed at i=%i of %i" % (i, npix)
        if i % printsize == 0:
            sys.stdout.write("\r")
            per_complete = float(i) / npix * 100
            sys.stdout.write(r"%.2f%% complete. " % (per_complete) + last_cp)
            sys.stdout.flush()

    np.savez(
        "gaiaStarDensity_nside_%i.npz" % (nside),
        starDensity=star_density,
        bins=bins,
        overMaxMask=over_max_mask,
    )
    print("")
    print("Completed!")
