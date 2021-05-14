from __future__ import print_function
import numpy as np
#from rubin_sim.utils import ObservationMetaData
import healpy as hp
import sys
import glob

# Modifying createStarDensityMap to use GAIA DR1 catalog

# Use the catsim framework to loop over a healpy map and generate a stellar density map

# Connect to fatboy with: ssh -L 51433:fatboy.phys.washington.edu:1433 gateway.astro.washington.edu
# If non-astro user, use simsuser@gateway.astro.washington.edu

if __name__ == '__main__':

    # Hide imports here so documentation builds
    from rubin_sim.catalogs.db import DBObject
    from rubin_sim.utils import halfSpaceFromRaDec
    from rubin_sim.utils import levelFromHtmid
    from rubin_sim.utils import angularSeparation, raDec2Hpid

    #from rubin_sim.catalogs.generation.db import CatalogDBObject
    # Import the bits needed to get the catalog to work
    #from rubin_sim.catUtils.baseCatalogModels import *
    #from rubin_sim.catUtils.exampleCatalogDefinitions import *



    # connect to fatboy
    gaia_db = DBObject(database='LSSTCATSIM', host='fatboy.phys.washington.edu',
                       port=1433, driver='mssql+pymssql')

    # get all of the column names for the gaia table in a list
    gaia_columns = gaia_db.get_column_names(tableName='gaia_2016')

    # Set up healpy map and ra, dec centers
    nside = 64

    # Set the min to 15 since we saturate there. CatSim max is 28
    mag_max = 15.2
    bins = np.arange(0., mag_max, .2)
    starDensity = np.zeros((hp.nside2npix(nside), np.size(bins)-1), dtype=float)
    overMaxMask = np.zeros(hp.nside2npix(nside), dtype=bool)
    lat, ra = hp.pix2ang(nside, np.arange(0, hp.nside2npix(nside)))
    dec = np.pi/2.-lat
    ra = np.degrees(ra)
    dec = np.degrees(dec)

    # Square root of pixel area.
    hpsizeDeg = hp.nside2resol(nside, arcmin=True)/60.

    # Limit things to a 10 arcmin radius
    hpsizeDeg = np.min([10./60., hpsizeDeg])

    indxMin = 0

    restoreFile = glob.glob('gaiaStarDensity_nside_%i.npz' % (nside))
    if len(restoreFile) > 0:
        data = np.load(restoreFile[0])
        starDensity = data['starDensity'].copy()
        indxMin = data['icheck'].copy()
        overMaxMask = data['overMaxMask'].copy()

    print('')
    # Look at a cirular area the same area as the healpix it's centered on.
    boundLength = hpsizeDeg/np.pi**0.5
    radius = boundLength

    blockArea = hpsizeDeg**2  # sq deg

    checksize = 1000
    printsize = 10
    npix = float(hp.nside2npix(nside))

    # If the area has more than this number of objects, flag it as a max
    breakLimit = 1e6
    chunk_size = 10000
    for i in np.arange(indxMin, int(npix)):
        lastCP = ''
        # wonder what the units of boundLength are...degrees! And it's a radius
        # The newer interface:
        #obs_metadata = ObservationMetaData(boundType='circle',
        ##                                   pointingRA=np.degrees(ra[i]),
        #                                   pointingDec=np.degrees(dec[i]),
        #                                   boundLength=boundLength, mjd=5700)

        #t = dbobj.getCatalog('ref_catalog_star', obs_metadata=obs_metadata)
        hs = halfSpaceFromRaDec(ra[i], dec[i], radius)
        current_level = 7
        n_bits_off = 2*(21-current_level)

        tx_list = hs.findAllTrixels(current_level)


        # actually construct the query
        query = 'SELECT ra, dec, phot_g_mean_mag '
        query += 'FROM gaia_2016 '
        query += 'WHERE ' 
        for i_pair, pair in enumerate(tx_list):
            min_tx = int(pair[0]<<n_bits_off)
            max_tx = int((pair[1]+1)<<n_bits_off)
            query += '(htmid>=%d AND htmid<=%d)' % (min_tx, max_tx)
            if i_pair<len(tx_list)-1:
                query += ' OR '

        dtype = np.dtype([('ra', float), ('dec', float), ('mag', float)])

        results = gaia_db.get_arbitrary_chunk_iterator(query, dtype=dtype,
                                                       chunk_size=10000)
        result = list(results)[0]

        distances = angularSeparation(result['ra'], result['dec'], ra[i], dec[i])  # Degrees
        result = result[np.where(distances < radius)]
        

        import pdb ; pdb.set_trace()
        # I could think of setting the chunksize to something really large, then only doing one chunk?
        # Or maybe setting up a way to break out of the loop if everything gets really dense?
        tempHist = np.zeros(np.size(bins)-1, dtype=float)
        counter = 0
        for chunk in results:
            chunkHist, bins = np.histogram(chunk[colName], bins)
            tempHist += chunkHist
            counter += chunk_size
            if counter >= breakLimit:
                overMaxMask[i] = True
                break

        starDensity[i] = np.add.accumulate(tempHist)/blockArea

        # Checkpoint
        if (i % checksize == 0) & (i != 0):
            np.savez('gaiaStarDensity_nside_%i.npz' % (nside),
                     starDensity=starDensity, bins=bins, icheck=i, overMaxMask=overMaxMask)
            lastCP = 'Checkpointed at i=%i of %i' % (i, npix)
        if i % printsize == 0:
            sys.stdout.write('\r')
            perComplete = float(i) / npix * 100
            sys.stdout.write(r'%.2f%% complete. ' % (perComplete) + lastCP)
            sys.stdout.flush()

    np.savez('gaiaStarDensity_nside_%i.npz' % (nside), starDensity=starDensity,
             bins=bins, overMaxMask=overMaxMask)
    print('')
    print('Completed!')
