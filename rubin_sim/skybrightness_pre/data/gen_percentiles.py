from builtins import zip
import numpy as np
from lsst.sims.utils import m5_flat_sed
from lsst.sims.photUtils import LSSTdefaults


def restore_files():
    roots = ['61390_61603', '61573_61786', '61756_61969']
    dicts = []
    sbs = []
    for root in roots:
        restore_file = 'healpix/'+root+'.npz'
        disk_data = np.load(restore_file)
        required_mjds = disk_data['header'][()]['required_mjds'].copy()
        dict_of_lists = disk_data['dict_of_lists'][()].copy()
        disk_data.close()
        sky_brightness = np.load('healpix/'+root+'.npy')
        # find the indices of all the evenly spaced mjd values
        even_mjd_indx = np.in1d(dict_of_lists['mjds'], required_mjds)
        for key in dict_of_lists:
            dict_of_lists[key] = dict_of_lists[key][even_mjd_indx]
        sky_brightness = sky_brightness[even_mjd_indx]

        dicts.append(dict_of_lists)
        sbs.append(sky_brightness)

    sky_brightness = sbs[0]
    dict_of_lists = dicts[0]

    try:
        for i in range(1, len(dicts)):
            new_mjds = np.where(dicts[i]['mjds'] > dict_of_lists['mjds'].max())[0]
            for key in dict_of_lists:
                if isinstance(dicts[i][key][new_mjds], list):
                    dict_of_lists[key].extend(dicts[i][key][new_mjds])
                else:
                    dict_of_lists[key] = np.concatenate((dict_of_lists[key], dicts[i][key][new_mjds]))
            sky_brightness = np.concatenate((sky_brightness, sbs[i][new_mjds]))
    except:
        import pdb ; pdb.set_trace()

    return sky_brightness, dict_of_lists



def generate_percentiles(nbins=20):
    """
    Make histograms of the 5-sigma limiting depths for each point and each filter.
    """

    filters = ['u', 'g', 'r', 'i', 'z', 'y']

    sky_brightness, dict_of_lists = restore_files()

    npix = sky_brightness['r'].shape[-1]

    histograms = np.zeros((nbins, npix), dtype=list(zip(filters, [float]*6)))
    histogram_npts = np.zeros(npix, dtype=list(zip(filters, [int]*6)))

    
    for filtername in filters:
        # convert surface brightness to m5
        FWHMeff = LSSTdefaults().FWHMeff(filtername)
        # increase as a function of airmass
        airmass_correction = np.power(dict_of_lists['airmass'], 0.6)
        FWHMeff *= airmass_correction
        m5_arr = m5_flat_sed(filtername, sky_brightness[filtername], FWHMeff, 30.,
                             dict_of_lists['airmass'])

        for indx in np.arange(npix):
            m5s = m5_arr[:, indx]
            m5s = m5s[np.isfinite(m5s)]
            m5s = np.sort(m5s)
            percentile_points = np.round(np.linspace(0, m5s.size-1, nbins))
            if m5s.size > percentile_points.size:
                histograms[filtername][:, indx] = m5s[percentile_points.astype(int)]
                histogram_npts[filtername][indx] = m5s.size
            # make the histogram for this point in the sky
            # histograms[filtername][:, indx] += np.histogram(m5s[np.isfinite(m5s)],
            #                                                bins=bins[filtername])[0]

    np.savez('percentile_m5_maps.npz', histograms=histograms, histogram_npts=histogram_npts)


if __name__ == '__main__':

    # make a giant 2-year file
    #mjd0 = 59853
    #test_length = 365.25  # days
    #generate_sky(mjd0=mjd0, mjd_max=mjd0+test_length, outpath='', outfile='big_temp_sky.npz')

    generate_percentiles()
