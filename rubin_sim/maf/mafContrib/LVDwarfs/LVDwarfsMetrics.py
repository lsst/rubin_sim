import numpy as np
from astropy.io import ascii, fits


# make fake LF for old galaxy of given integrated B, distance modulus mu, in any of filters ugrizY
def makeFakeLF(intB, mu, filtername):
    if (filtername == 'y'):
        filtername == 'Y'
    modelBmag = 6.856379  # integrated B mag of the model LF being read
    LF = ascii.read('LF_-1.5_10Gyr.dat', header_start=12)
    mags = LF['magbinc']
    counts = LF[filtername+'mag']
    # shift model LF to requested distance and dim it
    mags = mags + mu
    modelBmag = modelBmag + mu
    # scale model counts up/down to reach the requested intB
    factor = np.power(10.0, -0.4*(intB-modelBmag))
    counts = factor * counts
    return mags, counts


def make_LF_dicts():
    lf_dict_i = {}
    lf_dict_g = {}
    tmp_MB = -10.0

    for i in range(101):
        mbkey = f'MB{tmp_MB:.2f}'
        iLFmags, iLFcounts = makeFakeLF(tmp_MB, 0.0, 'i')
        lf_dict_i[mbkey] = (np.array(iLFmags), np.array(iLFcounts))
        gLFmags, gLFcounts = makeFakeLF(tmp_MB, 0.0, 'g')
        lf_dict_g[mbkey] = (np.array(gLFmags), np.array(gLFcounts))
        tmp_MB += 0.1

    return lf_dict_g, lf_dict_i


def sum_luminosity(LFmags, LFcounts):
    magref = LFmags[0]
    totlum = 0.0

    for mag, count in zip(LFmags, LFcounts):
        tmpmags = np.repeat(mag, count)
        totlum += np.sum(10.0**((magref - tmpmags)/2.5))

    mtot = magref-2.5*np.log10(totlum)
    return mtot


def sblimit(mags_g, mags_i, nstars_req, distlim):
    distance_limit = distlim*1e6  # distance limit in parsecs
    distmod_limit = 5.0*np.log10(distance_limit) - 5.0

    mg_lim = []
    mi_lim = []
    sbg_lim = []
    sbi_lim = []
    flag_lim = []

    for glim, ilim, nstars, distmod_lim in zip(mags_g, mags_i, nstars_req, distmod_limit):
    # for i in range(len(mags_g)):
        if (glim > 15) and (ilim > 15):
            # print(glim, ilim, nstars)
            fake_MB = -10.0
            ng = 1e6
            ni = 1e6

            while (ng > nstars) and (ni > nstars) and fake_MB < -2.0:
                # B_fake = distmod_limit+fake_MB
                mbkey = f'MB{fake_MB:.2f}'
                iLFmags0, iLFcounts0 = lf_dict_i[mbkey]
                gLFmags0, gLFcounts0 = lf_dict_g[mbkey]
                iLFcounts = np.random.poisson(iLFcounts0)
                gLFcounts = np.random.poisson(gLFcounts0)
                iLFmags = iLFmags0+distmod_lim  # Add the distance modulus to make it apparent mags
                gLFmags = gLFmags0+distmod_lim  # Add the distance modulus to make it apparent mags
                # print(iLFcounts0-iLFcounts)
                gsel = (gLFmags <= glim)
                isel = (iLFmags <= ilim)
                ng = np.sum(gLFcounts[gsel])
                ni = np.sum(iLFcounts[isel])
                # print('fake_MB: ',fake_MB, ' ng: ',ng, ' ni: ', ni, ' nstars: ', nstars)
                fake_MB += 0.1

            if fake_MB > -9.9:
                gmag_tot = sum_luminosity(gLFmags[gsel], gLFcounts[gsel]) - distmod_lim
                imag_tot = sum_luminosity(iLFmags[isel], iLFcounts[isel]) - distmod_lim
                # S = m + 2.5logA, where in this case things are in sq. arcmin, so A = 1 arcmin^2 = 3600 arcsec^2
                sbtot_g = distmod_lim + gmag_tot + 2.5*np.log10(3600.0)
                sbtot_i = distmod_lim + imag_tot + 2.5*np.log10(3600.0)
                mg_lim.append(gmag_tot)
                mi_lim.append(imag_tot)
                sbg_lim.append(sbtot_g)
                sbi_lim.append(sbtot_i)
                if (ng < ni):
                    flag_lim.append('g')
                else:
                    flag_lim.append('i')
            else:
                mg_lim.append(999.9)
                mi_lim.append(999.9)
                sbg_lim.append(999.9)
                sbi_lim.append(999.9)
                flag_lim.append('none')
        else:
            mg_lim.append(999.9)
            mi_lim.append(999.9)
            sbg_lim.append(-999.9)
            sbi_lim.append(-999.9)
            flag_lim.append('none')

    return mg_lim, mi_lim, sbg_lim, sbi_lim, flag_lim


#####################

lf_dict_g, lf_dict_i = make_LF_dicts()

lv_dat0 = fits.getdata('lsst_galaxies_1p25to9Mpc_table.fits')

# Keep only galaxies at dec < 35 deg., and with stellar masses > 10^7 M_Sun.

lv_dat_cuts = (lv_dat0['dec'] < 35.0) & (lv_dat0['MStars'] > 1e7) & (lv_dat0['MStars'] < 1e14)
lv_dat = lv_dat0[lv_dat_cuts]
