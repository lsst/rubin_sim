import os
import numpy as np
from astropy.io import ascii, fits
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u

from rubin_sim.data import get_data_dir
from rubin_sim.maf.slicers import UserPointsSlicer
from rubin_sim.maf.metrics import BaseMetric
from rubin_sim.maf.metrics import ExgalM5
from rubin_sim.maf.metrics import StarDensityMetric
from rubin_sim.maf.mafContrib import GalaxyCountsMetric_extended


__all__ = [
    "generateKnownLVDwarfSlicer",
    "make_FakeOldGalaxyLF",
    "make_dwarf_LF_dicts",
    "LVDwarfsMetric",
]


def generateKnownLVDwarfSlicer():
    """Read the Karachentsev+ catalog of nearby galaxies, and put the info about them
    into a UserPointSlicer object.
    """
    filename = os.path.join(
        get_data_dir(), "maf/lvdwarfs", "lsst_galaxies_1p25to9Mpc_table.fits"
    )
    lv_dat0 = fits.getdata(filename)

    # Keep only galaxies at dec < 35 deg., and with stellar masses > 10^7 M_Sun (and <1e14).
    lv_dat_cuts = (
        (lv_dat0["dec"] < 35.0) & (lv_dat0["MStars"] > 1e7) & (lv_dat0["MStars"] < 1e14)
    )
    lv_dat = lv_dat0[lv_dat_cuts]

    # Set up the slicer to evaluate the catalog we just made
    slicer = UserPointsSlicer(lv_dat["ra"], lv_dat["dec"], latLonDeg=True, badval=-666)
    # Add any additional information about each object to the slicer
    slicer.slicePoints["distance"] = lv_dat["dist_Mpc"]

    return slicer


# make a simulated LF for old galaxy of given integrated B, distance modulus mu, in any of filters ugrizY
def make_FakeOldGalaxyLF(intB, mu, filtername):
    """
    Make a simulated luminosity function for an old (10 Gyr) dwarf galaxy of given
        integrated B magnitude, at a given distance modulus, in any of the filters ugrizY.

    Parameters
    ----------
    intB : `float`
        Integrated B-band magnitude of the dwarf to simulate.
    mu : `float`
        Distance modulus at which to place the simulated dwarf.
    filternams: `str`
        Filter in which to produce the simulated luminosity function.
    """
    if filtername == "y":
        filtername == "Y"
    modelBmag = 6.856379  # integrated B mag of the model LF being read
    # Read a simulated luminosity function of [M/H]=-1.5, 10 Gyr stellar population:
    filename = os.path.join(get_data_dir(), "maf/lvdwarfs", "LF_-1.5_10Gyr.dat")
    LF = ascii.read(filename, header_start=12)
    mags = LF["magbinc"]
    counts = LF[filtername + "mag"]
    # shift model LF to requested distance and dim it
    mags = mags + mu
    modelBmag = modelBmag + mu
    # scale model counts up/down to reach the requested intB
    factor = np.power(10.0, -0.4 * (intB - modelBmag))
    counts = factor * counts
    return mags, counts


def make_dwarf_LF_dicts():
    """
    Create dicts containing g- and i-band LFs for simulated dwarfs between
        -10 < M_B < +3, so they can simply be looked up rather than having to
        recreate them each time. Dict is keyed on M_B value.
    """
    lf_dict_i = {}
    lf_dict_g = {}
    # Simulate a range from M_B=-10 to M_B=+5 in 0.1-mag increments.
    tmp_MB = -10.0

    for i in range(151):
        mbkey = f"MB{tmp_MB:.2f}"
        iLFmags, iLFcounts = make_FakeOldGalaxyLF(tmp_MB, 0.0, "i")
        lf_dict_i[mbkey] = (np.array(iLFmags), np.array(iLFcounts))
        gLFmags, gLFcounts = make_FakeOldGalaxyLF(tmp_MB, 0.0, "g")
        lf_dict_g[mbkey] = (np.array(gLFmags), np.array(gLFcounts))
        tmp_MB += 0.1

    return lf_dict_g, lf_dict_i


def _sum_luminosity(LFmags, LFcounts):
    """
    Sum the luminosities from a given luminosity function.

    Uses the first bin's magnitude as a reference, sums luminosities relative to
        that reference star, then converts back to magnitude at the end.

    Parameters
    ----------
    LFmags : np.array, `float`
        Magnitude bin values from the simulated LF.
    LFcounts : np.array, `int`
        Number of stars in each magnitude bin.
    """
    magref = LFmags[0]
    totlum = 0.0

    for mag, count in zip(LFmags, LFcounts):
        tmpmags = np.repeat(mag, count)
        totlum += np.sum(10.0 ** ((magref - tmpmags) / 2.5))

    mtot = magref - 2.5 * np.log10(totlum)
    return mtot


def _dwarf_sblimit(glim, ilim, nstars, lf_dict_g, lf_dict_i, distlim, rng):
    """
    Calculate the surface brightness limit given the g- and i-band limiting
    magnitudes and the number of stars required to detect the dwarf of
    interest.

    Parameters
    ----------
    glim : `float`
        Limiting magnitude (coaddM5) in g-band.
    ilim : `float`
        Limiting magnitude (coaddM5) in i-band.
    nstars : `int`
        Number of stars required to be able to detect the dwarf.
    lf_dict_g, lf_dict_i : `dict`
        Dicts of computed luminosity functions for artificial dwarfs, as
        calculated by the function make_LF_dicts.
    distlim : `float`
        Distance limit (in Mpc) for which to calculate the limiting
        surface brightness for dwarf detection.
    rng : `np.random.Generator`
        Random noise generator for poisson random number use.
    """
    distance_limit = distlim * 1e6  # distance limit in parsecs
    distmod_lim = 5.0 * np.log10(distance_limit) - 5.0

    if (glim > 15) and (ilim > 15):
        # print(glim, ilim, nstars)
        fake_MB = -10.0
        ng = 1e6
        ni = 1e6

        while (ng > nstars) and (ni > nstars) and fake_MB < 5.0:
            # B_fake = distmod_limit+fake_MB
            mbkey = f"MB{fake_MB:.2f}"
            iLFmags0, iLFcounts0 = lf_dict_i[mbkey]
            gLFmags0, gLFcounts0 = lf_dict_g[mbkey]
            iLFcounts = rng.poisson(iLFcounts0)
            gLFcounts = rng.poisson(gLFcounts0)
            iLFmags = (
                iLFmags0 + distmod_lim
            )  # Add the distance modulus to make it apparent mags
            gLFmags = (
                gLFmags0 + distmod_lim
            )  # Add the distance modulus to make it apparent mags
            # print(iLFcounts0-iLFcounts)
            gsel = gLFmags <= glim
            isel = iLFmags <= ilim
            ng = np.sum(gLFcounts[gsel])
            ni = np.sum(iLFcounts[isel])
            # print('fake_MB: ',fake_MB, ' ng: ',ng, ' ni: ', ni, ' nstars: ', nstars)
            fake_MB += 0.1

        if fake_MB > -9.9 and (ng > 0) and (ni > 0):
            gmag_tot = _sum_luminosity(gLFmags[gsel], gLFcounts[gsel]) - distmod_lim
            imag_tot = _sum_luminosity(iLFmags[isel], iLFcounts[isel]) - distmod_lim
            # S = m + 2.5logA, where in this case things are in sq. arcmin, so A = 1 arcmin^2 = 3600 arcsec^2
            sbtot_g = distmod_lim + gmag_tot + 2.5 * np.log10(3600.0)
            sbtot_i = distmod_lim + imag_tot + 2.5 * np.log10(3600.0)
            mg_lim = gmag_tot
            mi_lim = imag_tot
            sbg_lim = sbtot_g
            sbi_lim = sbtot_i
            if ng < ni:
                flag_lim = "g"
            else:
                flag_lim = "i"
        else:
            mg_lim = 999.9
            mi_lim = 999.9
            sbg_lim = 999.9
            sbi_lim = 999.9
            flag_lim = None
    else:
        mg_lim = 999.9
        mi_lim = 999.9
        sbg_lim = -999.9
        sbi_lim = -999.9
        flag_lim = None

    return mg_lim, mi_lim, sbg_lim, sbi_lim, flag_lim


class LVDwarfsMetric(BaseMetric):
    """
    Estimate the detection limit in total dwarf luminosity for resolved dwarf galaxies
    at a given distance.

    This metric class uses simulated luminosity functions of dwarf galaxies with
    known (assumed) luminosities to estimate the detection limit (in total dwarf
    luminosity, M_V) for resolved dwarf galaxies at a given distance. It can be
    applied to either known galaxies with their discrete positions and distances,
    or an entire survey simulation with a fixed distance limit.

    In the default use (with the KnownLvDwarfsSlicer), it returns detection limits for
    a catalog of known local volume dwarfs, from the Karachentsev+ catalog of nearby galaxies.

    Parameters
    ----------
    radius : `float`, default=2.45,
        Radius of the field being considered (for discrete fields only). By default,
        UserPointSlicer uses a 2.45-deg field radius.
    distlim : `float`,
        Distance threshold in Mpc for which to calculate the limiting dwarf detection
        luminosity. Only needed for healpix slicers, but *required* if healpix is used.
    cmd_frac : `float`, default=0.1,
        Fraction of the total area of the color-magnitude diagram that is spanned
        by the tracer selection criteria. (e.g., the size of a box in color and
        magnitude to select RGB-star candidates)
    stargal_contamination : `float`, default=0.4,
        Fraction of objects in CMD selection region that are actually unresolved
        galaxies that were mis-classified as stars.
    nsigma : `float`, default=10.0,
        Required detection significance to declare a simulated dwarf "detected."
    """

    def __init__(
        self,
        radius=2.45,
        distlim=None,
        cmd_frac=0.1,
        stargal_contamination=0.40,
        nsigma=10.0,
        metricName="LVDwarfs",
        seed=505,
        **kwargs,
    ):
        self.radius = radius
        self.filterCol = "filter"
        self.m5Col = "fiveSigmaDepth"
        self.cmd_frac = cmd_frac
        self.stargal_contamination = stargal_contamination
        self.nsigma = nsigma

        if distlim is not None:
            # If the distance limit was set on input, extract that info:
            self.distlim = distlim
        else:
            # If no distance limit specified, assume the intention is to search
            #   around known Local Volume host galaxies.
            self.distlim = None
            filename = os.path.join(
                get_data_dir(), "maf/lvdwarfs", "lsst_galaxies_1p25to9Mpc_table.fits"
            )
            lv_dat0 = fits.getdata(filename)
            # Keep only galaxies at dec < 35 deg., and with stellar masses > 10^7 M_Sun.
            lv_dat_cuts = (
                (lv_dat0["dec"] < 35.0)
                & (lv_dat0["MStars"] > 1e7)
                & (lv_dat0["MStars"] < 1e14)
            )
            lv_dat = lv_dat0[lv_dat_cuts]
            sc_dat = SkyCoord(
                ra=lv_dat["ra"] * u.deg,
                dec=lv_dat["dec"] * u.deg,
                distance=lv_dat["dist_Mpc"] * u.Mpc,
            )
            self.sc_dat = sc_dat

        self.lf_dict_g, self.lf_dict_i = make_dwarf_LF_dicts()

        # Set up already-defined metrics that we will need:
        self.ExgalCoaddm5 = ExgalM5(m5Col=self.m5Col, filterCol=self.filterCol)
        # The StarDensityMetric calculates the number of stars in i band
        self.StarDensityMetric = StarDensityMetric(filtername="i")
        # The galaxy counts metric calculates the number of galaxies in i band
        self.GalaxyCountsMetric = GalaxyCountsMetric_extended(
            m5Col=self.m5Col, filterBand="i", includeDustExtinction=True
        )
        # Set the scale for the GalaxyCountMetric_extended to 1, so it returns
        # galaxies per sq deg, not n galaxies per healpixel
        self.GalaxyCountsMetric.scale = 1

        cols = [self.m5Col, self.filterCol]
        # GalaxyCountsMetric needs the DustMap, and StarDensityMetric needs StellarDensityMap:
        maps = ["DustMap", "StellarDensityMap"]
        super().__init__(
            col=cols, metricName=metricName, maps=maps, units="M_V limit", **kwargs
        )

        # Set up a random number generator, so that metric results are repeatable
        self.rng = np.random.default_rng(seed)

    def run(self, dataSlice, slicePoint):

        # Identify observations in g and i bandpasses
        gband = dataSlice[self.filterCol] == "g"
        iband = dataSlice[self.filterCol] == "i"
        # if there are no visits in either of g or i band, exit
        if len(np.where(gband)[0]) == 0 or len(np.where(iband)[0]) == 0:
            return self.badval

        # calculate the dust-extincted coadded 5-sigma limiting mags in the g and i bands:
        g5 = self.ExgalCoaddm5.run(dataSlice[gband], slicePoint)
        i5 = self.ExgalCoaddm5.run(dataSlice[iband], slicePoint)

        if g5 < 15 or i5 < 15:
            # If the limiting magnitudes won't even match the stellar density maps, exit
            return self.badval

        # Find the number of stars per sq arcsecond at the i band limit
        # (this is a bit of a hack to get the starDensityMetric to calculate the nstars at this mag exactly)
        star_i5 = min(27.9, i5)
        self.StarDensityMetric.magLimit = star_i5

        nstar_sqarcsec = self.StarDensityMetric.run(dataSlice, slicePoint)

        # Calculate the number of galaxies per sq degree
        ngal_sqdeg = self.GalaxyCountsMetric.run(dataSlice, slicePoint)
        # GalaxyCountsMetric is undefined in some places. These cases return
        #   zero; catch these and set the galaxy counts in those regions to a
        #   very high value. (this may not be true after catching earlier no-visits issues)
        if ngal_sqdeg < 10.0:
            ngal_sqdeg = 1e7

        # Convert from per sq deg and per sq arcsecond into #'s per sq arcminute
        ngal_sqarcmin = ngal_sqdeg / 3600
        nstar_sqarcmin = nstar_sqarcsec * 3600

        if ngal_sqarcmin < 0 or nstar_sqarcmin < 0:
            print(
                f"Here be a problem - ngals_sqarcmin {ngal_sqarcmin} or nstar_sqarcmin {nstar_sqarcmin} "
                f'are negative. depths: {g5}, {i5}. {slicePoint["ra"], slicePoint["dec"], slicePoint["sid"]}'
            )
        # The number of stars required to reach nsigma is nsigma times the Poisson
        #   fluctuations of the background (stars+galaxies contamination):
        nstars_required = self.nsigma * np.sqrt(
            (ngal_sqarcmin * self.cmd_frac * self.stargal_contamination)
            + (nstar_sqarcmin * self.cmd_frac)
        )

        if self.distlim is not None:
            # Use the distlim if a healpix slicer is input
            distlim = self.distlim
        else:
            # Use discrete distances for known galaxies if a UserPointSlicer:
            distlim = slicePoint["distance"] * u.Mpc
            # sc_slice = SkyCoord(ra=slicePoint['ra']*u.rad, dec=slicePoint['dec']*u.rad)
            # seps = sc_slice.separation(self.sc_dat)
            # distlim = self.sc_dat[seps.argmin()].distance

        # Calculate the limiting luminosity and surface brightness based on g5 and i5:
        mg_lim, mi_lim, sb_g_lim, sb_i_lim, flag_lim = _dwarf_sblimit(
            g5,
            i5,
            nstars_required,
            self.lf_dict_g,
            self.lf_dict_i,
            distlim=distlim.value,
            rng=self.rng,
        )

        if flag_lim is None:
            mv = self.badval

        else:
            # To go from HSC g and i bands to V, use the conversion from Appendix A
            #   of Komiyama+2018, ApJ, 853, 29:
            # V = g_hsc - 0.371*(gi_hsc)-0.068
            mv = mg_lim - 0.371 * (mg_lim - mi_lim) - 0.068
            # sbv = sb_g_lim - 0.371 * (sb_g_lim - sb_i_lim) - 0.068

        return mv
