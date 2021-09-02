import os
import numpy as np
import healpy as hp
import warnings
from astropy.coordinates import SkyCoord
import astropy.units as u
from rubin_sim import data as rs_data
import rubin_sim.utils as rs_utils


class Sky_area_generator:
    """Build the sky footprint map.

    The Sky_area_generator sets default regions for the dust-free WFD, the Magellanic Clouds,
    the galactic plane (bulge and background), the Northern Ecliptic Spur, and
    the Southern Celestial Pole. The individual regions are stored in a dictionary of
    self.maps (and self.maps_perfilter) that are simple masks (0-1) which when multiplied
    by their relevant self.nvis dictionary items, produce the expected contribution toward the
    final survey footprint. The total survey footprint, when combined across regions, is
    stored in self.total (and self.total_perfilter), which is a footprint weighted by the
    expected number of visits per pointing; i.e. the value of 'total in the dust-free WFD region
    reflects the expected number of visits per pointing in the dust-free WFD.
    Note that this final 'total' map is not limited by the actual amount of survey time -- you can
    use tools in lsst-pst/survey_strategy/survey_utils to estimate how much time would be required
    to complete a particular footprint. The scheduler simply uses a weighted map to prioritize
    observations, so the achieved footprint will reflect the ratio of these inputs (combined
    with the amount of time available in any part of the sky).

    Parameters
    ----------
    nside : `int`
        Healpix resolution for map
    XXX:  Moved magic values to kwargs. Should refactor to eliminate nvis_wfd_default.
    """
    def __init__(self, nside=64, nvis_wfd_default=860, nvis_frac_nes=0.3, nvis_frac_gp=0.27,
                 nvis_frac_scp=0.14, dec_max=12.):
        self.nside = nside
        # healpix indexes
        self.hpid = np.arange(0, hp.nside2npix(nside))
        # Ra/dec in degrees and other coordinates
        self.ra, self.dec = hp.pix2ang(nside, self.hpid, lonlat=True)
        self.coord = SkyCoord(ra=self.ra * u.deg, dec=self.dec * u.deg, frame='icrs')
        self.eclip_lat = self.coord.barycentrictrueecliptic.lat.deg
        self.eclip_lon = self.coord.barycentrictrueecliptic.lon.deg
        self.gal_lon = self.coord.galactic.l.deg
        self.gal_lat = self.coord.galactic.b.deg
        # filterlist
        self.filterlist = ['u', 'g', 'r', 'i', 'z', 'y']
        # SRD values
        self.nvis_wfd_default = nvis_wfd_default
        self.nvis_frac_nes = nvis_frac_nes
        self.nvis_frac_gp = nvis_frac_gp
        self.nvis_frac_scp = nvis_frac_scp
        # These maps store the per-region information, on scales from 0-1
        self.maps = {}
        self.maps_perfilter = {}
        # The nvis values store the max per-region number of visits, so regions can be added together
        self.nvis = {}
        # Set a default self.dec_max = 12 deg here, but will be re-set/overriden when setting low-dust wfd
        self.dec_max = dec_max
        self.read_dustmap()

    def read_dustmap(self, dustmapFile=None):
        """Read the dustmap from rubin_sim, in the appropriate resolution."""
        # Dustmap from rubin_sim_data  - this is basically just a data directory
        # The dustmap data is downloadable from
        # https://lsst.ncsa.illinois.edu/sim-data/rubin_sim_data/maps_may_2021.tgz
        # (then just set RUBIN_SIM_DATA_DIR to where you downloaded it, after untarring the file)
        if dustmapFile is None:
            datadir = rs_data.get_data_dir()
            if datadir is None:
                raise Exception('Cannot find datadir, please set "RUBIN_SIM_DATA_DIR"')
            datadir = os.path.join(datadir, 'maps', 'DustMaps')
            filename = os.path.join(datadir, 'dust_nside_%i.npz' % self.nside)
        self.dustmap = np.load(filename)['ebvMap']

    def _normalize_filter_balance(self, filter_balance):
        """Normalize a filter balance, so the total is 1."""
        filtersum = np.array(list(filter_balance.values())).sum()
        tmp = {k: round(v / filtersum, 2) for k, v in filter_balance.items()}
        count = 0
        while ((np.array(list(tmp.values())).sum() > 1) and count < 100):
            count += 1
            mostvisits = max(tmp, key=tmp.get)
            tmp[mostvisits] = tmp[mostvisits] - 0.01
            filtersum = np.array(list(tmp.values())).sum()
            tmp = {k: round(v / filtersum, 2) for k, v in tmp.items()}
        for f in self.filterlist:
            if f not in tmp:
                tmp[f] = 0
        return tmp

    # The various regions take the approach that they should be independent
    # And after setting all of the regions, we take the max value (per filter?) in each part of the sky
    # The individual components are updated, so that we can still calculate survey fraction per part of the sky

    def set_dustfree_wfd(self, nvis_dustfree_wfd, dust_limit=0.199,
                         dec_min=-70, dec_max=15,
                         smoothing_cutoff=0.45, smoothing_beam=10,
                         dustfree_wfd_filter_balance=None,
                         adjust_halves=12):
        """Set the dust-free WFD region. This uses the dustmap information to determine low-dust regions.

        Parameters
        -----------
        nvis_dustfree_wfd : `int`
            Number of visits per pointing to plan for the dust-free WFD region
        dust_limit : `float`, optional
            E(B-V) limit for dust extinction. Default 0.1999.
        dec_min : `float`, optional
            Minimum declination boundary for the dust-free WFD (degrees). Default -70.
        dec_max : `float`, optional
            Maximum declination boundary for the dust-free WFD (degrees). Default 15.
        smoothing_cutoff : `float`, optional
            We apply a smoothing filter to the defined dust-free region to avoid sharp edges.
            This value sets the limit for the post-smoothing pixels to be considered 'dust-free WFD'
            or not. Larger values = less area, but guaranteed less dust extinction. Default 0.45.
        smoothing_beam : `float`, optional
            The size of the smoothing filter, in degrees. Default 10.
        dustfree_fwd_filter_balance : `dict` {`str` : `float`}, optional
            How to distribute visits between different filters.
            Default uses {'u': 0.07, 'g': 0.09, 'r': 0.22, 'i': 0.22, 'z': 0.20, 'y': 0.20}
        adjust_halves : `float`, optional
            The RA distribution for the dust-free WFD can be adjusted to account for the fact that the
            MCs and DDFs are concentrated in a small RA range, and that the dust-free region doesn't
            reach the minimum declination value (due to dust). The maximum declination value in the
            galactic south dust-free WFD (the part where RA crosses 0) will be reduced by this amount.
            Default 12 (i.e. in the galactic south, the dec_max will be dec_max - adjust_halves, default 2).
        """
        # Define low dust extinction WFD between dec_min and dec_max (ish) with low dust extinction
        # These dec and dust limits are used to define the other survey areas as well.
        # We're also going to weight the footprint differently in the region around RA=0
        # compared to the region around RA=180, as the RA=0 half is more heavily subscribed
        self.dust_limit = dust_limit
        self.dec_min = dec_min
        self.dec_max = dec_max
        if dustfree_wfd_filter_balance is None:
            self.dustfree_wfd_filter_balance = {'u': 0.07, 'g': 0.09, 'r': 0.22,
                                                'i': 0.22, 'z': 0.20, 'y': 0.20}
        else:
            self.dustfree_wfd_filter_balance = self._normalize_filter_balance(dustfree_wfd_filter_balance)

        self.dustfree = np.where((self.dec > self.dec_min) & (self.dec < self.dec_max)
                                 & (self.dustmap < self.dust_limit), 1, 0)
        # Set the smoothed dust boundary using the original dustmap and smoothing it with gaussian PSF
        self.maps['dustfree'] = np.where((self.dustmap < self.dust_limit), 1, 0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            self.maps['dustfree'] = hp.smoothing(self.maps['dustfree'], fwhm=np.radians(smoothing_beam))
        self.maps['dustfree'] = np.where((self.dec > self.dec_min) & (self.dec < self.dec_max)
                                         & (self.maps['dustfree'] > smoothing_cutoff), 1, 0)

        # Reset to downweight RA=0 and upweight RA=180 side by
        # reducing upper dec limit in one half, increasing lower dec limit in other half
        if adjust_halves > 0:
            self.maps['dustfree'] = np.where((self.gal_lat < 0) & (self.dec > dec_max - adjust_halves),
                                             0, self.maps['dustfree'])
            # The lower dec limit doesn't really apply for the other 'half' of the low-dust WFD
            # as the dust-extinction cuts off the footprint in that area at about Dec=-50
            # This is another reason that the dust-free region is oversubscribed.

        # Make per-filter maps for the footprint
        self.maps_perfilter['dustfree'] = {}
        for f in self.filterlist:
            self.maps_perfilter['dustfree'][f] = self.maps['dustfree'] * self.dustfree_wfd_filter_balance[f]
        self.nvis['dustfree'] = nvis_dustfree_wfd

    def _set_circular_region(self, ra_center, dec_center, radius):
        # find the healpixels that cover a circle of radius radius around ra/dec center (deg)
        result = np.zeros(len(self.ra))
        distance = rs_utils._angularSeparation(np.radians(ra_center), np.radians(dec_center),
                                               np.radians(self.ra), np.radians(self.dec))
        result[np.where(distance < np.radians(radius))] = 1
        return result

    def set_magellanic_clouds(self, nvis_mcs, lmc_radius=8, smc_radius=5,
                              mc_filter_balance=None):
        """Set the magellanic clouds region.

        Parameters
        ----------
        nvis_mcs : `int`
            Number of visits per pointing to expect in the MCs.
        lmc_radius : `float`, optional
            Radius around the LMC to include (degrees).
        smc_radius : `float`, optional
            Radius around the SMC to include (degrees).
        mc_filter_balance : `dict` {`str` : `float`}, optional
            How to distribute visits between different filters.
            Default uses {'u': 0.07, 'g': 0.09, 'r': 0.22, 'i': 0.22, 'z': 0.20, 'y': 0.20}
        """
        # Define the magellanic clouds region
        if mc_filter_balance is None:
            self.mcs_filter_balance = {'u': 0.07, 'g': 0.09, 'r': 0.22,
                                       'i': 0.22, 'z': 0.20, 'y': 0.20}
        else:
            self.mcs_filter_balance = self._normalize_filter_balance(mc_filter_balance)

        self.maps['mcs'] = np.zeros(hp.nside2npix(self.nside))
        # Define the LMC center and size
        self.lmc_ra = 80.893860
        self.lmc_dec = -69.756126
        self.lmc_radius = lmc_radius
        # Define the SMC center and size
        self.smc_ra = 13.186588
        self.smc_dec = -72.828599
        self.smc_radius = smc_radius
        # Define the LMC pixels
        self.maps['mcs'] += self._set_circular_region(self.lmc_ra, self.lmc_dec, self.lmc_radius)
        # Define the SMC pixels
        self.maps['mcs'] += self._set_circular_region(self.smc_ra, self.smc_dec, self.smc_radius)
        # Add a simple bridge between the two - to remove the gap
        mc_dec_min = self.dec[np.where(self.maps['mcs'] > 0)].min()
        mc_dec_max = self.dec[np.where(self.maps['mcs'] > 0)].max()
        self.maps['mcs'] += np.where(((self.ra > self.smc_ra) & (self.ra < self.lmc_ra))
                                     & ((self.dec > mc_dec_min) & (self.dec < mc_dec_max)), 1, 0)
        # We don't want to double-visit areas which may overlap
        self.maps['mcs'] = np.where(self.maps['mcs'] > 1, 1, self.maps['mcs'])

        # Make per-filter maps for the footprint
        self.maps_perfilter['mcs'] = {}
        for f in self.filterlist:
            self.maps_perfilter['mcs'][f] = self.maps['mcs'] * self.mcs_filter_balance[f]
        self.nvis['mcs'] = nvis_mcs

    def _set_bulge_diamond(self, center_width, end_width, gal_long1, gal_long2):
        """
        Define a Galactic Bulge diamond-ish region.

        Parameters
        ----------
        center_width : float
            Width at the center of the galactic plane region.
        end_width : float
            Width at the remainder of the galactic plane region.
        gal_long1 : float
            Longitude at which to start the GP region.
        gal_long2 : float
            Longitude at which to stop the GP region.
            Order matters for gal_long1 / gal_long2!
        Returns
        -------
        bulge : np.ndarray
        """
        # Reject anything beyond the central width.
        bulge = np.where(np.abs(self.gal_lat) < center_width, 1, 0)
        # Apply the galactic longitude cuts, so that plane goes between gal_long1 to gal_long2.
        # This is NOT the shortest distance between the angles.
        gp_length = (gal_long2 - gal_long1) % 360
        # If the length is greater than 0 then we can add additional cuts.
        if gp_length > 0:
            # First, remove anything outside the gal_long1/gal_long2 region.
            bulge = np.where(((self.gal_lon - gal_long1) % 360) < gp_length, bulge, 0)
            # Add the tapers.
            # These slope from the center (gp_center @ center_width)
            # to the edges (gp_center + gp_length/2 @ end_width).
            half_width = gp_length / 2.
            slope = (center_width - end_width) / half_width
            # The 'center' can have a wrap-around 0 problem
            gp_center = (gal_long1 + half_width) % 360
            # Calculate the longitude-distance between any point and the 'center'
            gp_dist = (self.gal_lon - gp_center) % 360
            gp_dist = np.abs(np.where((gp_dist > 180), (180 - gp_dist) % 180, gp_dist))
            lat_limit = np.abs(center_width - slope * gp_dist)
            bulge = np.where((np.abs(self.gal_lat)) < lat_limit, bulge, 0)
        return bulge

    def _set_bulge_rectangle(self, lat_width, gal_long1, gal_long2):
        """
        Define a Galactic Bulge as a simple rectangle in galactic coordinates, centered on the plane

        Parameters
        ----------
        lat_width : `float`
            Latitude with for the galactic bulge
        gal_long1 : float
            Longitude at which to start the GP region.
        gal_long2 : float
            Longitude at which to stop the GP region.
            Order matters for gal_long1 / gal_long2!

        Returns
        -------
        bulge : np.ndarray
        """
        bulge = np.where(np.abs(self.gal_lat) < lat_width, 1, 0)
        # This is NOT the shortest distance between the angles.
        gp_length = (gal_long2 - gal_long1) % 360
        # If the length is greater than 0 then we can add additional cuts.
        if gp_length > 0:
            # First, remove anything outside the gal_long1/gal_long2 region.
            bulge = np.where(((self.gal_lon - gal_long1) % 360) < gp_length, bulge, 0)
        return bulge

    def set_galactic_plane(self, nvis_gal_peak, nvis_gal_min=250, dec_max=12,
                           center_width=12, end_width=4, gal_long1=335, gal_long2=25,
                           gal_lat_width_max=23, gal_filter_balance=None):
        """Set the galactic plane region.

        Parameters
        ----------
        nvis_gal_peak : `int`
            Number of visits per pointing to expect in the galactic bulge.
        nvis_gal_min : `int`, optional
            Number of visits to expect in the background galactic region. Default 250.
        dec_min : `float`, optional
            Dec max for the galactic plane region (degrees). Default 12.
        center_width : `float`, optional
            Width of the central diamond of the galactic bulge (degrees). Default 12.
        end_width : `float`, optional
            Width at the end of the diamond of the galactic bulge (degrees). Default 4.
        gal_long1 : `float`, optional
            Galactic longitude to start the bulge diamond (degrees). Default 335.
        gal_long2 : `float`, optional
            Galactic longitude to end the bulge diamond (degrees). Default 25.
            Order matters for gal_long1 / gal_long2.
        gal_lat_width_max : `float`, optional
            Width of overall galactic plane region (degrees). Generally applies to the background GP,
            and to extending the bulge region toward the WFD in the galactic south. Default 23.
        gal_filter_balance : `dict` {`str` : `float`}, optional
            How to distribute visits between different filters.
            Default {'u': 0.04, 'g': 0.22, 'r': 0.23, 'i': 0.24, 'z': 0.22, 'y': 0.05}
        """
        if gal_filter_balance is None:
            self.gal_filter_balance = {'u': 0.04, 'g': 0.22, 'r': 0.23,
                                       'i': 0.24, 'z': 0.22, 'y': 0.05}
        else:
            self.gal_filter_balance = self._normalize_filter_balance(gal_filter_balance)

        self.gal_dec_max = dec_max
        self.nvis_gal_bg = nvis_gal_min

        # Set up central bulge
        self.bulge = self._set_bulge_diamond(center_width=center_width, end_width=end_width,
                                             gal_long1=gal_long1, gal_long2=gal_long2)
        # Remove any part which may protrude too far north
        self.bulge = np.where(self.dec > self.gal_dec_max, 0, self.bulge)
        # Add simple rectangle to join this part of the bulge to the WFD to the galactic north
        bulge_rectangle = self._set_bulge_rectangle(gal_lat_width_max, gal_long1, gal_long2)
        self.bulge = np.where((bulge_rectangle == 1) & (self.gal_lat < 0),
                              self.bulge + bulge_rectangle, self.bulge)
        # Make all of the bulge region 0 or 1. (and not more than 1)
        self.bulge = np.where(self.bulge > 0, 1, 0)

        # Remove regions of these bulges which go further north than dec_max
        # Set up 'background' galactic plane visits
        self.gp_bkgnd = np.where((np.abs(self.gal_lat) < gal_lat_width_max) & (self.dec < self.dec_max), 1, 0)
        self.gp_bkgnd = self.gp_bkgnd - self.bulge

        # Add them together
        self.maps['gal'] = (self.gp_bkgnd * nvis_gal_min / nvis_gal_peak
                            + self.bulge)

        # Make per-filter maps for the footprint
        self.maps_perfilter['gal'] = {}
        for f in self.filterlist:
            self.maps_perfilter['gal'][f] = self.maps['gal'] * self.gal_filter_balance[f]
        self.nvis['gal'] = nvis_gal_peak

    def set_nes(self, nvis_nes, eclat_min=-10, eclat_max=10, eclip_dec_min=-10,
                dec_cutoff=None, nes_filter_balance=None):
        """Set the Northern Ecliptic Spur region.

        Parameters
        ----------
        nvis_nes : `int`
            Number of visits per pointing to expect in the NES region.
        eclat_min : `float`, optional
            Ecliptic latitutde minimum, for the band of ecliptic coverage around the sky. Default -10.
        eclat_max : `float`, optional
            Ecliptic latitude maximum, for the band of ecliptic coverage around the sky. Default +10.
        eclip_dec_min : `float`, optional
            Declination minimum, for the coverage between the ecliptic plane and the remainder of the
            survey footprint. Default -10.
        dec_cutoff : `float`, optional
            If this value is not None, completely cut off ecliptic coverage below dec_cutoff (degrees).
            This removes the coverage of the ecliptic band through the standard survey area (near the plane).
        nes_filter_balance : `dict` {`str` : `float`}, optional
            How to distribute visits between different filters.
            Default {'u': 0.0, 'g': 0.2, 'r': 0.3, 'i': 0.3, 'z': 0.2, 'y': 0.0}
        """
        if nes_filter_balance is None:
            self.nes_filter_balance = {'u': 0.0, 'g': 0.2, 'r': 0.3, 'i': 0.3, 'z': 0.2, 'y': 0.0}
        else:
            self.nes_filter_balance = self._normalize_filter_balance(nes_filter_balance)
        # NES ecliptic latitude values tend to be assymetric because NES goes so far north
        self.eclat_min = eclat_min
        self.eclat_max = eclat_max
        self.eclip_dec_min = eclip_dec_min

        self.maps['nes'] = np.where(((self.eclip_lat > self.eclat_min) | (self.dec > self.eclip_dec_min))
                                    & (self.eclip_lat < self.eclat_max), 1, 0)
        # Add the option to completely cut off the ecliptic coverage below a given dec value (to match retro)
        if dec_cutoff is not None:
            self.maps['nes'] = np.where(self.dec < dec_cutoff, 0, self.maps['nes'])

        self.maps_perfilter['nes'] = {}
        for f in self.filterlist:
            self.maps_perfilter['nes'][f] = self.maps['nes'] * self.nes_filter_balance[f]
        self.nvis['nes'] = nvis_nes

    def set_scp(self, nvis_scp=120, dec_max=0, scp_filter_balance=None):
        """Set the southern celestial pole coverage.

        In the updated baseline, there is more coverage in the SCP due to the galactic plane region,
        so this region is essentially a backup, making sure there are no regions below dec_max that
        do not get any visits at all. (in the standard baseline, there are some gaps between the galactic
        plane and the dust-free WFD, for example).

        Parameters
        ----------
        nvis_scp : `int`, optional
            The number of visits per pointing to expect for this region. Default 120.
        dec_max : `float`, optional
            The declination cutoff to extend this backup region to. Default 12.
        scp_filter_balance: `dict` {`str` : `float`}
            How to distribute visits between different filters.
            Default {'u': 0.16, 'g': 0.16, 'r': 0.17, 'i': 0.17, 'z': 0.17, 'y': 0.17}
        """
        if scp_filter_balance is None:
            self.scp_filter_balance = {'u': 0.16, 'g': 0.16, 'r': 0.17, 'i': 0.17, 'z': 0.17, 'y': 0.17}
        else:
            self.scp_filter_balance = self._normalize_filter_balance(scp_filter_balance)
        # Basically this is a fill-in so that we don't have any gaps below the max dec limit for the survey
        # I would expect most of this to be ignored
        self.maps['scp'] = np.where(self.dec < dec_max, 1, 0)
        self.maps_perfilter['scp'] = {}
        for f in self.filterlist:
            self.maps_perfilter['scp'][f] = self.maps['scp'] * self.scp_filter_balance[f]
        self.nvis['scp'] = nvis_scp

    def set_ddf(self, nvis_ddf=18000, ddf_radius=1.8, ddf_filter_balance=None):
        """Set DDF regions.

        In general, these should NOT be set when using the Sky_area_generator with the scheduler.
        However, they are helpful placeholders when attempting to estimate sky coverage for other
        purposes and as configured, require about 5% of overall survey time, in the right locations
        of the sky.

        Parameters
        -----------
        nvis_ddf : `int`, optional
            Number of visits per DDF pointing. Default 18,0000.
        ddf_radius : `float`, optional
            Radius for the DDF locations. Default 1.8 degrees.
        ddf_filter_balance : `dict`, {`str` : `float`}
            How to distribute visits between the filters.
            Default {'u': 0.06, 'g': 0.12, 'r': 0.23, 'i': 0.23, 'z': 0.13, 'y': 0.23}
        """
        # These should not be set up for most footprint work with the scheduler, but are helpful
        # for evaluating RA over or under subscription
        if ddf_filter_balance is None:
            # This is an estimate based on existing simulations
            self.ddf_filter_balance = {'u': 0.06, 'g': 0.12, 'r': 0.23, 'i': 0.23, 'z': 0.13, 'y': 0.23}
        else:
            self.ddf_filter_balance = self._normalize_filter_balance(ddf_filter_balance)
        self.ddf_radius = ddf_radius
        self.ddf_centers = rs_utils.ddf_locations()
        self.maps['ddf'] = np.zeros(len(self.hpid))
        for dd in self.ddf_centers:
            self.maps['ddf'] += self._set_circular_region(self.ddf_centers[dd][0],
                                                          self.ddf_centers[dd][1],
                                                          self.ddf_radius)
        self.maps['ddf'] = np.where(self.maps['ddf'] > 1, 1, self.maps['ddf'])

        self.maps_perfilter['ddf'] = {}
        for f in self.filterlist:
            self.maps_perfilter['ddf'][f] = self.maps['ddf'] * self.ddf_filter_balance[f]
        self.nvis['ddf'] = nvis_ddf

    def set_maps(self, dustfree=True, mcs=True, gp=True, nes=True, scp=True, ddf=False):
        """Set maps for each region.

        Using set_maps just sets the defaults for each region it is directed to  set up.
        To set non-default parameters for any given region, set `region=False` when calling this
        method, then set that region individually.

        Parameters
        ----------
        dustfree : `bool`, optional
            Set the dust-free WFD region. Default True.
        mcs : `bool`, optional
            Set the MC (magellanic clouds) region. Default True.
        gp : `bool`, optional
            Set the galactic plane region. Default True.
        nes : `bool`, optional
            Set the NES (northern ecliptic spur) region. Default True.
        scp : `bool`, optional
            Set the SCP (southern celestial pole) region. Default True.
        ddf : `bool`, optional
            Set the DDFs region. Default False.
        """
        # This sets each component with default values.
        # Individual components could be set with non-default values by calling those methods -
        #  in general they are independent (just combined at the end).
        # Each component has a 'map' (with values from 0-1) for the total visits in all filters
        # and then a maps_per_filter (with values from 0-1) for the fraction of those visits
        # which will happen in each filter. Each component also has a 'nvis' value (nvis at the map max value),
        # which serves to weight each map to their final combined value in the footprint.
        if dustfree:
            self.set_dustfree_wfd(self.nvis_wfd_default)
        if mcs:
            self.set_magellanic_clouds(self.nvis_wfd_default)
        if gp:
            self.set_galactic_plane(nvis_gal_peak=self.nvis_wfd_default,
                                    nvis_gal_min=int(self.nvis_wfd_default * self.nvis_frac_gp))
        if nes:
            self.set_nes(int(self.nvis_wfd_default * self.nvis_frac_nes))
        if scp:
            self.set_scp(int(self.nvis_wfd_default * self.nvis_frac_scp))
        if ddf:
            self.set_ddf()

    def combine_maps(self, trim_overlap=True, smoothing_fwhm=3., dust_cut=0.3):
        """Combine the individual maps.

        Parameters
        ----------
        trim_overlap: bool, optional
            If True, look for WFD-like areas which overlap and drop areas according to how much dust there is.
        smoothing_fwhm : float (3)
            The smoothing to use (degrees)
        dust_cut : float (0.3)
            A trial-by-error value for assigning pixels to dust-free or galactic WFD
        """
        if trim_overlap:
            # Specifically look for regions in the various WFD sections which exceed the nvis expected
            # in these regions .. this can happen when the filter balances don't match.
            # Resolve the difference by setting truly low-dust regions to dust-free balance, otherwise use
            # mc/bulge balance.
            total_wfd = np.zeros(len(self.hpid), float)
            for f in self.filterlist:
                wfd_f = np.zeros(len(self.hpid), float)
                for m in ['dustfree', 'gal', 'mcs']:
                    if m in self.maps:
                        wfd_f = np.maximum(wfd_f, self.maps_perfilter[m][f] * self.nvis[m])
                total_wfd += wfd_f
            max_nvis = 0
            for m in ['dustfree', 'gal', 'mcs']:
                if m in self.maps:
                    max_nvis = max(self.nvis[m], max_nvis)
            # Where have we exceeded the expected number of visits in all bands?
            overlap = np.where(total_wfd > max_nvis, 1, 0)
            # Identify areas where overlap falls into really low-dust sky
            # 'self.dustfree' is already 0/1 dust-acceptable or not
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=UserWarning)
                dustfree = hp.smoothing(self.dustfree, fwhm=np.radians(smoothing_fwhm))
            # Choosing whether to assign a particular healpix to the dust-free WFD vs. the galactic WFD 
            # (which have different filter balances), is based on the underlying dust-map values. 
            # The choice of 0.3 means that healpixels are more likely to be assigned to dust-free WFD than galactic plane,
            # but that if a pixel corresponds to a very dusty region, it will be assigned to galactic plane. 
            # These numbers based generally on trial-and-error.
            overlap_dustfree = np.where((dustfree > dust_cut) & (overlap == 1))[0]
            overlap_gal = np.where((dustfree <= dust_cut) & (overlap == 1))[0]
            # And then trim the maps accordingly - just drop the non-priority region
            m = 'dustfree'
            self.maps[m][overlap_gal] = 0
            for f in self.filterlist:
                self.maps_perfilter[m][f][overlap_gal] = 0
            for m in ['gal', 'mcs']:
                self.maps[m][overlap_dustfree] = 0
                for f in self.filterlist:
                    self.maps_perfilter[m][f][overlap_dustfree] = 0

        self.total_perfilter = {}
        for f in self.filterlist:
            self.total_perfilter[f] = np.zeros(len(self.hpid), float)
        for m in self.maps:
            if m == 'ddf':
                continue  # skip DDF
            for f in self.filterlist:
                self.total_perfilter[f] = np.maximum(self.total_perfilter[f],
                                                     self.maps_perfilter[m][f] * self.nvis[m])
        if 'ddf' in self.maps:
            # Now add DDF on top of individual maps
            for f in self.filterlist:
                self.total_perfilter[f] += self.maps_perfilter['ddf'][f] * self.nvis['ddf']

        # Generate the total footprint using the combination of the per-filter values
        self.total = np.zeros(len(self.hpid), float)
        for f in self.filterlist:
            self.total += self.total_perfilter[f]

    def return_maps(self):
        """Call combine_maps and return self.total and self.total_perfilter.
        Returns
        --------
        self.total, self.total_perfilter : `np.ndarray`, `np.ndarray`
            HEALPix maps reflecting the total expected number of visits per pointing, and the
            number of visits per pointing per filter. These can then be scaled appropriately into
            target maps for the scheduler. 
        """
        self.combine_maps()
        return self.total, self.total_perfilter
