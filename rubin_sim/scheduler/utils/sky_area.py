import os
import numpy as np
import healpy as hp
import warnings
from astropy.coordinates import SkyCoord
import astropy.units as u
from rubin_sim import data as rs_data
import rubin_sim.utils as rs_utils

__all__ = ['Sky_area_generator']


class Sky_area_generator:
    def __init__(self, nside=64, default_filter_balance=None):
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
        self.nvis_min_srd = 750
        self.nvis_goal_srd = 825
        self.area_min_srd = 15000
        self.area_goal_srd = 18000
        self.nvis_wfd_default = 855
        self.nvis_frac_nes = 0.35
        self.nvis_frac_gp = 0.35
        if default_filter_balance is None:
            self.default_filter_balance = {'u': 0.07, 'g': 0.09, 'r': 0.22,
                                           'i': 0.22, 'z': 0.20, 'y': 0.20}
        else:
            self.default_filter_balance = self._normalize_filter_balance(default_filter_balance)
        # These maps store the per-region information, on scales from 0-1
        self.maps = {}
        self.maps_perfilter = {}
        # The nvis values store the max per-region number of visits, so regions can be added together
        self.nvis = {}
        # Set a default self.dec_max = 12 deg here, but will be re-set/overriden when setting low-dust wfd
        self.dec_max = 12

    def read_dustmap(self, dustmapFile=None):
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
        filtersum = np.array(list(filter_balance.values())).sum()
        tmp = {k: round(v / filtersum, 2) for k, v in filter_balance.items()}
        count = 0
        while ((np.array(list(tmp.values())).sum() > 1) and count<100):
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

    def _set_dustfree_wfd(self, nvis_dustfree_wfd, dust_limit=0.199,
                          dec_min=-70, dec_max=15,
                          smoothing_cutoff=0.45, smoothing_beam=10,
                          dustfree_wfd_filter_balance=None,
                          adjust_halves=12):
        # Define low dust extinction WFD between dec_min and dec_max (ish) with low dust extinction
        # These dec and dust limits are used to define the other survey areas as well.
        # We're also going to weight the footprint differently in the region around RA=0
        # compared to the region around RA=180, as the RA=0 half is more heavily subscribed (if adjust_halves True)
        self.dust_limit = dust_limit
        self.dec_min = dec_min
        self.dec_max = dec_max
        if dustfree_wfd_filter_balance is None:
            self.dustfree_wfd_filter_balance = self.default_filter_balance
        else:
            self.dustfree_wfd_filter_balance = self._normalize_filter_balance(dustfree_wfd_filter_balance)

        # Set the detailed dust boundary
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

    def _set_magellanic_clouds(self, nvis_mcs, lmc_radius=8, smc_radius=5,
                               mc_filter_balance=None):
        # Define the magellanic clouds region
        if mc_filter_balance is None:
            self.mcs_filter_balance = self.default_filter_balance
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
        np.ndarray
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

    def _set_galactic_plane(self, nvis_gal_A, nvis_gal_B=300, nvis_gal_min=250,
                            dec_max=12,
                            center_width_A=12, end_width_A=4, gal_long1_A=335, gal_long2_A=25,
                            center_width_B=15, end_width_B=5, gal_long1_B=260, gal_long2_B=90,
                            gal_lat_width_max=23, gal_filter_balance=None):
        if gal_filter_balance is None:
            self.gal_filter_balance = {'u': 0.04, 'g': 0.22, 'r': 0.23,
                                       'i': 0.24, 'z': 0.22, 'y': 0.05}
        else:
            self.gal_filter_balance = self._normalize_filter_balance(gal_filter_balance)
        self.gal_dec_max = dec_max

        # Set up central bulge
        self.bulge_A = self._set_bulge_diamond(center_width=center_width_A, end_width=end_width_A,
                                               gal_long1=gal_long1_A, gal_long2=gal_long2_A)
        self.bulge_A = np.where(self.dec > self.gal_dec_max, 0, self.bulge_A)
        # And a secondary bulge-ish region to follow further stars
        self.bulge_B = self._set_bulge_diamond(center_width=center_width_B, end_width=end_width_B,
                                               gal_long1=gal_long1_B, gal_long2=gal_long2_B)
        self.bulge_B = np.where(self.dec > self.gal_dec_max, 0, self.bulge_B)
        # Remove regions of these bulges which go further north than dec_max
        # Set up 'background' galactic plane visits
        self.gp_bkgnd = np.where((np.abs(self.gal_lat) < gal_lat_width_max) & (self.dec < self.dec_max), 1, 0)
        # Remove the areas that overlap
        self.bulge_B = self.bulge_B - self.bulge_A
        self.gp_bkgnd = self.gp_bkgnd - self.bulge_A - self.bulge_B

        # Add them together
        self.maps['gal'] = (self.gp_bkgnd * nvis_gal_min / nvis_gal_A
                            + self.bulge_B * nvis_gal_B / nvis_gal_A
                            + self.bulge_A)

        # Make per-filter maps for the footprint
        self.maps_perfilter['gal'] = {}
        for f in self.filterlist:
            self.maps_perfilter['gal'][f] = self.maps['gal'] * self.gal_filter_balance[f]
        self.nvis['gal'] = nvis_gal_A

    def _set_nes(self, nvis_nes=375, eclat_min=-10, eclat_max=10, eclip_dec_min=-10, eclip_ra_max=180,
                 dec_cutoff=None, nes_filter_balance=None):
        if nes_filter_balance is None:
            self.nes_filter_balance = {'u': 0.0, 'g': 0.2, 'r': 0.3, 'i': 0.3, 'z': 0.2, 'y': 0.0}
        else:
            self.nes_filter_balance = self._normalize_filter_balance(nes_filter_balance)
        # NES ecliptic latitude values tend to be assymetric because NES goes so far north
        self.eclat_min = eclat_min
        self.eclat_max = eclat_max
        self.eclip_dec_min = eclip_dec_min
        self.eclip_ra_max = eclip_ra_max

        self.maps['nes'] = np.where(((self.eclip_lat > self.eclat_min) |
                                     ((self.dec > self.eclip_dec_min) & (self.ra < self.eclip_ra_max)))
                                    & (self.eclip_lat < self.eclat_max), 1, 0)
        # Add the option to completely cut off the ecliptic coverage below a given dec value (to match retro)
        if dec_cutoff is not None:
            self.maps['nes'] = np.where(self.dec < dec_cutoff, 0, self.maps['nes'])

        self.maps_perfilter['nes'] = {}
        for f in self.filterlist:
            self.maps_perfilter['nes'][f] = self.maps['nes'] * self.nes_filter_balance[f]
        self.nvis['nes'] = nvis_nes

    def _set_scp(self, nvis_scp=120, dec_max=0, scp_filter_balance=None):
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

    def _set_ddf(self, nvis_ddf=18000, ddf_radius=1.8, ddf_filter_balance=None):
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
        # This sets each component with default values.
        # Individual components could be set with non-default values by calling those methods -
        #  in general they are independent (just combined at the end).
        # Each component has a 'map' (with values from 0-1) for the total visits in all filters
        # and then a maps_per_filter (with values from 0-1) for the fraction of those visits
        # which will happen in each filter. Each component also has a 'nvis' value (nvis at the map max value),
        # which serves to weight each map to their final combined value in the footprint.
        self.read_dustmap()
        if dustfree:
            self._set_dustfree_wfd(self.nvis_wfd_default)
        if mcs:
            self._set_magellanic_clouds(self.nvis_wfd_default)
        if gp:
            self._set_galactic_plane(nvis_gal_A=self.nvis_wfd_default,
                                     nvis_gal_B=int(self.nvis_wfd_default * self.nvis_frac_gp),
                                     nvis_gal_min=int(self.nvis_wfd_default * self.nvis_frac_gp * 0.8))
        if nes:
            self._set_nes(int(self.nvis_wfd_default * self.nvis_frac_nes))
        if scp:
            self._set_scp(120)
        if ddf:
            self._set_ddf()

    def combine_maps(self, trim_overlap=True):
        """Combine the individual maps.

        Parameters
        ----------
        trim_overlap: bool, opt
            If True, look for WFD-like areas which overlap and drop areas according to how much dust there is.
            (due to competing filter balances, the total number of visits can be )
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
                dustfree = hp.smoothing(self.dustfree, fwhm=np.radians(3))
            overlap_dustfree = np.where((dustfree > 0.3) & (overlap == 1))[0]
            overlap_gal = np.where((dustfree <= 0.3) & (overlap == 1))[0]
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
        self.combine_maps()
        return self.total, self.total_perfilter
