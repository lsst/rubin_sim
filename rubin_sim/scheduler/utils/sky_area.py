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
    """
    Generate survey footprint maps in each filter.

    Parameters
    ----------
    nside : `int` (32)
        Healpix nside (32)
    dust_limit : `float` (0.199)
        E(B-V) limit for dust extinction. Default of 0.199.
    smoothing_cutoff : `float` (0.45)
        Magic number
    smoothing_beam : `float` (10)
        Magic number
    """
    def __init__(self, nside=32, dust_limit=0.199, smoothing_cutoff=0.45, smoothing_beam=10):

        self.nside = nside
        self.hpid = np.arange(0, hp.nside2npix(nside))
        self.read_dustmap()

        # Array to hold the labels for each pixel
        self.pix_labels = np.zeros(hp.nside2npix(self.nside), dtype='U20')
        self.healmaps = np.zeros(hp.nside2npix(self.nside), dtype=list(zip(['u', 'g', 'r', 'i', 'z', 'y'], [float]*7)))

        # Ra/dec in degrees and other coordinates
        self.ra, self.dec = hp.pix2ang(nside, self.hpid, lonlat=True)
        self.coord = SkyCoord(ra=self.ra * u.deg, dec=self.dec * u.deg, frame='icrs')
        self.eclip_lat = self.coord.barycentrictrueecliptic.lat.deg
        self.eclip_lon = self.coord.barycentrictrueecliptic.lon.deg
        self.gal_lon = self.coord.galactic.l.deg
        self.gal_lat = self.coord.galactic.b.deg

        # Set the low extinction area
        self.low_dust = np.where((self.dustmap < dust_limit), 1, 0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            self.low_dust = hp.smoothing(self.low_dust, fwhm=np.radians(smoothing_beam))
        self.low_dust = np.where(self.low_dust > smoothing_cutoff, 1, 0)

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

    def _set_circular_region(self, ra_center, dec_center, radius):
        # find the healpixels that cover a circle of radius radius around ra/dec center (deg)
        result = np.zeros(len(self.ra))
        distance = rs_utils._angularSeparation(np.radians(ra_center), np.radians(dec_center),
                                               np.radians(self.ra), np.radians(self.dec))
        result[np.where(distance < np.radians(radius))] = 1
        return result

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
            HEALpix array with 1 for bulge pixels, 0 otherwise
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
            HEALpix array with 1 for bulge pixels, 0 otherwise
        """
        bulge = np.where(np.abs(self.gal_lat) < lat_width, 1, 0)
        # This is NOT the shortest distance between the angles.
        gp_length = (gal_long2 - gal_long1) % 360
        # If the length is greater than 0 then we can add additional cuts.
        if gp_length > 0:
            # First, remove anything outside the gal_long1/gal_long2 region.
            bulge = np.where(((self.gal_lon - gal_long1) % 360) < gp_length, bulge, 0)
        return bulge

    def add_magellanic_clouds(self, filter_ratios,
                              lmc_radius=8, smc_radius=5, label='LMC_SMC',
                              lmc_ra=80.893860, lmc_dec=-69.756126, smc_ra=13.186588, smc_dec=-72.828599):
        """Add the Magellanic Clouds to the map

        Parameters
        ----------
        lmc_radius : `float`, optional
            Radius around the LMC to include (degrees).
        smc_radius : `float`, optional
            Radius around the SMC to include (degrees).
        filter_ratios : `dict` {`str` : `float`}, optional
            How to distribute visits between different filters.
            Default uses {'u': 0.31, 'g': 0.44, 'r': 1., 'i': 1., 'z': 0.9, 'y': 0.9}
        label : str
            What to label the pixels in the map
        """
        temp_map = np.zeros(hp.nside2npix(self.nside))
        # Define the LMC pixels
        temp_map += self._set_circular_region(lmc_ra, lmc_dec, lmc_radius)
        # Define the SMC pixels
        temp_map += self._set_circular_region(smc_ra, smc_dec, smc_radius)
        # Add a simple bridge between the two - to remove the gap
        mc_dec_min = self.dec[np.where(temp_map > 0)].min()
        mc_dec_max = self.dec[np.where(temp_map > 0)].max()
        temp_map += np.where(((self.ra > smc_ra) & (self.ra < lmc_ra))
                             & ((self.dec > mc_dec_min) & (self.dec < mc_dec_max)), 1, 0)

        # Don't overide any pixels that have already been designated
        indx = np.where((temp_map > 0) & (self.pix_labels == ''))

        self.pix_labels[indx] = label
        for filtername in filter_ratios:
            self.healmaps[filtername][indx] = filter_ratios[filtername]

    def add_scp(self, filter_ratios, dec_max=-60,
                label='scp'):

        indx = np.where((self.dec < dec_max) & (self.pix_labels == ''))

        self.pix_labels[indx] = label
        for filtername in filter_ratios:
            self.healmaps[filtername][indx] = filter_ratios[filtername]

    def add_bulge(self, filter_ratios, label='bulge', gal_long1=335, gal_long2=25,
                  gal_lat_width_max=23, center_width=12, end_width=4,
                  gal_dec_max=12):
        """Add bulge to the map
        """
        b1 = self._set_bulge_diamond(center_width=center_width, end_width=end_width,
                                     gal_long1=gal_long1, gal_long2=gal_long2)
        b2 = self._set_bulge_rectangle(gal_lat_width_max, gal_long1, gal_long2)
        b2[np.where(self.gal_lat > 0)] = 0

        bulge = b1 + b2

        bulge[np.where(np.abs(self.gal_lat) > gal_lat_width_max)] = 0
        indx = np.where((bulge > 0) & (self.pix_labels == ''))
        self.pix_labels[indx] = label
        for filtername in filter_ratios:
            self.healmaps[filtername][indx] = filter_ratios[filtername]

    def add_lowdust_wfd(self, filter_ratios, dec_min=-70,
                        dec_max=15, adjust_halves=12, label='lowdust'):

        dustfree = np.where((self.dec > dec_min) & (self.dec < dec_max)
                            & (self.low_dust == 1), 1, 0)

        dustfree[np.where(self.low_dust == 0)] = 0

        if adjust_halves > 0:
            dustfree = np.where((self.gal_lat < 0) & (self.dec > dec_max - adjust_halves),
                                0, dustfree)

        indx = np.where((dustfree > 0) & (self.pix_labels == ''))
        self.pix_labels[indx] = label
        for filtername in filter_ratios:
            self.healmaps[filtername][indx] = filter_ratios[filtername]

    def add_dusty_plane(self, filter_ratios, dec_min=-90, dec_max=15,
                        label='dusty_plane'):

        dusty = np.where(((self.dec > dec_min) & (self.dec < dec_max)
                          & (self.low_dust == 0)), 1, 0)

        indx = np.where((dusty > 0) & (self.pix_labels == ''))
        self.pix_labels[indx] = label
        for filtername in filter_ratios:
            self.healmaps[filtername][indx] = filter_ratios[filtername]

    def add_nes(self, filter_ratios, label='nes', eclat_min=-10, eclat_max=10, eclip_dec_min=0,
                glon_limit=45.):
        """
        """
        nes = np.where(((self.eclip_lat > eclat_min) | (self.dec > eclip_dec_min))
                       & (self.eclip_lat < eclat_max), 1, 0)

        nes[np.where(self.gal_lon < glon_limit)] = 0
        nes[np.where(self.gal_lon > (360-glon_limit))] = 0

        indx = np.where((nes > 0) & (self.pix_labels == ''))
        self.pix_labels[indx] = label
        for filtername in filter_ratios:
            self.healmaps[filtername][indx] = filter_ratios[filtername]

    def return_maps(self,
                    magellenic_clouds_ratios={'u': 0.32, 'g': 0.4, 'r': 1.0, 'i': 1.0, 'z': 0.9, 'y': 0.9},
                    scp_ratios={'u': 0.1, 'g': 0.1, 'r': 0.1, 'i': 0.1, 'z': 0.1, 'y': 0.1},
                    nes_ratios={'g': 0.28, 'r': 0.4, 'i': 0.4, 'z': 0.28},
                    dusty_plane_ratios={'u': 0.1, 'g': 0.28, 'r': 0.28, 'i': 0.28, 'z': 0.28, 'y': 0.1},
                    low_dust_ratios={'u': 0.32, 'g': 0.4, 'r': 1.0, 'i': 1.0, 'z': 0.9, 'y': 0.9},
                    bulge_ratios={'u': 0.18, 'g': 1.0, 'r': 1.05, 'i': 1.05, 'z': 1.0, 'y': 0.23}):
        """
        Returns
        --------
        self.total, self.total_perfilter : `np.ndarray`, `np.ndarray`
            HEALPix maps 
        """
        self.add_magellanic_clouds(magellenic_clouds_ratios)
        self.add_lowdust_wfd(low_dust_ratios)
        self.add_bulge(bulge_ratios)
        self.add_nes(nes_ratios)
        self.add_dusty_plane(dusty_plane_ratios)
        self.add_scp(scp_ratios)

        return self.healmaps, self.pix_labels
