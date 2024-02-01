__all__ = ("NgalScaleMetric", "NlcPointsMetric")

import healpy as hp
import numpy as np

from rubin_sim.maf.maps import TrilegalDensityMap
from rubin_sim.phot_utils import DustValues

from .base_metric import BaseMetric


class NgalScaleMetric(BaseMetric):
    """Approximate number of galaxies, scaled by median seeing.

    Parameters
    ----------
    a_max : `float`
        The maximum dust extinction to allow. Anything with higher dust
        extinction is considered to have zero usable galaxies.
    m5min : `float`
        The minimum coadded 5-sigma depth to allow. Anything less is
        considered to have zero usable galaxies.
    filter : `str`
        The filter to use. Any visits in other filters are ignored.
    """

    def __init__(
        self,
        seeing_col="seeingFwhmEff",
        m5_col="fiveSigmaDepth",
        metric_name="NgalScale",
        filtername="i",
        a_max=0.2,
        m5min=26.0,
        filter_col="filter",
        **kwargs,
    ):
        maps = ["DustMap"]
        units = "N gals"
        self.seeing_col = seeing_col
        self.m5_col = m5_col
        self.filtername = filtername
        self.filter_col = filter_col
        self.a_max = a_max
        self.m5min = m5min

        super().__init__(
            col=[self.m5_col, self.filter_col, self.seeing_col],
            maps=maps,
            units=units,
            metric_name=metric_name,
            **kwargs,
        )
        dust_properties = DustValues()
        self.ax1 = dust_properties.ax1

    def run(self, data_slice, slice_point):
        # I'm a little confused why there's a dust cut and an M5 cut,
        # but whatever
        a_x = self.ax1[data_slice[self.filter_col][0]] * slice_point["ebv"]
        if a_x > self.a_max:
            return 0

        in_filt = np.where(data_slice[self.filter_col] == self.filtername)
        coadd_m5 = 1.25 * np.log10(np.sum(10.0 ** (0.8 * data_slice[self.m5_col][in_filt])))
        if coadd_m5 < self.m5min:
            return 0

        theta = np.median(data_slice[self.seeing_col])
        # N gals per arcmin2
        ngal_per_arcmin2 = 57 * (0.75 / theta) ** 1.5

        area = hp.nside2pixarea(slice_point["nside"], degrees=True) * 3600.0

        ngal = ngal_per_arcmin2 * area
        return ngal


class NlcPointsMetric(BaseMetric):
    """Number of points in stellar light curves

    Parameters
    ----------
    ndpmin : `int`
        The number of points to demand on a lightcurve in a single
        filter to have that light curve qualify.
    mags : `float`
        The magnitude of our fiducial object (maybe make it a dict in the
        future to support arbitrary colors).
    maps : `list` [`~rubin_sim.maf.map`] or None
        List of stellar density maps to use.
        Default of None loads Trilegal maps.
    nside : `int`
        The nside is needed to make sure the loaded maps
        match the slicer nside.
    """

    def __init__(
        self,
        ndpmin=10,
        mags=21.0,
        m5_col="fiveSigmaDepth",
        filter_col="filter",
        metric_name="NlcPoints",
        maps=None,
        nside=128,
        **kwargs,
    ):
        units = "N LC points"
        self.ndpmin = ndpmin
        self.mags = mags
        self.m5_col = m5_col
        self.filter_col = filter_col
        self.nside = nside
        if maps is None:
            maps = [TrilegalDensityMap(filtername=fn, nside=nside) for fn in "ugrizy"]
        super().__init__(
            col=[self.m5_col, self.filter_col],
            maps=maps,
            units=units,
            metric_name=metric_name,
            **kwargs,
        )

    def run(self, data_slice, slice_point):
        if self.nside != slice_point["nside"]:
            raise ValueError("nside of metric does not match slicer")
        pix_area = hp.nside2pixarea(slice_point["nside"], degrees=True)

        nlcpoints = 0
        # Let's do it per filter
        for filtername in np.unique(data_slice[self.filter_col]):
            in_filt = np.where(
                (data_slice[self.filter_col] == filtername) & (data_slice[self.m5_col] > self.mags)
            )[0]
            n_obs = np.size(in_filt)
            if n_obs > self.ndpmin:
                nstars = (
                    np.interp(
                        self.mags,
                        slice_point[f"starMapBins_{filtername}"][1:],
                        slice_point[f"starLumFunc_{filtername}"],
                    )
                    * pix_area
                )
                nlcpoints += n_obs * nstars

        return nlcpoints
