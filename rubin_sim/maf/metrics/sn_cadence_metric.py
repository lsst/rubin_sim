__all__ = ("SNCadenceMetric",)

import numpy as np

import rubin_sim.maf.metrics as metrics


class SNCadenceMetric(metrics.BaseMetric):
    """
    Metric to estimate the redshift limit for faint supernovae
    (x1,color) = (-2.0,0.2)

    Parameters
    ----------
    list : `str`, optional
        Name of the columns used to estimate the metric
    coadd :  `bool`, optional
        to make "coaddition" per night (uses snStacker)
        Default True
    lim_sn : `class`, optional
       Reference data used to estimate redshift values (interpolation)
    """

    def __init__(
        self,
        metric_name="SNCadenceMetric",
        mjd_col="observationStartMJD",
        ra_col="fieldRA",
        dec_col="fieldDec",
        filter_col="filter",
        m5_col="fiveSigmaDepth",
        exptime_col="visitExposureTime",
        night_col="night",
        obsid_col="observationId",
        nexp_col="numExposures",
        vistime_col="visitTime",
        coadd=True,
        lim_sn=None,
        **kwargs,
    ):
        self.mjd_col = mjd_col
        self.m5_col = m5_col
        self.filter_col = filter_col
        self.ra_col = ra_col
        self.dec_col = dec_col
        self.exptime_col = exptime_col
        self.season_col = "season"
        self.night_col = night_col
        self.obsid_col = obsid_col
        self.nexp_col = nexp_col
        self.vistime_col = vistime_col

        cols = [
            self.night_col,
            self.m5_col,
            self.filter_col,
            self.mjd_col,
            self.obsid_col,
            self.nexp_col,
            self.vistime_col,
            self.exptime_col,
            self.season_col,
        ]
        if coadd:
            cols += ["coadd"]

        super(SNCadenceMetric, self).__init__(col=cols, metric_name=metric_name, **kwargs)

        self.filter_names = np.array(["u", "g", "r", "i", "z", "y"])

        self.lim_sn = lim_sn

    def run(self, data_slice, slice_point=None):
        # Cut down to only include filters in correct wave range.

        good_filters = np.isin(data_slice["filter"], self.filter_names)
        data_slice = data_slice[good_filters]
        if data_slice.size == 0:
            return None
        data_slice.sort(order=self.mjd_col)

        r = []
        field_ra = np.mean(data_slice[self.ra_col])
        field_dec = np.mean(data_slice[self.dec_col])
        band = np.unique(data_slice[self.filter_col])[0]

        sel = data_slice
        bins = np.arange(np.floor(sel[self.mjd_col].min()), np.ceil(sel[self.mjd_col].max()), 1.0)
        c, b = np.histogram(sel[self.mjd_col], bins=bins)
        if (c.mean() < 1.0e-8) | np.isnan(c).any() | np.isnan(c.mean()):
            cadence = 0.0
        else:
            cadence = 1.0 / c.mean()
        # time_diff = sel[self.mjd_col][1:]-sel[self.mjd_col][:-1]
        r.append((field_ra, field_dec, band, np.mean(sel[self.m5_col]), cadence))

        res = np.rec.fromrecords(r, names=["field_ra", "field_dec", "band", "m5_mean", "cadence_mean"])

        zref = self.lim_sn.interp_griddata(res)

        if np.isnan(zref):
            zref = self.badval

        return zref
