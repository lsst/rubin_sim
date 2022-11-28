"""
Photometric precision metrics 
Authors: Sergey Koposov, Thomas Collett
"""

import numpy as np
from rubin_sim.maf.metrics import BaseMetric

__all__ = ["SNMetric", "ThreshSEDSNMetric", "SEDSNMetric"]


twopi = 2.0 * np.pi


class RelRmsMetric(BaseMetric):
    """Relative scatter metric (RMS over median)."""

    def run(self, data_slice, slice_point=None):
        return np.std(data_slice[self.colname]) / np.median(data_slice[self.colname])


class SNMetric(BaseMetric):
    """Calculate the signal to noise metric in a given filter for an object of a given magnitude.
    We assume point source aperture photometry and assume that we do
    the measurement over the stack
    """

    def __init__(
        self,
        m5_col="fiveSigmaDepth",
        seeing_col="finSeeing",
        sky_b_col="filtSkyBrightness",
        exp_t_col="visitExpTime",
        filter_col="filter",
        metric_name="SNMetric",
        filter=None,
        mag=None,
        **kwargs
    ):
        super(SNMetric, self).__init__(
            col=[m5_col, seeing_col, sky_b_col, exp_t_col, filter_col],
            metric_name=metric_name,
            **kwargs
        )
        self.filter = filter
        self.mag = mag

    def run(self, data_slice, slice_point=None):
        # print 'x'
        npoints = len(data_slice[self.seeingCol])
        seeing = data_slice[self.seeingCol]
        depth5 = data_slice[self.m5Col]
        # mag = depth5
        mag = self.mag

        zpt0 = 25.85
        curfilt = self.filter  #'r'
        zpts = {"u": zpt0, "g": zpt0, "r": zpt0, "i": zpt0, "z": zpt0, "y": zpt0}

        gain = 4.5

        zpt_arr = np.zeros(npoints)
        for filt in "ugrizy":
            zpt_arr[data_slice[self.filterCol] == filt] = zpts[filt]
        sky_mag_arcsec = data_slice[self.skyBCol]
        exptime = data_slice[self.expTCol]
        sky_adu = 10 ** (-(sky_mag_arcsec - zpt_arr) / 2.5) * exptime
        sky_adu = sky_adu * np.pi * seeing**2  # adu per seeing circle

        source_fluxes = 10 ** (-mag / 2.5)
        source_adu = 10 ** (-(mag - zpt_arr) / 2.5) * exptime
        err_adu = np.sqrt(source_adu + sky_adu) / np.sqrt(gain)
        err_fluxes = err_adu * (source_fluxes / source_adu)

        ind = data_slice[self.filterCol] == curfilt
        flux0 = source_fluxes
        stack_flux_err = 1.0 / np.sqrt((1 / err_fluxes[ind] ** 2).sum())
        err_mag = 2.5 / np.log(10) * stack_flux_err / flux0
        # return err_mag
        return flux0 / stack_flux_err
        # return (source_fluxes/err_fluxes).mean()
        # 1/0
        # return errMag
        # return 1.25 * np.log10(np.sum(10.**(.8*dataSlice['fiveSigmaDepth'])))


class SEDSNMetric(BaseMetric):
    """Computes the S/Ns for a given SED."""

    def __init__(
        self,
        m5_col="fiveSigmaDepth",
        seeing_col="finSeeing",
        sky_b_col="filtSkyBrightness",
        exp_t_col="visitExpTime",
        filter_col="filter",
        metric_name="SEDSNMetric",
        # filter=None,
        mags=None,
        **kwargs
    ):
        super(SEDSNMetric, self).__init__(
            col=[m5_col, seeing_col, sky_b_col, exp_t_col, filter_col],
            metric_name=metric_name,
            **kwargs
        )
        self.mags = mags
        self.metrics = {}
        for curfilt, curmag in mags.items():
            self.metrics[curfilt] = SNMetric(mag=curmag, filter=curfilt)
        # self.filter = filter
        # self.mag = mag

    def run(self, data_slice, slice_point=None):
        res = {}
        for curf, curm in self.metrics.items():
            curr = curm.run(data_slice, slice_point=slice_point)
            res["sn_" + curf] = curr
        return res

    def reduce_sn_g(self, metric_value):
        # print 'x',metric_value['sn_g']
        return metric_value["sn_g"]

    def reduce_sn_r(self, metric_value):
        # print 'x',metric_value['sn_r']
        return metric_value["sn_r"]

    def reduce_sn_i(self, metric_value):
        return metric_value["sn_i"]


class ThreshSEDSNMetric(BaseMetric):
    """Computes the metric whether the S/N is bigger than the threshold in all the bands for a given SED"""

    def __init__(
        self,
        m5_col="fiveSigmaDepth",
        seeing_col="finSeeing",
        sky_b_col="filtSkyBrightness",
        exp_t_col="visitExpTime",
        filter_col="filter",
        metric_name="ThreshSEDSNMetric",
        snlim=20,
        # filter=None,
        mags=None,
        **kwargs
    ):
        """Instantiate metric."""
        super(ThreshSEDSNMetric, self).__init__(
            col=[m5_col, seeing_col, sky_b_col, exp_t_col, filter_col],
            metric_name=metric_name,
            **kwargs
        )
        self.xmet = SEDSNMetric(mags=mags)
        self.snlim = snlim
        # self.filter = filter
        # self.mag = mag

    def run(self, data_slice, slice_point=None):
        res = self.xmet.run(data_slice, slice_point=slice_point)
        cnt = 0
        for k, v in res.items():
            if v > self.snlim:
                cnt += 1
        if cnt > 0:
            cnt = 1
        return cnt
