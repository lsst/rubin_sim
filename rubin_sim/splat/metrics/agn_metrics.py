__all__ = (
    "QSONumberCountsMetric",
    "SFUncertMetric",
    "AgnTimeLagMetric",
)

import numpy as np
import warnings
from astropy.stats import mad_std
import os

import healpy as hp
from rubin_scheduler.data import get_data_dir
from scipy import interpolate


from rubin_sim.maf.utils import m52snr
from rubin_sim.phot_utils import DustValues
from rubin_sim.splat.utils import eb_v_hp

from .metrics import BaseMetric, CoaddM5ExtinctionMetric


class AgnTimeLagMetric(BaseMetric):
    """XXX--mystery units on what is coming out of this."""

    def __init__(
        self,
        lag=100,
        z=1,
        log=False,
        threshold=2.2,
        calc_type="mean",
        mjd_col="observationStartMJD",
        filter_col="band",
        m5_col="fiveSigmaDepth",
        dust=True,
        g_cutoff=22.0,
        r_cutoff=21.8,
        name=None,
        unit="AGN time lag (XXX)",
        badval=np.nan,
    ):
        self.lag = lag
        self.z = z
        self.log = log
        self.threshold = threshold
        self.calc_type = calc_type
        self.mjd_col = mjd_col
        self.filter_col = filter_col
        self.m5_col = m5_col
        self.badval = badval
        if name is None:
            name = f"AGN_TimeLag_{lag}_days"
        self.dust = dust
        self.g_cutoff = g_cutoff
        self.r_cutoff = r_cutoff
        if dust:
            dust_properties = DustValues()
            self.ax1 = dust_properties.ax1
        super().__init__(
            col=[self.mjd_col, self.filter_col, self.m5_col],
            name=name,
            unit=unit,
        )

    # Calculate NQUIST value for time-lag and sampling time
    # (redshift is included in formula if desired)
    def _get_nquist_value(self, caden, lag, z):
        return lag / ((1 + z) * caden)

    def __call__(self, visits, slice_point=None):
        # Dust extinction
        filterlist = np.unique(visits[self.filter_col])
        if self.dust:
            m5 = np.zeros(len(visits))
            for filtername in filterlist:
                ebv = eb_v_hp(slice_point["nside"], pixels=slice_point["sid"])
                in_filt = np.where(visits[self.filter_col] == filtername)[0]
                a_x = self.ax1[visits[self.filter_col][0]] * ebv
                m5[in_filt] = visits[self.m5_col][in_filt] - a_x
        else:
            m5 = visits[self.m5_col]

        # Identify times which pass magnitude cuts (chosen by AGN contributors)
        mjds = np.zeros(len(visits))
        for filtername in filterlist:
            in_filt = np.where(visits[self.filter_col] == filtername)[0]
            if filtername in ("u", "i", "z", "y", "r", "g"):
                mjds[in_filt] = visits[self.mjd_col][in_filt]
            elif filtername == "g":
                faint = np.where(m5[in_filt] > self.g_cutoff)
                mjds[in_filt][faint] = visits[self.mjd_col][in_filt][faint]
            elif filtername == "r":
                faint = np.where(m5[in_filt] > self.r_cutoff)
                mjds[in_filt][faint] = visits[self.mjd_col][in_filt][faint]
        # Remove the visits which were not faint enough
        mjds = mjds[np.where(mjds > 0)]

        # Calculate differences in time between visits
        mv = np.sort(mjds)
        val = np.diff(mv)
        # If there was only one visit; bail out now.
        if len(val) == 0:
            return self.badval

        # Otherwise summarize the time differences as:
        if self.calc_type == "mean":
            val = np.mean(val)
        elif self.calc_type == "min":
            val = np.min(val)
        elif self.calc_type == "max":
            val = np.max(val)
        else:
            # find the greatest common divisor
            val = np.rint(val).astype(int)
            val = np.gcd.reduce(val)

        # Will always have a value at this point
        nquist = self._get_nquist_value(val, self.lag, self.z)
        if self.log:
            nquist = np.log(nquist)

        # Threshold nquist value is 2.2,
        # hence we are aiming to show values higher than threshold (2.2) value
        threshold = self.threshold
        if self.log:
            threshold = np.log(threshold)

        if nquist < threshold:
            nquist = self.badval

        return nquist


class SFUncertMetric(BaseMetric):
    """Structure Function (SF) Uncertainty Metric.
    Developed on top of LogTGaps

    Adapted from Weixiang Yu & Gordon Richards at:
    https://github.com/RichardsGroup/
    LSST_SF_Metric/blob/main/notebooks/00_SFErrorMetric.ipynb

    Parameters
    ----------
    mag : `float`
        The magnitude of the fiducial object. Default 22.
    times_col : `str`
        Time column name. Defaults to "observationStartMJD".
    all_gaps : `bool`
         Whether to use all gaps (between any two pairs of observations).
         If False, only use consecutive paris. Defaults to True.
    units : `str`
        Unit of this metric. Defaults to "mag".
    bins : `object`
        An array of bin edges.
        Defaults to "np.logspace(0, np.log10(3650), 16)" for a
        total of 15 (final) bins.
    weight : `object`
        The weight assigned to each delta_t bin for deriving the final metric.
        Defaults to flat weighting with sum of 1.
        Should have length 1 less than bins.
    snr_cut : `float`
        Ignore observations below an SNR limit, default 5.
    dust : `bool`
        Apply dust extinction to the fiducial object magnitude. Default True.
    """

    def __init__(
        self,
        mag=22,
        times_col="observationStartMJD",
        m5_col="fiveSigmaDepth",
        all_gaps=True,
        unit="Structure Function Uncert (mag)",
        bins=np.logspace(0, np.log10(3650), 16),
        weight=None,
        name="Structure Function Uncert",
        snr_cut=5,
        filter_col="band",
        dust=True,
        badval=np.nan,
        **kwargs,
    ):
        # Assign metric parameters to instance object
        self.times_col = times_col
        self.m5_col = m5_col
        self.filter_col = filter_col
        self.all_gaps = all_gaps
        self.bins = bins
        self.badval = badval
        if weight is None:
            # If weight is none, set weight so that sum over bins = 1
            self.weight = np.ones(len(self.bins) - 1)
            self.weight /= self.weight.sum()

        self.mag = mag
        self.snr_cut = snr_cut
        self.dust = dust

        super(SFUncertMetric, self).__init__(
            col=[self.times_col, m5_col, filter_col],
            name=name,
            unit=unit,
            **kwargs,
        )
        dust_properties = DustValues()
        self.ax1 = dust_properties.ax1

    def __call__(self, visits, slice_point=None):
        """Code executed at each healpix pixel to compute the metric"""

        df = np.unique(visits[self.filter_col])
        if np.size(df) > 1:
            msg = """Running structure function on multiple filters simultaneously.
                     Should probably change your SQL query to limit to a single filter."""
            warnings.warn(msg)
        if self.dust:
            ebv = eb_v_hp(slice_point["nside"], pixels=slice_point["sid"])
            a_x = self.ax1[visits[self.filter_col][0]] * ebv
            extincted_mag = self.mag + a_x
        else:
            extincted_mag = self.mag
        snr = m52snr(extincted_mag, visits[self.m5_col])
        bright_enough = np.where(snr > self.snr_cut)[0]

        # If the total number of visits < 2, mask as bad pixel
        if visits[bright_enough].size < 2:
            return self.badval

        # sort data by time column
        order = np.argsort(visits[self.times_col][bright_enough])
        times = visits[self.times_col][bright_enough][order]
        # Using the simple Gaussian approximation for magnitude uncertainty.
        mag_err = 2.5 * np.log10(1.0 + 1.0 / snr[bright_enough][order])

        # check if use all gaps (between any pairs of observations)
        if self.all_gaps:
            # use the vectorized method
            dt_matrix = times.reshape((1, times.size)) - times.reshape((times.size, 1))
            dts = dt_matrix[dt_matrix > 0].flatten().astype(np.float16)
        else:
            dts = np.diff(times)

        # bin delta_t using provided bins;
        # if zero pair found at any delta_t bin,
        # replace 0 with 0.01 to avoid the exploding 1/sqrt(n) term
        # in this metric
        result, bins = np.histogram(dts, self.bins)
        new_result = np.where(result > 0, result, 0.01)

        # compute photometric_error^2 population variance and population mean
        # note that variance is replaced by median_absolute_deviate^2
        # mean is replaced by median in this implementation to make it robust
        # to outliers in simulations (e.g., dcr simulations)
        err_var = mag_err**2
        err_var_mu = np.median(err_var)
        err_var_std = mad_std(err_var)

        # compute SF error
        sf_var_dt = 2 * (err_var_mu + err_var_std / np.sqrt(new_result))
        sf_var_metric = np.sum(sf_var_dt * self.weight)

        return sf_var_metric


class QSONumberCountsMetric(BaseMetric):
    """Calculate the number of quasars expected with SNR>=5
    according to the Shen et al. (2020) QLF - model A in the redshift
    range zmin < z < zmax.

    The 5 sigma depths are obtained using the ExgalM5 metric.
    Only quasars fainter than the saturation magnitude are counted.

    By default, zmin is 0.3 and zmax is the minimum between 6.7 and the
    redshift at which the Lyman break matches the effective wavelength
    of the band. For bands izy, zmax is 6.7. This default choice is to
    match Table 10.2 for i-band quasar counts in the LSST Science book.
    """

    def __init__(
        self,
        lsst_filter,
        m5_col="fiveSigmaDepth",
        unit="N QSO",
        extinction_cut=1.0,
        filter_col="band",
        name="QSONumberCountsMetric",
        qlf_module="Shen20",
        qlf_model="A",
        sed_model="Richards06",
        zmin=0.3,
        zmax=None,
        badval=np.nan,
        **kwargs,
    ):
        # Declare the effective wavelengths.
        self.effwavelen = {
            "u": 367.0,
            "g": 482.5,
            "r": 622.2,
            "i": 754.5,
            "z": 869.1,
            "y": 971.0,
        }
        self.badval = badval

        # Dust Extinction limit.
        # Regions with larger extinction and dropped from the counting.
        self.extinction_cut = extinction_cut

        # Save the filter information.
        self.filter_col = filter_col
        self.lsst_filter = lsst_filter

        # Save zmin and zmax, or set zmax to the default value.
        # The default zmax is the lower number between 6.7 and the
        # redshift at which the Lyman break (91.2nm) hits the
        # effective wavelength of the filter.
        # Note that this means that for i, z and y,
        # the default value for zmax is 6.7
        self.zmin = zmin
        if zmax is None:
            zmax = np.min([6.7, self.effwavelen[self.lsst_filter] / 91.2 - 1.0])
        self.zmax = zmax

        # This calculation uses the ExgalM5 metric. So declare that here.
        self.exgal_m5 = CoaddM5ExtinctionMetric(self.lsst_filter, col=m5_col)

        # Save the input parameters that relate to the QLF model.
        self.qlf_module = qlf_module
        self.qlf_model = qlf_model
        self.sed_model = sed_model

        # Read the long tables, which the number of quasars expected
        # for a given band, qlf_module and qlf_model in a range of
        # redshifts and magnitudes.
        table_name = "Long_Table.LSST{0}.{1}.{2}.{3}.txt".format(
            self.lsst_filter, self.qlf_module, self.qlf_model, self.sed_model
        )
        data_dir = os.path.join(get_data_dir(), "maf", "quasarNumberCounts")
        filename = os.path.join(data_dir, table_name)
        with open(filename, "r") as f:
            mags = np.array([float(x) for x in f.readline().split()])
            zs = np.array([float(x) for x in f.readline().split()])
        mz_data = np.loadtxt(filename, skiprows=2)

        # Make the long table cumulative.
        c_mz_data = np.zeros((mz_data.shape[0] + 1, mz_data.shape[1] + 1))
        c_mz_data[1:, 1:] = mz_data
        c_mz_data = np.cumsum(c_mz_data, axis=0)
        c_mz_data = np.cumsum(c_mz_data, axis=1)

        # Create a 2D interpolation object for the long table.
        # self.nqso_cumulative = interpolate.interp2d(zs[:-1], mags[:-1],
        # #c_mz_data[:-1, :-1], kind="cubic")
        self.nqso_cumulative_aux = interpolate.RectBivariateSpline(
            zs[:-1], mags[:-1], c_mz_data[:-1, :-1].T, kx=3, ky=3
        )

        self.nqso_cumulative = lambda z_new, m_new: self.nqso_cumulative_aux(z_new, m_new).T[0]

        super().__init__(
            col=[m5_col, filter_col, "saturation_mag"],
            name=name,
            unit=unit,
            **kwargs,
        )

    def __call__(self, visits, slice_point=None):
        # exclude areas with high extinction
        ebv = eb_v_hp(slice_point["nside"], pixels=slice_point["sid"])
        if ebv > self.extinction_cut:
            return self.badval

        # For the visits, get the 5 sigma limiting magnitude.
        d_s = visits[visits[self.filter_col] == self.lsst_filter]

        if np.size(d_s) == 0:
            return self.badval

        mlim5 = self.exgal_m5(d_s, slice_point)

        # Get the slicer pixel area.
        nside = slice_point["nside"]
        pix_area = hp.nside2pixarea(nside, degrees=True)

        # tranform that limiting magnitude into an expected number of quasars.
        # If there is more than one, take the faintest.
        m_bright = np.max(d_s["saturation_mag"])
        n11 = self.nqso_cumulative(self.zmin, m_bright)
        n12 = self.nqso_cumulative(self.zmin, mlim5)
        n21 = self.nqso_cumulative(self.zmax, m_bright)
        n22 = self.nqso_cumulative(self.zmax, mlim5)

        nqso = (n22 - n21 - n12 + n11) * pix_area
        return np.asarray(nqso).item()
