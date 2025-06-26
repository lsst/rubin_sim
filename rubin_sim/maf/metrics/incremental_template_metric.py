import numpy as np
from rubin_scheduler.utils import SURVEY_START_MJD, Site, SysEngVals, calc_season

from .base_metric import BaseMetric

__all__ = ["TemplateTime"]


class TemplateTime(BaseMetric):
    """Find the time at which we expect to hit incremental template
    availability.

    Note that there are some complications to real template generation
    that make this an approximation and not an exact answer -- one aspect is
    that templates are generated in `patches` and not per pixel. However, it
    may be possible to generate parts of these patches at about 5 arcsecond
    scales, which implies running with a healpix slicer at nside=512 or 1024.

    Parameters
    ----------
    n_images_in_template : `int`, opt
        Number of qualified visits required for template generation.
        Default 3.
    seeing_threshold : `float`, opt
        The threshold for counting "good seeing" images, in arcseconds.
        This is at zenith, at 500 nm - it will be scaled to the slicer's
        best possible value at the relevant declination and in the template's
        bandpass.
    seeing_percentile : `float`, opt
        When evaluating the local seeing distribution, if there
        `n_images_in_template` better than `seeing_percentile`, go ahead
        and make the template. This allows template creation in locations
        with poor seeing compared to
        Maximum percentile seeing to allow in the qualified images (0 - 100).
        The maximum of seeing_threshold (scaled) or the seeing at
        seeing_percentil will be used as the cutoff for template creation.
    m5_threshold : `float` or None, opt
        The threshold for counting an image as "good m5". This is also
        scaled to the expected best minimum airmass/seeing given the
        slicepoints' declination.
    m5_percentile : `float`, opt
        Maximum percentile m5 to allow in the qualified images (0 - 100).
        The minimum of m5_threshold (scaled) or m5 resulting from
        m5_percentile will be used as the cutoff for template creation.
    seeing_col : `str`, opt
        Name of the seeing column to use.
    m5_col : `str`, opt
        Name of the five sigma depth columns.
    night_col : `str`, opt
        Name of the column describing the night of the visit.
    mjd_col : `str`, opt
        Name of column describing time of the visit
    band_col : `str`, opt
        Name of column describing band
    mjd_start : `float`, opt
        The starting time of the survey, to use to count "nights before
        template creation", etc.
    site_latiude_rad : `float`, opt
        The latitude of the site, used to determine minimum best airmass.
    """

    def __init__(
        self,
        n_images_in_template=3,
        seeing_threshold=0.8,
        m5_threshold=None,
        seeing_percentile=50,
        m5_percentile=50,
        seeing_col="seeingFwhmEff",
        m5_col="fiveSigmaDepth",
        night_col="night",
        mjd_col="observationStartMJD",
        band_col="band",
        mjd_start=None,
        filter_col=None,
        site_latitude_rad=Site("LSST").latitude_rad,
        **kwargs,
    ):
        self.n_images_in_template = n_images_in_template
        self.seeing_threshold = seeing_threshold
        if m5_threshold is None:
            # Update these - currently just m5 defaults for v1.9
            m5_threshold = {"u": 23.7, "g": 24.9, "r": 24.5, "i": 24.1, "z": 23.5, "y": 22.5}
        self.m5_threshold = m5_threshold
        self.seeing_percentile = seeing_percentile
        self.m5_percentile = m5_percentile
        self.seeing_col = seeing_col
        self.m5_col = m5_col
        self.night_col = night_col
        self.mjd_col = mjd_col
        self.band_col = band_col
        if filter_col is not None:
            self.band_col = filter_col
        if "metric_name" in kwargs:
            self.metric_name = kwargs["metric_name"]
            del kwargs["metric_name"]
        else:
            self.metric_name = "TemplateTime"
        if mjd_start is None:
            self.mjd_start = SURVEY_START_MJD
        else:
            self.mjd_start = mjd_start

        self.site_latitude_rad = site_latitude_rad

        sev = SysEngVals()
        self.eff_wavelens = dict([(b, sev.eff_wavelengths[b]) for b in "ugrizy"])
        self.extinction_coeffs = {
            "u": -0.46,
            "g": -0.21,
            "r": -0.12,
            "i": -0.07,
            "z": -0.06,
            "y": -0.10,
        }

        super().__init__(
            col=[self.seeing_col, self.m5_col, self.night_col, self.mjd_col, self.band_col],
            metric_name=self.metric_name,
            units="days",
            **kwargs,
        )

    def run(self, data_slice, slice_point):
        result = {}

        # Bail if not enough visits at all
        if len(data_slice) < self.n_images_in_template:
            return self.badval

        # Check that the visits are sorted in time
        data_slice.sort(order=self.mjd_col)

        min_z_possible = np.abs(slice_point["dec"] - self.site_latitude_rad)
        min_airmass_possible = 1.0 / np.cos(min_z_possible)
        best_seeing_possible = self.seeing_threshold * np.power(min_airmass_possible, 0.6)
        current_band = np.unique(data_slice[self.band_col])[0]
        best_seeing_possible = best_seeing_possible * np.power(500 / self.eff_wavelens[current_band], 0.3)

        percentile_seeing = np.percentile(data_slice[self.seeing_col], self.seeing_percentile)
        bench_seeing = np.max([best_seeing_possible, percentile_seeing])

        try:
            m5_threshold = self.m5_threshold[current_band]
        except TypeError:
            m5_threshold = self.m5_threshold

        best_m5 = (
            m5_threshold
            - self.extinction_coeffs[current_band] * (1 - min_airmass_possible)
            + 2.5 * np.log10(self.seeing_threshold / best_seeing_possible)
        )
        percentile_m5 = np.percentile(data_slice[self.m5_col], 100 - self.m5_percentile)
        bench_m5 = np.min([best_m5, percentile_m5])

        seeing_ok = np.where(data_slice[self.seeing_col] <= bench_seeing, True, False)

        m5_ok = np.where(data_slice[self.m5_col] >= bench_m5, True, False)

        ok_template_input = np.where(seeing_ok & m5_ok)[0]
        if (
            len(ok_template_input) < self.n_images_in_template
        ):  # If seeing_ok and/or m5_ok are "false", returned as bad value
            return self.badval

        idx_template_created = ok_template_input[
            self.n_images_in_template - 1
        ]  # Last image needed for template
        idx_template_inputs = ok_template_input[: self.n_images_in_template]  # Images included in template

        night_template_created = data_slice[self.night_col][idx_template_created]

        frac_into_season_template_created = calc_season(
            np.degrees(slice_point["ra"]),
            data_slice[self.mjd_col][idx_template_created],
            mjd_start=self.mjd_start,
        )

        where_template = (
            data_slice[self.night_col] > night_template_created
        )  # of later images where we have a template
        n_images_with_template = np.sum(where_template)

        template_m5 = 1.25 * np.log10(np.sum(10.0 ** (0.8 * data_slice[self.m5_col][idx_template_inputs])))

        # derive variance-weighted PSF width
        # 1. normalize the weights
        template_f0 = np.sqrt(np.sum(25 * 10 ** (0.8 * data_slice[self.m5_col][idx_template_inputs])))
        # 2. compute per-input noise
        image_noise = template_f0 / 5 * 10 ** (-0.4 * data_slice[self.m5_col])
        # 3. used variance-weighted sum of squares to derive template seeing
        template_seeing = np.sqrt(
            np.sum(
                (data_slice[self.seeing_col][idx_template_inputs] / image_noise[idx_template_inputs]) ** 2.0
            )
        )

        delta_m5 = -2.5 * np.log10(np.sqrt(1.0 + 10 ** (-0.8 * (template_m5 - data_slice[self.m5_col]))))
        diff_m5s = data_slice[self.m5_col] + delta_m5

        n_alerts_per_diffim = 1e4 * 10 ** (0.6 * (diff_m5s - 24.7))

        result["bench_seeing"] = bench_seeing
        result["bench_m5"] = bench_m5
        result["night_template_created"] = night_template_created
        result["frac_into_season_template_created"] = frac_into_season_template_created
        result["n_images_until_template"] = idx_template_created + 1
        result["n_images_with_template"] = n_images_with_template
        result["template_m5"] = template_m5
        result["template_seeing"] = template_seeing
        result["total_alerts"] = np.sum(n_alerts_per_diffim[where_template])
        result["template_input_m5s"] = data_slice[self.m5_col][idx_template_inputs]
        result["template_input_seeing"] = data_slice[self.seeing_col][idx_template_inputs]
        result["fraction_better_template_seeing"] = np.sum(
            (data_slice[self.seeing_col][where_template] > template_seeing)
        ) / np.sum(where_template)
        result["diffim_lc"] = {
            "mjd": data_slice[self.mjd_col][where_template],
            "night": data_slice[self.night_col][where_template],
            "band": data_slice[self.band_col][where_template],
            "diff_m5": diff_m5s[where_template],
            "science_m5": data_slice[self.m5_col][where_template],
            "template_m5": template_m5 * np.ones(np.sum(where_template)),
            "science_seeing": data_slice[self.seeing_col][where_template],
            "template_seeing": template_seeing * np.ones(np.sum(where_template)),
            "n_alerts": n_alerts_per_diffim[where_template],
        }

        return result

    def reduce_night_template_created(self, metric_val):  # returns night of template creation
        return metric_val["night_template_created"]

    def reduce_n_images_until_template(
        self, metric_val
    ):  # returns number of images needed to complete template
        return metric_val["n_images_until_template"]

    def reduce_n_images_with_template(self, metric_val):  # returns number of images with a template
        return metric_val["n_images_with_template"]

    def reduce_template_m5(self, metric_val):  # calculated coadded m5 of resulting template
        return metric_val["template_m5"]

    def reduce_template_seeing(self, metric_val):  # calculated seeing of resulting template
        return metric_val["template_seeing"]

    def reduce_fraction_better_template_seeing(self, metric_val):  # calculated seeing of resulting template
        return metric_val["fraction_better_template_seeing"]

    def reduce_total_alerts(self, metric_val):
        return metric_val["total_alerts"]

    def reduce_frac_into_season(self, metric_val):
        return metric_val["frac_into_season_template_created"]
