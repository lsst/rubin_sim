import numpy as np

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
    seeing_percentile : `float`, opt
        Maximum percentile seeing to allow in the qualified images (0 - 100).

        Default 50.
    m5_percentile : `float`, opt
        Maximum percentile m5 to allow in the qualified images (0 - 100).

        Default 50.
    seeing_col : `str`, opt
        Name of the seeing column to use.
    m5_col : `str`, opt
        Name of the five sigma depth columns.
    night_col : `str`, opt
        Name of the column describing the night of the visit.
    mjd_col : `str`, opt
        Name of column describing time of the visit
    filter_col : `str`, opt
        Name of column describing filter
    """

    def __init__(
        self,
        n_images_in_template=3,
        seeing_percentile=50,
        m5_percentile=50,
        seeing_col="seeingFwhmEff",
        m5_col="fiveSigmaDepth",
        night_col="night",
        mjd_col="observationStartMJD",
        filter_col="filter",
        **kwargs,
    ):
        self.n_images_in_template = n_images_in_template
        self.seeing_percentile = seeing_percentile
        self.m5_percentile = m5_percentile
        self.seeing_col = seeing_col
        self.m5_col = m5_col
        self.night_col = night_col
        self.mjd_col = mjd_col
        self.filter_col = filter_col
        if "metric_name" in kwargs:
            self.metric_name = kwargs["metric_name"]
            del kwargs["metric_name"]
        else:
            self.metric_name = "TemplateTime"
        super().__init__(
            col=[self.seeing_col, self.m5_col, self.night_col, self.mjd_col, self.filter_col],
            metric_name=self.metric_name,
            units="days",
            **kwargs,
        )

    def run(self, data_slice, slice_point=None):
        result = {}

        # Bail if not enough visits at all
        if len(data_slice) < self.n_images_in_template:
            return self.badval

        # Check that the visits are sorted in time
        data_slice.sort(order=self.mjd_col)

        # Find the threshold seeing and m5
        bench_seeing = np.percentile(data_slice[self.seeing_col], self.seeing_percentile)
        bench_m5 = np.percentile(data_slice[self.m5_col], 100 - self.m5_percentile)

        seeing_ok = np.where(data_slice[self.seeing_col] < bench_seeing, True, False)

        m5_ok = np.where(data_slice[self.m5_col] > bench_m5, True, False)

        ok_template_input = np.where(seeing_ok & m5_ok)[0]
        if (
            len(ok_template_input) < self.n_images_in_template
        ):  # If seeing_ok and/or m5_ok are "false", returned as bad value
            return self.badval

        idx_template_created = ok_template_input[
            self.n_images_in_template - 1
        ]  # Last image needed for template
        idx_template_inputs = ok_template_input[: self.n_images_in_template]  # Images included in template

        Night_template_created = data_slice[self.night_col][idx_template_created]
        N_nights_without_template = Night_template_created - data_slice[self.night_col][0]

        where_template = (
            data_slice[self.night_col] > Night_template_created
        )  # of later images where we have a template
        N_images_with_template = np.sum(where_template)

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

        result["Night_template_created"] = Night_template_created
        result["N_nights_without_template"] = N_nights_without_template
        result["N_images_until_template"] = idx_template_created + 1
        result["N_images_with_template"] = N_images_with_template
        result["Template_m5"] = template_m5
        result["Template_seeing"] = template_seeing
        result["Total_alerts"] = np.sum(n_alerts_per_diffim[where_template])
        result["Template_input_m5s"] = data_slice[self.m5_col][idx_template_inputs]
        result["Template_input_seeing"] = data_slice[self.seeing_col][idx_template_inputs]
        result["Fraction_better_template_seeing"] = np.sum(
            (data_slice[self.seeing_col][where_template] > template_seeing)
        ) / np.sum(where_template)
        result["Diffim_lc"] = {
            "mjd": data_slice[self.mjd_col][where_template],
            "night": data_slice[self.night_col][where_template],
            "band": data_slice[self.filter_col][where_template],
            "diff_m5": diff_m5s[where_template],
            "science_m5": data_slice[self.m5_col][where_template],
            "template_m5": template_m5 * np.ones(np.sum(where_template)),
            "science_seeing": data_slice[self.seeing_col][where_template],
            "template_seeing": template_seeing * np.ones(np.sum(where_template)),
            "n_alerts": n_alerts_per_diffim[where_template],
        }

        return result

    def reduce_Night_template_created(self, metric_val):  # returns night of template creation
        return metric_val["Night_template_created"]

    def reduce_N_nights_without_template(
        self, metric_val
    ):  # returns number of nights needed to complete template
        return metric_val["N_nights_without_template"]

    def reduce_N_images_until_template(
        self, metric_val
    ):  # returns number of images needed to complete template
        return metric_val["N_images_until_template"]

    def reduce_N_images_with_template(self, metric_val):  # returns number of images with a template
        return metric_val["N_images_with_template"]

    def reduce_Template_m5(self, metric_val):  # calculated coadded m5 of resulting template
        return metric_val["Template_m5"]

    def reduce_Template_seeing(self, metric_val):  # calculated seeing of resulting template
        return metric_val["Template_seeing"]

    def reduce_fraction_better_template_seeing(self, metric_val):  # calculated seeing of resulting template
        return metric_val["Fraction_better_template_seeing"]

    def reduce_Total_alerts(self, metric_val):
        return metric_val["Total_alerts"]
