__all__ = (
    "power_law_dndh",
    "neo_dndh_granvik",
    "neo_dndh_grav",
    "pha_dndh_granvik",
    "pha_dndh_grav",
    "integrate_over_h",
    "sum_over_h",
    "TotalNumberSSO",
    "ValueAtHMetric",
    "MeanValueAtHMetric",
    "MoCompletenessMetric",
    "MoCompletenessAtTimeMetric",
)

import warnings

import numpy as np

from .mo_metrics import BaseMoMetric


def power_law_dndh(hvalues, hindex=0.33, no=None, ho=None, **kwargs):
    """Power law distribution of objects.

    Parameters
    ----------
    hvalues : `np.ndarray`, (N,)
        The H values corresponding to each metric_value
        (must be the same length).
        The hvalues are expected to be evenly spaced.
    hindex : `float`, optional
        The power-law index expected for the H value distribution.
        Default is 0.33  (dN/dH = 10^(hindex * H) ).
    no : `float`, optional
    ho: `float`, optional
        If no and ho are specified, this provides an anchor
        for the power law distribution,so that the expected number no
        of objects at ho is returned.
        Does not need to be set if just doing comparative weighting.

    Returns
    -------
    dndh : `np.ndarray`, (N,)
    """
    if no is None or ho is None:
        ho = hvalues.min()
        no = 10
    binratio = (np.diff(hvalues, append=hvalues[-1] + np.diff(hvalues)[-1])) / 0.1
    dndh = (no * 0.1) * np.power(10.0, hindex * (hvalues - ho)) * binratio
    return dndh


def neo_dndh_granvik(hvalues, **kwargs):
    binratio = (np.diff(hvalues, append=hvalues[-1] + np.diff(hvalues)[-1])) / 0.1
    y0 = 0  # 10 * np.power(10, 0.55 * (x - 17))
    y1 = 150 * np.power(10, 0.3 * (hvalues - 18.5))
    y2 = 2500 * np.power(10, 0.92 * (hvalues - 23.2))
    dndh = (y0 + y1 + y2) * binratio
    return dndh


def neo_dndh_grav(hvalues, **kwargs):
    binratio = (np.diff(hvalues, append=hvalues[-1] + np.diff(hvalues)[-1])) / 0.1
    y1 = 110 * np.power(10, 0.35 * (hvalues - 18.5))
    dndh = y1 * binratio
    return dndh


def pha_dndh_granvik(hvalues, **kwargs):
    binratio = (np.diff(hvalues, append=hvalues[-1] + np.diff(hvalues)[-1])) / 0.1
    y0 = 0  # 10 * np.power(10, 0.55 * (x - 17))
    y1 = 20 * np.power(10, 0.3 * (hvalues - 18.5))
    y2 = 500 * np.power(10, 0.92 * (hvalues - 23.2))
    dndh = (y0 + y1 + y2) * binratio
    return dndh


def pha_dndh_grav(hvalues, **kwargs):
    binratio = (np.diff(hvalues, append=hvalues[-1] + np.diff[hvalues][-1])) / 0.1
    y1 = 23.5 * np.power(10, 0.35 * (hvalues - 18.5))
    dndh = y1 * binratio
    return dndh


def integrate_over_h(metric_values, hvalues, dndh_func=power_law_dndh, **kwargs):
    """Calculate a metric value integrated over an h_range.
    This is the metric value *weighted* by the size distribution.

    Parameters
    ----------
    metric_values : `numpy.ndarray`
        The metric values at each H value.
    hvalues : `numpy.ndarray`
        The H values corresponding to each metric_value
        (must be the same length).
    dndh_func : function, optional
        One of the dN/dH functions defined below.
    **kwargs : `dict`, optional
        Keyword arguments to pass to dndh_func

    Returns
    --------
    int_vals : `np.ndarray`, (N,)
       The integrated metric values.
    """
    # Set expected H distribution.
    # dndh = differential size distribution (number in this bin)
    dndh = dndh_func(hvalues, **kwargs)
    # calculate the metric values *weighted* by the number of objects
    # in this bin and brighter
    int_vals = np.cumsum(metric_values * dndh) / np.cumsum(dndh)
    return int_vals


def sum_over_h(metric_values, hvalues, dndh_func=power_law_dndh, **kwargs):
    """Calculate the sum of the metric value multiplied by the number of
    objects at each H value. This is equivalent to calculating the
    number of objects meeting X requirement in the differential completeness
    or fraction of objects with lightcurves, etc.

    Parameters
    ----------
    metric_values : `np.ndarray`, (N,)
        The metric values at each H value.
    hvalues : `np.ndarray`, (N,)
        The H values corresponding to each metric_value.
    dndh_func : function, optional
        One of the dN/dH functions defined below.
    **kwargs : `dict`, optional
        Keyword arguments to pass to dndh_func

    Returns
    --------
    sum_vals : `np.ndarray`, (N,)
       The cumulative metric values.
    """
    # Set expected H distribution.
    # dndh = differential size distribution (number in this bin)
    dndh = dndh_func(hvalues, **kwargs)
    # calculate the metric values *weighted* by the number of objects
    # in this bin and brighter
    sum_vals = np.cumsum(metric_values * dndh)
    return sum_vals


class TotalNumberSSO(BaseMoMetric):
    """Calculate the total number of objects of a given population
    expected at a given H value or larger.

    Operations on differential completeness values
    (or equivalent; fractions of the population is ok if
    still a differential metric result, not cumulative).

    Parameters
    ----------
    h_mark : `float`, optional
        The H value at which to calculate the expected total number of objects.
    dndh_func : function, optional
        The dN/dH distribution used calculate the expected population size.

    Returns
    -------
    nObj : `float`
        The predicted number of objects in the population.
    """

    def __init__(self, h_mark=22, dndh_func=neo_dndh_granvik, **kwargs):
        self.h_mark = h_mark
        self.dndh_func = dndh_func
        metric_name = "Nobj <= %.1f" % (h_mark)
        self.kwargs = kwargs
        super().__init__(metric_name=metric_name, **kwargs)

    def run(self, metric_vals, h_vals):
        totals = sum_over_h(metric_vals, h_vals, self.dndh_func, **self.kwargs)
        n_obj = np.interp(self.h_mark, h_vals, totals)
        return n_obj


class ValueAtHMetric(BaseMoMetric):
    """Return the metric value at a given H value.

    Requires the metric values to be one-dimensional
    (typically, completeness values).

    Parameters
    ----------
    h_mark : `float`, optional
        The H value at which to look up the metric value.

    Returns
    -------
    value: : `float`
    """

    def __init__(self, h_mark=22, **kwargs):
        metric_name = "Value At H=%.1f" % (h_mark)
        self.units = "<= %.1f" % (h_mark)
        super().__init__(metric_name=metric_name, **kwargs)
        self.h_mark = h_mark

    def run(self, metric_vals, h_vals):
        # Check if desired H value is within range of H values.
        if (self.h_mark < h_vals.min()) or (self.h_mark > h_vals.max()):
            warnings.warn("Desired H value of metric outside range of provided H values.")
            return None
        if metric_vals.shape[0] != 1:
            warnings.warn("This is not an appropriate summary statistic for this data - need 1d values.")
            return None
        value = np.interp(self.h_mark, h_vals, metric_vals[0])
        return value


class MeanValueAtHMetric(BaseMoMetric):
    """Return the mean value of a metric at a given H.

    Allows the metric values to be multi-dimensional
    (i.e. use a cloned H distribution).

    Parameters
    ----------
    h_mark : `float`, optional
        The H value at which to look up the metric value.

    Returns
    -------
    value: : `float`
    """

    def __init__(self, h_mark=22, reduce_func=np.mean, metric_name=None, **kwargs):
        if metric_name is None:
            metric_name = "Mean Value At H=%.1f" % (h_mark)
        self.units = "@ H= %.1f" % (h_mark)
        super().__init__(metric_name=metric_name, **kwargs)
        self.h_mark = h_mark
        self.reduce_func = reduce_func

    def run(self, metric_vals, h_vals):
        # Check if desired H value is within range of H values.
        if (self.h_mark < h_vals.min()) or (self.h_mark > h_vals.max()):
            warnings.warn("Desired H value of metric outside range of provided H values.")
            return None
        value = np.interp(self.h_mark, h_vals, self.reduce_func(metric_vals.swapaxes(0, 1), axis=1))
        return value


class MoCompletenessMetric(BaseMoMetric):
    """Calculate the fraction of the population that meets `threshold` value
    or higher. This is equivalent to calculating the completeness
    (relative to the entire population) given the output of a
    Discovery_N_Chances metric, or the fraction of the population that meets
    a given cutoff value for Color determination metrics.

    Any moving object metric that outputs a float value can thus have
    the 'fraction of the population' with greater than X value calculated here,
    as a summary statistic.

    Parameters
    ----------
    threshold : `int`, optional
        Count the fraction of the population that exceeds this value.
    nbins : `int`, optional
        If the H values for the metric are not a cloned distribution,
        then split up H into this many bins.
    min_hrange : `float`, optional
        If the H values for the metric are not a cloned distribution,
        then split up H into at least this
        range (otherwise just use the min/max of the H values).
    cumulative : `bool`, optional
        If False, simply report the differential fractional value
        (or differential completeness).
        If True, integrate over the H distribution (using IntegrateOverH)
        to report a cumulative fraction.
        Default of None will use True, unless metric_name is set and starts
        with "Differential" - then default will use False.
    hindex : `float`, optional
        Use hindex as the power law to integrate over H,
        if cumulative is True.
    """

    def __init__(
        self,
        threshold=1,
        nbins=20,
        min_hrange=1.0,
        cumulative=None,
        hindex=0.33,
        **kwargs,
    ):
        if cumulative is None:
            # if metric_name does not start with 'differential',
            # then cumulative->True
            if "metric_name" not in kwargs:
                self.cumulative = True
                metric_name = "CumulativeCompleteness"
            else:
                #  'metric_name' in kwargs:
                metric_name = kwargs.pop("metric_name")
                if metric_name.lower().startswith("differential"):
                    self.cumulative = False
                else:
                    self.cumulative = True
        else:
            # cumulative was set
            self.cumulative = cumulative
            if "metric_name" in kwargs:
                metric_name = kwargs.pop("metric_name")
                if metric_name.lower().startswith("differential") and self.cumulative:
                    warnings.warn(f"Completeness metric_name is {metric_name} but cumulative is True")
            else:
                if self.cumulative:
                    metric_name = "CumulativeCompleteness"
                else:
                    metric_name = "DifferentialCompleteness"
        if self.cumulative:
            units = "<=H"
        else:
            units = "@H"
        super().__init__(metric_name=metric_name, units=units, **kwargs)
        self.threshold = threshold
        # If H is not a cloned distribution,
        # then we need to specify how to bin these values.
        self.nbins = nbins
        self.min_hrange = min_hrange
        self.hindex = hindex

    def run(self, metric_values, h_vals):
        n_ssos = metric_values.shape[0]
        n_hval = len(h_vals)
        metric_val_h = metric_values.swapaxes(0, 1)
        if n_hval == metric_values.shape[1]:
            # h_vals array is probably the same as the cloned H array.
            completeness = np.zeros(len(h_vals), float)
            for i, H in enumerate(h_vals):
                completeness[i] = np.where(metric_val_h[i].filled(0) >= self.threshold)[0].size
            completeness = completeness / float(n_ssos)
        else:
            # The h_vals are spread more randomly among the objects
            # (we probably used one per object).
            hrange = h_vals.max() - h_vals.min()
            min_h = h_vals.min()
            if hrange < self.min_hrange:
                hrange = self.min_hrange
                min_h = h_vals.min() - hrange / 2.0
            stepsize = hrange / float(self.nbins)
            bins = np.arange(min_h, min_h + hrange + stepsize / 2.0, stepsize)
            h_vals = bins[:-1]
            n_all, b = np.histogram(metric_val_h[0], bins)
            condition = np.where(metric_val_h[0] >= self.threshold)[0]
            n_found, b = np.histogram(metric_val_h[0][condition], bins)
            completeness = n_found.astype(float) / n_all.astype(float)
            completeness = np.where(n_all == 0, 0, completeness)
        if self.cumulative:
            completeness_int = integrate_over_h(completeness, h_vals, power_law_dndh, Hindex=self.hindex)
            summary_val = np.empty(len(completeness_int), dtype=[("name", np.str_, 20), ("value", float)])
            summary_val["value"] = completeness_int
            for i, Hval in enumerate(h_vals):
                summary_val["name"][i] = "H <= %f" % (Hval)
        else:
            summary_val = np.empty(len(completeness), dtype=[("name", np.str_, 20), ("value", float)])
            summary_val["value"] = completeness
            for i, Hval in enumerate(h_vals):
                summary_val["name"][i] = "H = %f" % (Hval)
        return summary_val


class MoCompletenessAtTimeMetric(BaseMoMetric):
    """Calculate the completeness (relative to the entire population)
    <= a given H as a function of time, given the times of each discovery.

    Input values of the discovery times can come from the Discovery_Time
    (child) metric or the KnownObjects metric.

    Parameters
    ----------
    times : `np.ndarray`, (N,) or `list` [`float`]
        The bins to distribute the discovery times into.
        Same units as the discovery time (typically MJD).
    hval : `float`, optional
        The value of H to count completeness at, or cumulative completeness to.
        Default None, in which case a value halfway through h_vals
        (the slicer H range) will be chosen.
    cumulative : `bool`, optional
        If True, calculate the cumulative completeness (completeness <= H).
        If False, calculate the differential completeness (completeness @ H).
        Default None which becomes 'True',
        unless metric_name starts with 'differential'.
    hindex : `float`, optional
        Use hindex as the power law to integrate over H,
        if cumulative is True.
    """

    def __init__(self, times, hval=None, cumulative=None, hindex=0.33, **kwargs):
        self.hval = hval
        self.times = times
        self.hindex = hindex
        if cumulative is None:
            # if metric_name does not start with 'differential',
            # then cumulative->True
            if "metric_name" not in kwargs:
                self.cumulative = True
                metric_name = "CumulativeCompleteness@Time@H=%.2f" % self.hval
            else:
                #  'metric_name' in kwargs:
                metric_name = kwargs.pop("metric_name")
                if metric_name.lower().startswith("differential"):
                    self.cumulative = False
                else:
                    self.cumulative = True
        else:
            # cumulative was set
            self.cumulative = cumulative
            if "metric_name" in kwargs:
                metric_name = kwargs.pop("metric_name")
                if metric_name.lower().startswith("differential") and self.cumulative:
                    warnings.warn(f"Completeness metric_name is {metric_name} but cumulative is True")
            else:
                if self.cumulative:
                    metric_name = "CumulativeCompleteness@Time@H=%.2f" % self.hval
                else:
                    metric_name = "DifferentialCompleteness@Time@H=%.2f" % self.hval
        self._set_labels()
        super().__init__(metric_name=metric_name, units=self.units, **kwargs)

    def _set_labels(self):
        if self.hval is not None:
            if self.cumulative:
                self.units = "H <=%.1f" % (self.hval)
            else:
                self.units = "H = %.1f" % (self.hval)
        else:
            self.units = "H"

    def run(self, discovery_times, h_vals):
        if len(h_vals) != discovery_times.shape[1]:
            warnings.warn("This summary metric expects cloned H distribution. Cannot calculate summary.")
            return
        n_ssos = discovery_times.shape[0]
        timesin_h = discovery_times.swapaxes(0, 1)
        completeness_h = np.empty([len(h_vals), len(self.times)], float)
        for i, H in enumerate(h_vals):
            n, b = np.histogram(timesin_h[i].compressed(), bins=self.times)
            completeness_h[i][0] = 0
            completeness_h[i][1:] = n.cumsum()
        completeness_h = completeness_h / float(n_ssos)
        completeness = completeness_h.swapaxes(0, 1)
        if self.cumulative:
            for i, t in enumerate(self.times):
                completeness[i] = integrate_over_h(completeness[i], h_vals)
        # To save the summary statistic, we must pick out a given H value.
        if self.hval is None:
            hidx = len(h_vals) // 2
            self.hval = h_vals[hidx]
            self._set_labels()
        else:
            hidx = np.where(np.abs(h_vals - self.hval) == np.abs(h_vals - self.hval).min())[0][0]
            self.hval = h_vals[hidx]
            self._set_labels()
        summary_val = np.empty(len(self.times), dtype=[("name", np.str_, 20), ("value", float)])
        summary_val["value"] = completeness[:, hidx]
        for i, time in enumerate(self.times):
            summary_val["name"][i] = "%s @ %.2f" % (self.units, time)
        return summary_val
