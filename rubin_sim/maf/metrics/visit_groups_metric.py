__all__ = ("VisitGroupsMetric", "PairFractionMetric")


import numpy as np

from .base_metric import BaseMetric

# Example of more complex metric
# Takes multiple columns of data (although 'night' could be calculable
# from 'expmjd')
# Returns variable length array of data
# Uses multiple reduce functions


class PairFractionMetric(BaseMetric):
    """What fraction of observations are part of a pair.

    Note, an observation can be a member of more than one "pair". For example,
    t=[0, 5, 30], all observations would be considered part of a pair because
    they all have an observation within the given window to pair with (the
    observation at t=30 pairs twice).

    Parameters
    ----------
    min_gap : float, optional
        Minimum time to consider something part of a pair (minutes).
        Default 15.
    max_gap : float, optional
        Maximum time to consider something part of a pair (minutes).
        Default 90.
    """

    def __init__(
        self, mjd_col="observationStartMJD", metric_name="PairFraction", min_gap=15.0, max_gap=90.0, **kwargs
    ):
        self.mjd_col = mjd_col
        self.min_gap = min_gap / 60.0 / 24.0
        self.max_gap = max_gap / 60.0 / 24.0
        units = ""
        super(PairFractionMetric, self).__init__(
            col=[mjd_col], metric_name=metric_name, units=units, **kwargs
        )

    def run(self, data_slice, slice_point=None):
        nobs = np.size(data_slice[self.mjd_col])
        times = np.sort(data_slice[self.mjd_col])

        # Check which ones have a forard match
        t_plus = times + self.max_gap
        t_minus = times + self.min_gap
        ind1 = np.searchsorted(times, t_plus)
        ind2 = np.searchsorted(times, t_minus)
        # If ind1 and ind2 are the same, there is no pairable image for
        # that exposure
        diff1 = ind1 - ind2

        # Check which have a back match
        t_plus = times - self.max_gap
        t_minus = times - self.min_gap
        ind1 = np.searchsorted(times, t_plus)
        ind2 = np.searchsorted(times, t_minus)

        diff2 = ind1 - ind2

        # The exposure has a pair ahead or behind
        is_paired = np.where((diff1 != 0) | (diff2 != 0))[0]
        result = np.size(is_paired) / float(nobs)
        return result


class VisitGroupsMetric(BaseMetric):
    """Count the number of visits per night within delta_t_min and delta_t_max.

    Parameters
    ----------
    time_col : str, optional
        Column with the time of the visit.
        Default: 'observationStartMJD'
    nights_col : str, optional
        Column with the night of the visit
        Default: 'night'
    delta_t_min : float, min
        Minimum time of window: units are days
        Default: 15.0 / 60.0 / 24.0 (15min in days)
    delta_t_max : float, optional
        Maximum time of window: units are days
        Default: 90.0 / 60.0 / 24.0 (90min in days)
    min_n_visits : int, optional
        Minimum number of visits within a night (with spacing between
        delta_t_min/max from any other visit) required
        Default: 2
    window : int, optional
        Number of nights to consider within a window (for reduce methods)
        Default: 30
    min_n_nights : int, optional
        minimum required number of nights within window to make a full 'group'
        Default: 3

    """

    def __init__(
        self,
        time_col="observationStartMJD",
        nights_col="night",
        metric_name="VisitGroups",
        delta_t_min=15.0 / 60.0 / 24.0,
        delta_t_max=90.0 / 60.0 / 24.0,
        min_n_visits=2,
        window=30,
        min_n_nights=3,
        **kwargs,
    ):
        self.times = time_col
        self.nights = nights_col
        eps = 1e-10
        self.delta_tmin = float(delta_t_min) - eps
        self.delta_tmax = float(delta_t_max)
        self.min_n_visits = int(min_n_visits)
        self.window = int(window)
        self.min_n_nights = int(min_n_nights)
        super(VisitGroupsMetric, self).__init__(
            col=[self.times, self.nights], metric_name=metric_name, **kwargs
        )
        self.reduce_order = {
            "Median": 0,
            "NNightsWithNVisits": 1,
            "NVisitsInWindow": 2,
            "NNightsInWindow": 3,
            "NLunations": 4,
            "MaxSeqLunations": 5,
        }
        self.comment = "Evaluation of the number of visits within a night, with separations between "
        self.comment += "tMin %.1f and tMax %.1f minutes." % (
            self.delta_tmin * 24.0 * 60.0,
            self.delta_tmax * 24.0 * 60.0,
        )
        self.comment += "Groups of visits use a minimum number of visits per night of %d, " % (
            self.min_n_visits
        )
        self.comment += "and minimum number of nights of %d." % (self.min_n_nights)
        self.comment += "Two visits within this interval would count as 2. "
        self.comment += "Visits closer than tMin, paired with visits that do fall within tMin/tMax, "
        self.comment += "count as half visits. <br>"
        self.comment += "VisitsGroups_Median calculates the median number of visits between tMin/tMax for "
        self.comment += "all nights. <br>"
        self.comment += "VisitGroups_NNightsWithNNights calculates the number of nights that have at "
        self.comment += "least %d visits. <br>" % (self.min_n_visits)
        self.comment += "VisitGroups_NVisitsInWindow calculates the max number of visits within a window of "
        self.comment += "%d days. <br>" % (self.window)
        self.comment += "VisitGroups_NNightsInWindow calculates the max number of nights that have more "
        self.comment += "than %d visits within %d days. <br>" % (
            self.min_n_visits,
            self.window,
        )
        self.comment += "VisitGroups_NLunations calculates the number of lunations (30 days) that have "
        self.comment += "at least one group of more than %d nights with more than %d visits, within " % (
            self.min_n_nights,
            self.min_n_visits,
        )
        self.comment += "%d days. <br>" % (self.window)
        self.comment += "VisitGroups_MaxSeqLunations calculates the maximum sequential lunations that have "
        self.comment += 'at least one "group". <br>'

    def run(self, data_slice, slice_point=None):
        """
        Return a dictionary of:
        the number of visits within a night (within delta tmin/tmax of another
        visit), and the nights with visits > minNVisits.
        Count two visits which are within tmin of each other, but which have
        another visit within tmin/tmax interval, as one and a half (instead of
        two).

        So for example: 4 visits, where 1, 2, 3 were all within deltaTMax of
        each other, and 4 was later but within deltaTmax of visit 3 -- would
        give you 4 visits. If visit 1 and 2 were closer together than
        deltaTmin, the two would be counted as 1.5 visits together (if only 1
        and 2 existed, then there would be 0 visits as none would be within the
        qualifying time interval).
        """
        uniquenights = np.unique(data_slice[self.nights])
        nights = []
        visit_num = []
        # Find the nights with visits within deltaTmin/max of one another and
        # count the number of visits
        for n in uniquenights:
            condition = data_slice[self.nights] == n
            times = np.sort(data_slice[self.times][condition])
            nvisits = 0
            ntooclose = 0
            # Calculate difference between each visit and time of previous
            # visit (tnext- tnow)
            timediff = np.diff(times)
            timegood = np.where(
                (timediff <= self.delta_tmax) & (timediff >= self.delta_tmin),
                True,
                False,
            )
            timetooclose = np.where(timediff < self.delta_tmin, True, False)
            if len(timegood) > 1:
                # Count visits for all but last index in timediff
                for tg1, ttc1, tg2, ttc2 in zip(
                    timegood[:-1], timetooclose[:-1], timegood[1:], timetooclose[1:]
                ):
                    if tg1:
                        nvisits += 1
                        if not tg2:
                            nvisits += 1
                    if ttc1:
                        ntooclose += 1
                        if not tg2 and not ttc2:
                            ntooclose += 1
                # Take care of last timediff
                if timegood[-1]:
                    nvisits += 2  # close out visit sequence
                if timetooclose[-1]:
                    ntooclose += 1
                    if not timegood[-2] and not timetooclose[-2]:
                        ntooclose += 1
            else:
                if timegood.size > 0:
                    nvisits += 2
            # Count up all visits for night.
            if nvisits > 0:
                nvisits = nvisits + ntooclose / 2.0
                visit_num.append(nvisits)
                nights.append(n)
        # Convert to numpy arrays.
        visit_num = np.array(visit_num)
        nights = np.array(nights)
        metricval = {"visits": visit_num, "nights": nights}
        if len(visit_num) == 0:
            return self.badval
        return metricval

    def reduce_median(self, metricval):
        """Reduce to median number of visits per night."""
        return np.median(metricval["visits"])

    def reduce_n_nights_with_n_visits(self, metricval):
        """Reduce to total number of nights with more than 'minNVisits'
        visits."""
        condition = metricval["visits"] >= self.min_n_visits
        return len(metricval["visits"][condition])

    def _in_window(self, visits, nights, night, window, min_n_visits):
        condition = (nights >= night) & (nights < night + window)
        condition2 = visits[condition] >= min_n_visits
        return visits[condition][condition2], nights[condition][condition2]

    def reduce_n_visits_in_window(self, metricval):
        """Reduce to max number of total visits on all nights with more than
        minNVisits, within any 'window' (default=30 nights)."""
        maxnvisits = 0
        for n in metricval["nights"]:
            vw, nw = self._in_window(
                metricval["visits"],
                metricval["nights"],
                n,
                self.window,
                self.min_n_visits,
            )
            maxnvisits = max((vw.sum(), maxnvisits))
        return maxnvisits

    def reduce_n_nights_in_window(self, metricval):
        """Reduce to max number of nights with more than minNVisits, within
        'window' over all windows."""
        maxnights = 0
        for n in metricval["nights"]:
            vw, nw = self._in_window(
                metricval["visits"],
                metricval["nights"],
                n,
                self.window,
                self.min_n_visits,
            )
            maxnights = max(len(nw), maxnights)
        return maxnights

    def _in_lunation(self, visits, nights, lunation_start, lunation_length):
        condition = (nights >= lunation_start) & (nights < lunation_start + lunation_length)
        return visits[condition], nights[condition]

    def reduce_n_lunations(self, metricval):
        """Reduce to number of lunations (unique 30 day windows) that contain
        at least one 'group': a set of more than minNVisits per night, with
        more than minNNights of visits within 'window' time period.
        """
        lunation_length = 30
        lunations = np.arange(
            metricval["nights"][0],
            metricval["nights"][-1] + lunation_length / 2.0,
            lunation_length,
        )
        n_lunations = 0
        for lunation in lunations:
            # Find visits within lunation.
            vl, nl = self._in_lunation(metricval["visits"], metricval["nights"], lunation, lunation_length)
            for n in nl:
                # Find visits which are in groups within the lunation.
                vw, nw = self._in_window(vl, nl, n, self.window, self.min_n_visits)
                if len(nw) >= self.min_n_nights:
                    n_lunations += 1
                    break
        return n_lunations

    def reduce_max_seq_lunations(self, metricval):
        """Count the max number of sequential lunations (unique 30 day windows
        that contain at least one 'group': a set of more than minNVisits per
        night, with more than minNNights of visits within 'window' time period.
        """
        lunation_length = 30
        lunations = np.arange(
            metricval["nights"][0],
            metricval["nights"][-1] + lunation_length / 2.0,
            lunation_length,
        )
        max_sequence = 0
        cur_sequence = 0
        for lunation in lunations:
            # Find visits within lunation.
            vl, nl = self._in_lunation(metricval["visits"], metricval["nights"], lunation, lunation_length)
            # If no visits this lunation:
            if len(vl) == 0:
                max_sequence = max(max_sequence, cur_sequence)
                cur_sequence = 0
            # Else, look to see if groups can be made from the visits.
            for n in nl:
                # Find visits which are in groups within the lunation.
                vw, nw = self._in_window(vl, nl, n, self.window, self.min_n_visits)
                # If there was a group within this lunation:
                if len(nw) >= self.min_n_nights:
                    cur_sequence += 1
                    break
                # Otherwise we're not in a sequence (anymore, or still).
                else:
                    max_sequence = max(max_sequence, cur_sequence)
                    cur_sequence = 0
        # Pick up last sequence if were in a sequence at last lunation.
        max_sequence = max(max_sequence, cur_sequence)
        return max_sequence
