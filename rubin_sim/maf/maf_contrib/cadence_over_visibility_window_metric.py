from sys import argv

# from astropy.visualization import astropy_mpl_style
# plt.style.use(astropy_mpl_style)
import numpy as np
from astropy.time import Time, TimeDelta

import rubin_sim.maf.db as db
import rubin_sim.maf.metricBundles as metricBundles
import rubin_sim.maf.slicers as slicers
from rubin_sim.maf.metrics import BaseMetric

from .calc_expected_visits import CalcExpectedVisitsMetric


class CadenceOverVisibilityWindowMetric(BaseMetric):
    """Metric to compare the lightcurve cadence produced by LSST over the visibility window
    for a given position in the sky to the desired cadence.

    This metric determines the number of
    visits to a given field (RA,Dec) performed, including all exposures taken
    with the given set of filters.

    It compares the actual number of visits with the maximum possible visits,
    calculated from the visibility window of the field for the given start and
    end dates, and desired cadence.

    The returned result = ([sum_j (n_visits_actual / n_visits_desired)]/N_filters ) * 100 (%)

    For cadences less than 1 day, this is the sum over all anticipated visits
    per night.  For cadences greater than 1 day, this is calculated as a fraction
    of the anticipated number of visits during batches of nights.
    """

    def __init__(
        self,
        filters,
        cadence,
        start_date,
        end_date,
        metric_name="CadenceOverVisibilityWindowMetric",
        ra_col="fieldRA",
        dec_col="fieldDec",
        exp_col="visitExposureTime",
        n_exp_col="numExposures",
        filter_col="filter",
        obstime_col="observationStartMJD",
        visittime_col="visitTime",
        verbose=False,
        **kwargs,
    ):
        """Arguments:
        filters  list Filterset over which to compute the metric
        cadence  list Cadence desired for each filter in units of decimal hours
                e.g. [ 0.5, 1.0, 1.2 ]
        start_date string Start of observing window YYYY-MM-DD
        end_date string End of observing window YYYY-MM-DD
        """

        self.filters = filters
        self.cadence = cadence
        self.start_date = start_date
        self.end_date = end_date
        self.ra_col = ra_col
        self.dec_col = dec_col
        self.exp_col = exp_col
        self.n_exp_col = n_exp_col
        self.obstime_col = obstime_col
        self.visittime_col = visittime_col
        self.filter_col = filter_col
        self.verbose = verbose

        if len(self.filters) != len(self.cadence):
            raise ValueError(
                "ERROR: The list of filters requested must correspond to the list of required cadences"
            )
            exit()

        columns = [
            self.ra_col,
            self.dec_col,
            self.exp_col,
            self.n_exp_col,
            self.obstime_col,
            self.visittime_col,
            self.filter_col,
        ]

        super(CadenceOverVisibilityWindowMetric, self).__init__(col=columns, metric_name=metric_name)

    def run(self, data_slice, slice_point=None):
        t = np.empty(data_slice.size, dtype=list(zip(["time", "filter"], [float, "|S1"])))
        t["time"] = data_slice[self.obstime_col]

        t_start = Time(self.start_date + " 00:00:00")
        t_end = Time(self.end_date + " 00:00:00")
        n_days = int((t_end - t_start).value)
        dates = np.array([t_start + TimeDelta(i, format="jd", scale=None) for i in range(0, n_days, 1)])

        result = 0.0

        for i, f in enumerate(self.filters):
            if self.verbose:
                print(
                    "Calculating the expected visits in filter "
                    + f
                    + " given required cadence "
                    + str(self.cadence[i])
                )

            # Returns a list of the number of visits per night for each pointing
            pointing = [(data_slice[self.ra_col][0], data_slice[self.dec_col][0])]

            visit = CalcExpectedVisitsMetric(
                pointing,
                self.cadence[i],
                self.start_date,
                self.end_date,
                self.filters[i],
                self.ra_col,
                self.dec_col,
                verbose=self.verbose,
            )

            (n_visits_desired, hrs_visibility) = visit.run(data_slice)

            n_visits_actual = []

            for j, d in enumerate(dates):
                idx = np.where(data_slice[self.filter_col] == f)

                actual_visits_per_filter = data_slice[idx]

                tdx = np.where(
                    actual_visits_per_filter[self.obstime_col].astype(int) == int(d.jd - 2400000.5)
                )

                n_visits_actual.append(float(len(actual_visits_per_filter[tdx])))

            # Case 1: Required cadence is less than 1 day, meaning we
            #         anticipate more than 1 observation per night
            if self.cadence[i] <= 24.0:
                for j, d in enumerate(dates):
                    if n_visits_desired[0][j] > 0:
                        night_efficiency = n_visits_actual[j] / float(n_visits_desired[0][j])

                        result += night_efficiency

                result = result / float(len(dates))

            # Case 2: Required cadence is greater than 1 day, meaning we
            #         expect at least 1 observation within batches of nights
            #         self.cadence[i] long
            else:
                n_nights = int(self.cadence[i] / 24.0)

                for j in range(0, len(dates), n_nights):
                    hrs_available = (np.array(hrs_visibility[0][j : j + n_nights])).sum()

                    n_actual = (np.array(n_visits_actual[j : j + n_nights])).sum()

                    if hrs_available >= 1.0 and n_actual > 1:
                        result += 1.0

                result = result / float(len(dates) / n_nights)

        result = (result / float(len(self.filters))) * 100.0

        if self.verbose:
            print("METRIC RESULT: Observing cadence percentage = " + str(result))

        return result


def compute_metric(params):
    """Function to execute the metric calculation when code is called from
    the commandline"""

    obsdb = db.OpsimDatabase("/home/docmaf/my_repoes/data/baseline2018a.db")
    output_dir = "/home/docmaf/"
    results_db = db.ResultsDb(out_dir=output_dir)

    (propids, proptags) = obsdb.fetchPropInfo()
    survey_where = obsdb.createSQLWhere(params["survey"], proptags)

    obs_params = {"verbose": params["verbose"]}

    metric = CadenceOverVisibilityWindowMetric(
        params["filters"], params["cadence"], params["start_date"], params["end_date"], **obs_params
    )

    slicer = slicers.HealpixSlicer(nside=64)
    sqlconstraint = survey_where
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint)

    bgroup = metricBundles.MetricBundleGroup(
        {0: bundle}, obsdb, outDir="newmetric_test", results_db=results_db
    )
    bgroup.run_all()


if __name__ == "__main__":
    if len(argv) == 1:
        print("Metric requires the following commandline sequence, e.g.:")
        print(
            "> python CadenceOverVisibilityWindowMetric.py filters=g,r,i,z cadence=168.0,168.0,1.0,168.0 start_date=2020-01-02 end_date=2020-04-02 survey=option"
        )
        print("  where:")
        print("  filters may be specified as a comma-separated list without spaces")
        print(
            "  cadence is the cadence corresponding to each filter in hours, in a comma-separated list without spaces"
        )
        print("  start_date, end_date are the UTC dates of the start and end of the observing window")
        print("  survey indicates which survey to select data from.  Options are {WFD, DD, NES}")

    else:
        params = {"verbose": False}

        for arg in argv:
            try:
                (key, value) = arg.split("=")

                if key == "filters":
                    params[key] = value.split(",")

                if key == "cadence":
                    cadence_list = []

                    for val in value.split(","):
                        cadence_list.append(float(val))

                    params[key] = cadence_list

                if key in ["start_date", "end_date", "survey"]:
                    params[key] = value

            except ValueError:
                pass

            if "verbose" in arg:
                params["verbose"] = True

        compute_metric(params)
