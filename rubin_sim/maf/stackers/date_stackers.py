__all__ = (
    "ObservationStartDatetime64Stacker",
    "ObservationStartTimestampStacker",
    "DayObsStacker",
    "DayObsMJDStacker",
    "DayObsISOStacker",
)

import numpy as np
import pandas as pd
from astropy.time import Time

from .base_stacker import BaseStacker


class ObservationStartDatetime64Stacker(BaseStacker):
    """Add the observation start time as a numpy.datetime64."""

    cols_added = ["observationStartDatetime64"]

    def __init__(
        self,
        mjd_col="observationStartMJD",
    ):
        self.mjd_col = mjd_col
        self.cols_req = [self.mjd_col]
        self.units = [None]
        self.cols_added_dtypes = ["datetime64[ns]"]

    def _run(self, sim_data, cols_present=False):
        if cols_present:
            # Column already present in data; assume it is correct and does not
            # need recalculating.
            return sim_data

        sim_data["observationStartDatetime64"] = Time(sim_data[self.mjd_col], format="mjd").datetime64

        return sim_data


class ObservationStartTimestampStacker(BaseStacker):
    """Add the observation start time as a pandas.Timestamp."""

    cols_added = ["start_timestamp"]

    def __init__(
        self,
        mjd_col="observationStartMJD",
    ):
        self.mjd_col = mjd_col
        self.cols_req = [self.mjd_col]
        self.units = [None]
        self.cols_added_dtypes = ["O"]

    def run(self, sim_data, override=False):
        # Override the run from the base class, not _run,
        # because the implementation
        # of _add_stackers_cols in run in the base closs
        # clobbers the type for the new column,
        # which we really need.

        visits = sim_data if isinstance(sim_data, pd.DataFrame) else pd.DataFrame(sim_data)

        if "start_timestamp" in visits.columns and not override:
            return sim_data

        if len(visits[self.mjd_col]) > 0:
            visits["start_timestamp"] = pd.to_datetime(
                visits[self.mjd_col] + 2400000.5, origin="julian", unit="D", utc=True
            )
        else:
            # If we are passed an empty series, be sure to return add an empty
            # series of the correct type back.
            # This is handy if the result is being passed to bokeh
            # for plotting.
            visits["start_timestamp"] = pd.to_datetime(2460000.5, origin="julian", unit="D", utc=True)

        match sim_data:
            case pd.DataFrame():
                return visits
            case dict():
                return visits.to_dict()
            case _:
                return visits.to_records(index=False)


def _compute_day_obs_mjd(mjd):
    day_obs_mjd = np.floor(mjd - 0.5).astype("int")
    return day_obs_mjd


def _compute_day_obs_astropy_time(mjd):
    day_obs_time = Time(_compute_day_obs_mjd(mjd), format="mjd")
    return day_obs_time


def _compute_day_obs_iso8601(mjd):
    iso_times = _compute_day_obs_astropy_time(mjd).iso

    # Work both for mjd as a scalar and a numpy array
    if isinstance(iso_times, str):
        day_obs_iso = iso_times[:10]
    else:
        day_obs_iso = np.array([d[:10] for d in iso_times])

    return day_obs_iso


def _compute_day_obs_int(mjd):
    day_obs_int = np.array([d.replace("-", "") for d in _compute_day_obs_iso8601(mjd)])

    return day_obs_int


class DayObsStacker(BaseStacker):
    """Add dayObs as, as defined by SITCOMTN-32.

    Parameters
    ----------
    mjd_col : `str`
        The column with the observatin start MJD.
    """

    cols_added = ["dayObs"]

    def __init__(self, mjd_col="observationStartMJD"):
        self.mjd_col = mjd_col
        self.cols_req = [self.mjd_col]
        self.units = ["days"]
        self.cols_added_dtypes = [int]

    def _run(self, sim_data, cols_present=False):
        if cols_present:
            # Column already present in data; assume it is correct and does not
            # need recalculating.
            return sim_data

        sim_data[self.cols_added[0]] = _compute_day_obs_int(sim_data[self.mjd_col])
        return sim_data


class DayObsMJDStacker(BaseStacker):
    """Add dayObs defined by SITCOMTN-32, as an MJD.

    Parameters
    ----------
    mjd_col : `str`
        The column with the observatin start MJD.
    """

    cols_added = ["day_obs_mjd"]

    def __init__(self, mjd_col="observationStartMJD"):
        self.mjd_col = mjd_col
        self.cols_req = [self.mjd_col]
        self.units = ["days"]
        self.cols_added_dtypes = [int]

    def _run(self, sim_data, cols_present=False):
        if cols_present:
            # Column already present in data; assume it is correct and does not
            # need recalculating.
            return sim_data

        sim_data[self.cols_added[0]] = _compute_day_obs_mjd(sim_data[self.mjd_col])
        return sim_data


class DayObsISOStacker(BaseStacker):
    """Add dayObs as defined by SITCOMTN-32, in ISO 8601 format.

    Parameters
    ----------
    mjd_col : `str`
        The column with the observatin start MJD."""

    cols_added = ["day_obs_iso8601"]

    def __init__(self, mjd_col="observationStartMJD"):
        self.mjd_col = mjd_col
        self.cols_req = [self.mjd_col]
        self.units = [None]
        self.cols_added_dtypes = [(str, 10)]

    def _run(self, sim_data, cols_present=False):
        if cols_present:
            # Column already present in data; assume it is correct and does not
            # need recalculating.
            return sim_data

        sim_data[self.cols_added[0]] = _compute_day_obs_iso8601(sim_data[self.mjd_col])
        return sim_data
