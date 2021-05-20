"""MAF slicers to slice into time intervals based on MJD

Primarily intended for hourglass plots.
"""

# pylint: disable=too-many-arguments

# imports
from functools import wraps
from collections import defaultdict

import numpy as np
import pandas as pd

from .baseSlicer import BaseSlicer

# constants

# exception classes


class SlicerNotSetup(Exception):
    """Thrown when a slicer is not setup for the method called."""


# interface functions

# classes


class TimeIntervalSlicer(BaseSlicer):
    """Base for all time interval slicers.

    Slices in constant time intervals.

    Parameters
    ----------
    interval_seconds : `int`
        Duration of slice time intervals, in seconds
    mjd_column_name : `str`
        Name of column on which to slice, must be in units of days
    badval : `float`
        Value to use for bad values in slice
    verbose : `bool`
        Print extra information?
    """

    def __init__(
        self,
        interval_seconds=90,
        mjd_column_name="observationStartMJD",
        badval=np.NaN,
        verbose=False,
    ):
        super().__init__(verbose=verbose, badval=badval)
        self.interval_seconds = interval_seconds
        self.mjd_column_name = mjd_column_name
        self.columnsNeeded = [mjd_column_name]
        self.simIdxs = defaultdict(list)  # pylint: disable=invalid-name

    def setupSlicer(self, simData, maps=None):
        visit_mjds = simData[self.mjd_column_name]
        start_mjd = np.floor(np.min(visit_mjds)).astype(int)
        end_mjd = np.ceil(np.max(visit_mjds)).astype(int)
        interval_days = self.interval_seconds / (24 * 60 * 60.0)

        mjd_bin_edges = np.arange(
            start_mjd, end_mjd + interval_days, interval_days
        )

        self.simIdxs.update(
            pd.DataFrame(
                {
                    "visit_idx": np.arange(len(visit_mjds)),
                    "sid": np.digitize(visit_mjds, mjd_bin_edges),
                }
            )
            .groupby("sid")
            .agg(list)
            .to_dict()["visit_idx"]
        )

        mjds = np.arange(start_mjd, end_mjd, interval_days)
        self.slicePoints["sid"] = np.arange(len(mjds))
        self.slicePoints["mjd"] = mjds
        self.slicePoints["duration"] = np.full_like(
            mjds, self.interval_seconds
        )
        self.nslice = len(mjds)
        self.shape = self.nslice
        self._runMaps(maps)

        @wraps(self._sliceSimData)
        def _sliceSimData(islice):  # pylint: disable=invalid-name
            idxs = self.simIdxs[islice]

            try:
                _ = idxs[0]
            except (TypeError, IndexError):
                idxs = [idxs]

            slice_points = {
                "mjd": self.slicePoints["mjd"][islice],
                "duration": self.slicePoints["duration"][islice],
            }

            return {"idxs": idxs, "slicePoint": slice_points}

        setattr(self, "_sliceSimData", _sliceSimData)

    def __eq__(self, otherSlicer):
        """Evaluate if slicers are equivalent."""
        if not isinstance(otherSlicer, self.__class__):
            return False

        for key in ["sid", "mjd", "duration"]:
            if not np.array_equal(
                otherSlicer.slicePoints[key], self.slicePoints[key]
            ):
                return False

        return True

    def _sliceSimData(self, *args, **kwargs):
        raise SlicerNotSetup()


class BlockIntervalSlicer(TimeIntervalSlicer):
    """Slices into intervals with common "note" values and no long gaps.

    Parameters
    ----------
    mjd_column_name : `str`
        Name of column on which to slice, must be in units of days
    duration_column_name : `str`
        Name of column with the duration of each visit (in seconds)
    note_column_name : `str`
        Name of column with the visit note.
    badval : `float`
        Value to use for bad values in slice
    verbose : `bool`
        Print extra information?
    """

    # Gap between visits in the same block, in hours
    gap_tolerance = 0.1

    def __init__(
        self,
        mjd_column_name="observationStartMJD",
        duration_column_name="visitTime",
        note_column_name="note",
        badval=np.NaN,
        verbose=False,
    ):
        super().__init__(verbose=verbose, badval=badval)
        self.mjd_column_name = mjd_column_name
        self.duration_column_name = duration_column_name
        self.note_column_name = note_column_name
        self.columnsNeeded = [
            mjd_column_name,
            duration_column_name,
            note_column_name,
        ]
        self.simIdxs = defaultdict(list)  # pylint: disable=invalid-name

    def setupSlicer(self, simData, maps=None):
        visits = pd.DataFrame(
            simData,
            index=pd.Index(
                np.arange(len(simData[self.mjd_column_name])), name="visit_idx"
            ),
        )
        visits.rename(
            columns={
                self.mjd_column_name: "mjd",
                self.duration_column_name: "duration",
                self.note_column_name: "note",
            },
            inplace=True,
        )
        # convert to hours
        visits.sort_values("mjd", inplace=True)
        visits["end_mjd"] = visits.mjd + visits.duration / (60 * 60 * 24.0)

        same_note = visits.note == visits.note.shift(-1)
        adjacent_times = (
            visits.end_mjd + self.gap_tolerance / 24.0 > visits.mjd.shift(-1)
        )
        visits["sid"] = (
            np.logical_not(np.logical_and(same_note, adjacent_times))
            .cumsum()
            .shift()
        )
        visits["sid"].fillna(0, inplace=True)
        visits["sid"] = visits["sid"].astype(int)

        blocks = visits.groupby("sid").agg(
            mjd=pd.NamedAgg(column="mjd", aggfunc="min"),
            end_mjd=pd.NamedAgg(column="end_mjd", aggfunc="max"),
        )
        blocks["duration"] = (blocks.end_mjd - blocks.mjd) * 24 * 60 * 60

        self.nslice = len(blocks)
        self.shape = self.nslice
        self.simIdxs.update(
            visits.reset_index()[["sid", "visit_idx"]]
            .groupby("sid")
            .agg(list)
            .to_dict()["visit_idx"]
        )

        self.slicePoints["sid"] = blocks.reset_index().sid.values
        self.slicePoints["mjd"] = blocks.mjd.values
        self.slicePoints["duration"] = blocks.duration.values
        self._runMaps(maps)

        @wraps(self._sliceSimData)
        def _sliceSimData(islice):  # pylint: disable=invalid-name
            idxs = self.simIdxs[islice]

            try:
                _ = idxs[0]
            except TypeError:
                idxs = [idxs]

            slice_points = {
                "mjd": self.slicePoints["mjd"][islice],
                "duration": self.slicePoints["duration"][islice],
            }

            return {"idxs": idxs, "slicePoint": slice_points}

        setattr(self, "_sliceSimData", _sliceSimData)


class VisitIntervalSlicer(TimeIntervalSlicer):
    """Slices into intervals each of which contain one visit

    Parameters
    ----------
    mjd_column_name : `str`
        Name of column on which to slice, must be in units of days
    duration_column_name : `str`
        Name of column with the duration of each visit (in seconds)
    badval : `float`
        Value to use for bad values in slice
    verbose : `bool`
        Print extra information?
    """

    def __init__(
        self,
        mjd_column_name="observationStartMJD",
        duration_column_name="visitTime",
        extra_column_names=tuple(),
        badval=np.NaN,
        verbose=False,
    ):
        super().__init__(verbose=verbose, badval=badval)
        self.mjd_column_name = mjd_column_name
        self.duration_column_name = duration_column_name
        self.extra_column_names = extra_column_names
        self.columnsNeeded = [mjd_column_name, duration_column_name]
        self.simIdxs = None  # pylint: disable=invalid-name

    def setupSlicer(self, simData, maps=None):
        self.nslice = len(simData[self.mjd_column_name])
        self.shape = self.nslice

        self.simIdxs = np.argsort(simData[self.mjd_column_name])
        self.slicePoints["sid"] = np.arange(self.nslice)
        self.slicePoints["mjd"] = simData[self.mjd_column_name]
        self.slicePoints["duration"] = simData[self.duration_column_name]
        for column_name in self.extra_column_names:
            self.slicePoints[column_name] = simData[column_name]
        self._runMaps(maps)

        @wraps(self._sliceSimData)
        def _sliceSimData(islice):  # pylint: disable=invalid-name
            idxs = self.simIdxs[islice]
            slice_points = {
                "sid": [idxs],
                "mjd": self.slicePoints["mjd"][islice],
                "duration": self.slicePoints["duration"][islice],
            }
            for column_name in self.extra_column_names:
                slice_points[column_name] = self.slicePoints[column_name]

            return {"idxs": idxs, "slicePoint": slice_points}

        setattr(self, "_sliceSimData", _sliceSimData)


# internal functions & classes
