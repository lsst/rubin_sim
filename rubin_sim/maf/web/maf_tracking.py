__all__ = ("MafTracking",)

import os
from collections import OrderedDict

import numpy as np

from rubin_sim.maf.utils import get_sim_data

from .maf_run_results import MafRunResults


class MafTracking:
    """Hold and serve the MAF tracking (sqlite) database content for
    the show_maf web server.

    Parameters
    ----------
    database : `str`, optional
        Path to the sqlite tracking database file.
        If None, looks for `trackingDb_sqlite.db` default file in the
        current directory.
    """

    def __init__(self, database=None):
        if database is None:
            database = os.path.join(os.getcwd(), "trackingDb_sqlite.db")
        self.tracking_db = database

        # Read in the tracking database.
        cols = [
            "maf_run_id",
            "run_name",
            "run_group",
            "maf_comment",
            "run_comment",
            "db_file",
            "maf_dir",
            "run_version",
            "run_date",
            "maf_version",
            "maf_date",
        ]
        self.runs = get_sim_data(self.tracking_db, "", cols, table_name="runs")
        self.runs = self.sort_runs(self.runs, order=["maf_run_id", "run_name", "maf_comment"])
        self.runs_page = {}

    def run_info(self, run):
        """Get the tracking database information relevant for a given run
        in a format that the jinja2 templates for show_maf  can use.

        Parameters
        ----------
        run : `np.ndarray`, (1,)
           One line from self.runs

        Returns
        -------
        run_info : `OrderedDict`
            Ordered dict version of the numpy structured array.
        """
        runInfo = OrderedDict()
        maf_dir = os.path.relpath(run["maf_dir"], start=os.path.dirname(self.tracking_db))
        runInfo["Run Name"] = run["run_name"]
        runInfo["Group"] = run["run_group"]
        runInfo["Maf Comment"] = run["maf_comment"]
        runInfo["Run Comment"] = run["run_comment"]
        runInfo["RunDb File"] = [
            os.path.relpath(run["db_file"]),
            os.path.split(run["db_file"])[1],
        ]
        runInfo["ResultsDb"] = os.path.join(maf_dir, "resultsDb_sqlite.db")
        runInfo["maf_dir"] = maf_dir
        runInfo["sched_version"] = run["run_version"]
        runInfo["sched_date"] = run["run_date"]
        runInfo["maf_version"] = run["maf_version"]
        runInfo["maf_date"] = run["maf_date"]
        return runInfo

    def sort_runs(self, runs, order=["run_name", "maf_comment", "maf_run_id"]):
        """Sort  the numpy array of run data.

        Parameters
        ----------
        runs : `np.ndarray`, (N,)
           The runs from self.runs to sort.
        order : `list` [`str`]
           The fields to use to sort the runs array.

        Returns
        -------
        runs : `numpy.NDarray`
           A sorted numpy array.
        """
        return np.sort(runs, order=order)

    def get_run(self, maf_run_id):
        """Set up a mafRunResults object to read and handle the data
        from a single individual run.
        Caches the mafRunResults object, meaning the metric information from
        a particular run is only read once from disk.

        Parameters
        ----------
        maf_run_id : `int`
           maf_run_id value in the tracking database
           corresponding to a particular MAF run.

        Returns
        -------
        runPage : `MafRunResults`
           A MafRunResults object containing the information for this run.
           Stored internally in self.runs_page dict, but also passed
           back to the tornado server.
        """
        if not isinstance(maf_run_id, int):
            if isinstance(maf_run_id, dict):
                maf_run_id = int(maf_run_id["maf_run_id"][0][0])
            if isinstance(maf_run_id, list):
                maf_run_id = int(maf_run_id[0])
        if maf_run_id in self.runs_page:
            return self.runs_page[maf_run_id]
        match = self.runs["maf_run_id"] == maf_run_id
        maf_dir = self.runs[match]["maf_dir"][0]
        run_name = self.runs[match]["run_name"][0]
        if run_name == "NULL":
            run_name = None
        self.runs_page[maf_run_id] = MafRunResults(maf_dir, run_name)
        return self.runs_page[maf_run_id]
