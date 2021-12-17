from builtins import object
import os
from collections import OrderedDict
import numpy as np
from .mafRunResults import MafRunResults
import sqlite3
import pandas as pd
from rubin_sim.maf.utils import getSimData

__all__ = ["MafTracking"]


class MafTracking(object):
    """
    Class to read MAF's tracking SQLite database (tracking a set of MAF runs)
    and handle the output for web display.
    """

    def __init__(self, database=None):
        """
        Instantiate the (multi-run) layout visualization class.

        Parameters
        ----------
        database :str
           Path to the sqlite tracking database file.
           If not set, looks for 'trackingDb_sqlite.db' file in current directory.
        """
        if database is None:
            database = os.path.join(os.getcwd(), "trackingDb_sqlite.db")

        # Read in the results database.
        cols = [
            "mafRunId",
            "opsimRun",
            "opsimGroup",
            "mafComment",
            "opsimComment",
            "dbFile",
            "mafDir",
            "opsimVersion",
            "opsimDate",
            "mafVersion",
            "mafDate",
        ]
        self.runs = getSimData(database, "", cols, tableName="runs")
        self.runs = self.sortRuns(
            self.runs, order=["mafRunId", "opsimRun", "mafComment"]
        )
        self.runsPage = {}

    def runInfo(self, run):
        """
        Provide the tracking database information relevant for a given run in a format
        that the jinja2 templates can use.

        Parameters
        ----------
        run : `numpy.NDarray`
           One line from self.runs

        Returns
        -------
        runInfo : `OrderedDict`
            Ordered dict version of the numpy structured array.
        """
        runInfo = OrderedDict()
        runInfo["OpsimRun"] = run["opsimRun"]
        runInfo["OpsimGroup"] = run["opsimGroup"]
        runInfo["MafComment"] = run["mafComment"]
        runInfo["OpsimComment"] = run["opsimComment"]
        runInfo["SQLite File"] = [
            os.path.relpath(run["dbFile"]),
            os.path.split(run["dbFile"])[1],
        ]
        runInfo["ResultsDb"] = os.path.relpath(
            os.path.join(run["mafDir"], "resultsDb_sqlite.db")
        )
        runInfo["MafDir"] = run["mafDir"]
        runInfo["OpsimVersion"] = run["opsimVersion"]
        runInfo["OpsimDate"] = run["opsimDate"]
        runInfo["MafVersion"] = run["mafVersion"]
        runInfo["MafDate"] = run["mafDate"]
        return runInfo

    def sortRuns(self, runs, order=["opsimRun", "mafComment", "mafRunId"]):
        """
        Sort the numpy array of run data.

        Parameters
        ----------
        runs : `numpy.NDarray`
           The runs from self.runs to sort.
        order : `list`
           The fields to use to sort the runs array.

        Returns
        -------
        runs : `numpy.NDarray`
           A sorted numpy array.
        """
        return np.sort(runs, order=order)

    def getRun(self, mafRunId):
        """
        Set up a mafRunResults object to read and handle the data from an individual run.
        Caches the mafRunResults object, meaning the metric information from a particular run
        is only read once from disk.

        Parameters
        ----------
        mafRunId : `int`
           mafRunId value in the tracking database corresponding to a particular MAF run.

        Returns
        -------
        runPage : `MafRunResults`
           A MafRunResults object containing the information about a particular run.
           Stored internally in self.runsPage dict, but also passed back to the tornado server.
        """
        if not isinstance(mafRunId, int):
            if isinstance(mafRunId, dict):
                mafRunId = int(mafRunId["runId"][0][0])
            if isinstance(mafRunId, list):
                mafRunId = int(mafRunId[0])
        if mafRunId in self.runsPage:
            return self.runsPage[mafRunId]
        match = self.runs["mafRunId"] == mafRunId
        mafDir = self.runs[match]["mafDir"][0]
        runName = self.runs[match]["opsimRun"][0]
        if runName == "NULL":
            runName = None
        self.runsPage[mafRunId] = MafRunResults(mafDir, runName)
        return self.runsPage[mafRunId]
