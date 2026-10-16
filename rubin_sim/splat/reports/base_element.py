__all__ = ("BaseReportElement",)


import sqlite3
from os.path import basename

import pandas as pd

from rubin_sim.data import get_baseline


class DoNothing:
    def __call__(*args, **kwargs):
        pass

    def save_stats(*args, **kwargs):
        pass


class BaseReportElement:
    """Base class for making report element(s)."""

    def __init__(
        self, observations=None, run_name=None, quick_test=False, fig_saver=None, info=None, 
        subset="", quick_night=61
    ):

        if fig_saver is None:
            self.fig_saver = DoNothing()
        else:
            self.fig_saver = fig_saver

        self.run_name = run_name
        self.observations = observations
        self.quick_test = quick_test
        self.info = info
        self.quick_night = quick_night
        self.subset = subset

        if quick_test:
            self.load_visits()

    def load_visits(self):
        if self.observations is None:
            self.observations = get_baseline()

        if self.run_name is None:
            self.run_name = basename(self.observations).replace(".db", "")

        if isinstance(self.observations, str):
            con = sqlite3.connect(self.observations)
            # Dataframe is handy for some calcs
            if self.quick_test:
                visits_df = pd.read_sql(
                    "select * from observations where night < %i;" % self.quick_night, con
                )
                self.subset = "night < %i" % self.quick_night
            else:
                visits_df = pd.read_sql("select * from observations;", con)
                self.subset = "all"
            con.close()
        else:
            visits_df = self.observations
            self.subset = self.subset

        # Add any columns to the dataframe
        visits_df = self.add_cols(visits_df)
        self.visits_df = visits_df

    def add_cols(self, df):
        """Add any columns to the dataframe that are needed"""
        # Check if needed columns already exist, if not add them

        return df

    def visit_resolver(self, visits_df, visits_array, make_array=True):
        if visits_df is None:
            self.load_visits()
            visits_df = self.visits_df
        else:
            visits_df = self.add_cols(visits_df)

        if make_array:
            if visits_array is None:
                visits_array = visits_df.to_records(index=False)

        return visits_df, visits_array

    def compute_elements(self, visits_df=None, visits_array=None, **kwargs):
        list_of_data = []
        list_of_infos = []
        summary_dataframe = None

        return list_of_data, list_of_infos, summary_dataframe

    def __call__(self, visits_df=None, visits_array=None, **kwargs):
        """return a list of arrays, list of infos, and dataframe of summary stats
        """
        # Make sure we have some data loaded and generate the array if needed.
        visits_df, visits_array = self.visit_resolver(visits_df, visits_array)

        list_of_data, list_of_infos, summary_dataframe = self.compute_elements(visits_df=visits_df,
                                                                               visits_array=visits_array,
                                                                               **kwargs)

        return list_of_data, list_of_infos, summary_dataframe
