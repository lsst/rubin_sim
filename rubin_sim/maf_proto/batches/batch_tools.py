__all__ = ("batch_preamble",)

import sqlite3
from os.path import basename

import pandas as pd

from rubin_sim.data import get_baseline


class DoNothing:
    def __call__(*args, **kwargs):
        pass
        
    def save_stats(*args, **kwargs):
        pass


def batch_preamble(observations=None, run_name=None, quick_test=False, fig_saver=None):
    if fig_saver is None:
        fig_saver = DoNothing()

    if observations is None:
        observations = get_baseline()
    if run_name is None:
        run_name = basename(observations).replace(".db", "")

    if isinstance(observations, str):
        con = sqlite3.connect(observations)
        # Dataframe is handy for some calcs
        if quick_test:
            df = pd.read_sql("select * from observations where night < 61;", con)
            subset = "night < 61"
        else:
            df = pd.read_sql("select * from observations;", con)
            subset = "all"
        con.close()
    else:
        df = observations

    # But mostly want numpy array for speed.
    visits_array = df.to_records(index=False)

    return visits_array, df, run_name, subset, fig_saver
