__all__ = ("sne_batch",)

import sqlite3
from os.path import basename

import pandas as pd

import rubin_sim.maf_proto as maf
from rubin_sim.data import get_baseline

DAYS_IN_YEAR = 365.25


class DoNothing:
    def __call__(*args, **kwargs):
        pass


def sne_batch(observations=None, run_name=None, quick_test=False, fig_saver=None, nside=16):
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
            df = pd.read_sql("select * from observations where night < 365;", con)
        else:
            df = pd.read_sql("select * from observations;", con)
    else:
        df = observations

    # But mostly want numpy array for speed.
    visits_array = df.to_records(index=False)
    con.close()

    info = maf.empty_info()
    info["run_name"] = run_name
    sl = maf.Slicer(nside=nside)
    metric = maf.SNNSNMetric()

    sn_array, info = sl(visits_array, metric, info=info)

    pm = maf.PlotMoll(info=info)

    fig = pm(sn_array["n_sn"], unit="N SNe to z limit")
    fig = pm(sn_array["zlim"], unit="z limit")
