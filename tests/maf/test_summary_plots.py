# imports
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import default_rng

from rubin_sim import maf

# constants

RANDOM_SEED = 6563

# exception classes

# interface functions

# classes


class TestSummaryPlots(unittest.TestCase):
    num_runs = 11
    num_metrics = 7

    def setUp(self):
        # Make tests reproducible
        self.rng = default_rng(RANDOM_SEED)

        self.runs = [f"run{i}" for i in range(self.num_runs)]
        self.baseline = self.runs[0]

        self.metrics = [f"metric{i}" for i in range(self.num_metrics)]
        self.inverted_metrics = ["metric3", "metric5", "metric6"]
        self.mag_metrics = ["metric1", "metric2", "metric6"]

        self.metric_values = pd.DataFrame(
            self.rng.normal(loc=3, scale=5, size=(self.num_runs, self.num_metrics)),
            columns=self.metrics,
            index=self.runs,
        )
        self.metric_values.columns.name = "metric"
        self.metric_values.index.name = "run"

        styles = ["-" for i in range(self.num_metrics)]

        self.metric_subset = pd.DataFrame(
            {"mag": False, "invert": False, "metric": self.metrics, "style": styles}
        ).set_index("metric", drop=False)
        self.metric_subset.loc[self.mag_metrics, "mag"] = True
        self.metric_subset.loc[self.inverted_metrics, "invert"] = True
        self.metric_subset.loc["metric3", "style"] = "b--"

    def test_normalize_metric_summaries(self):
        # Test standard normalization with one run
        norm_values = maf.normalize_metric_summaries(self.baseline, self.metric_values, self.metric_subset)

        ref_norm_values = _run_infos_norm_df(
            self.metric_values,
            self.baseline,
            invert_cols=self.inverted_metrics,
            mag_cols=self.mag_metrics,
        )
        np.testing.assert_allclose(norm_values.values, ref_norm_values.values)

        # test normalizing against one run, as a list
        norm_values = maf.normalize_metric_summaries([self.baseline], self.metric_values, self.metric_subset)
        np.testing.assert_allclose(norm_values.values, ref_norm_values.values)

        # test similar but pretend that self.baseline is two runs
        norm_values = maf.normalize_metric_summaries(
            [self.baseline, self.baseline], self.metric_values, self.metric_subset
        )
        np.testing.assert_allclose(norm_values.values, ref_norm_values.values)

        # test similar but different runs
        norm_values = maf.normalize_metric_summaries(
            [self.runs[0], self.runs[1]], self.metric_values, self.metric_subset
        )

    def test_plot_run_metric(self):
        fig, ax = maf.plot_run_metric(self.metric_values)

        fig, ax = maf.plot_run_metric(
            self.metric_values,
            baseline_run=self.baseline,
            metric_subset=self.metric_subset,
        )

        fig, ax = maf.plot_run_metric(
            self.metric_values,
            vertical_quantity="value",
            horizontal_quantity="run",
        )

        fig, ax = maf.plot_run_metric(
            self.metric_values,
            vertical_quantity="value",
            horizontal_quantity="metric",
        )

        run_label_map = {r: r + "foo" for r in self.runs}
        fig, ax = maf.plot_run_metric(self.metric_values, run_label_map=run_label_map)

        metric_label_map = {r: r + "foo" for r in self.metrics}
        fig, ax = maf.plot_run_metric(self.metric_values, metric_label_map=metric_label_map)

        fig, ax = maf.plot_run_metric(self.metric_values, cmap=plt.get_cmap("Set2"))

        fig, ax = maf.plot_run_metric(self.metric_values, linestyles=["-", "--", ":"])

        fig, ax = maf.plot_run_metric(self.metric_values, markers=["o", "v", "^", "<", ">", "8", "s"])

        fig, ax = plt.subplots()
        maf.plot_run_metric(self.metric_values, ax=ax)

    def test_plot_run_metric_mesh(self):
        fig, ax = maf.plot_run_metric_mesh(self.metric_values)

        fig, ax = maf.plot_run_metric_mesh(
            self.metric_values,
            baseline_run=self.baseline,
            metric_subset=self.metric_subset,
        )

        fig, ax = maf.plot_run_metric_mesh(
            self.metric_values,
            baseline_run=self.baseline,
        )

        fig, ax = maf.plot_run_metric_mesh(self.metric_values, color_range=12)

        run_label_map = {r: r + "foo" for r in self.runs}
        fig, ax = maf.plot_run_metric_mesh(self.metric_values, run_label_map=run_label_map)

        metric_label_map = {r: r + "foo" for r in self.metrics}
        fig, ax = maf.plot_run_metric_mesh(self.metric_values, metric_label_map=metric_label_map)

        fig, ax = maf.plot_run_metric_mesh(self.metric_values, cmap=plt.get_cmap("Spectral"))

        fig, ax = plt.subplots()
        maf.plot_run_metric_mesh(self.metric_values, ax=ax)


# internal functions & classes


def _run_infos_norm_df(df, norm_run, invert_cols=None, mag_cols=None):
    """
    Normalize values in a DataFrame, based on the values in a given run.
    Can normalize some columns (metric values) differently
    (invert_cols, reverse_cols, mag_cols)
    if those columns are specified; this lets the final normalized dataframe
    'look' the same way
    in a plot (i.e. "up" is better (reverse_cols), they center on 1 (mag_cols),
    and the magnitude scales as expected (invert_cols)).

    Parameters
    ----------
    df : pd.DataFrame
        The data frame containing the metric values to compare
    norm_run: str
        The name of the simulation to normalize to (typically family_baseline)
    invert_cols: list
        Columns (metric values) to convert to 1 / value
    mag_cols: list
        Columns (metrics values) to treat as magnitudes
        (1 + (difference from norm_run))

    Returns
    -------
    pd.DataFrame
        Normalized data frame
    """

    # Copy the dataframe but drop the columns containing only strings
    out_df = df.copy()
    if invert_cols is None:
        invert_cols = []
    if mag_cols is None:
        mag_cols = []
    both_cols = [c for c in invert_cols if c in mag_cols]
    invert_cols = [c for c in invert_cols if c not in both_cols]
    mag_cols = [c for c in mag_cols if c not in both_cols]
    out_df[mag_cols] = 1 + out_df[mag_cols] - out_df[mag_cols].loc[norm_run]
    out_df[invert_cols] = 1 / out_df[invert_cols]
    out_df[both_cols] = 1 - out_df[both_cols] + out_df[both_cols].loc[norm_run]
    # which columns are strings?
    string_cols = [c for c, t in zip(df.columns, df.dtypes) if t == "object"]
    cols = [c for c in out_df.columns.values if not (c in mag_cols or c in string_cols)]
    out_df[cols] = 1 + (out_df[cols] - out_df[cols].loc[norm_run]) / out_df[cols].loc[norm_run]
    return out_df


run_tests_now = __name__ == "__main__"
if run_tests_now:
    unittest.main()
