# imports
import os
import sys
import unittest
from os import path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from rubin_sim import maf
from rubin_sim.data import get_data_dir

# constants

URLROOT = "https://raw.githubusercontent.com/lsst-pst/survey_strategy/main/fbs_2.0/"
FAMILY_SOURCE = URLROOT + "runs_v2.1.json"
METRIC_SUBSET_SOURCE = os.path.join(get_data_dir(), "maf", "metric_subsets.json")
SUMMARY_SOURCE = os.path.join(get_data_dir(), "maf", "baseline_summary.h5")

# exception classes

# interface functions

# classes


class TestArchive(unittest.TestCase):
    num_runs = 11
    num_metrics = 7

    def test_get_family_runs(self):
        runs = maf.get_family_runs(FAMILY_SOURCE)

        self.assertIsInstance(runs, pd.DataFrame)

        columns = ("run", "brief", "filepath", "url")
        for column in columns:
            self.assertIn(column, runs.columns)

        self.assertEqual(runs.index.name, "family")

    def test_get_runs(self):
        runs = maf.get_runs(FAMILY_SOURCE)
        self.assertEqual(runs.index.name, "run")

        columns = ("family", "brief", "filepath", "url")
        for column in columns:
            self.assertIn(column, runs.columns)

        self.assertEqual(runs.index.name, "run")

    @unittest.skip("skipping long running test.")
    def test_download_runs(self):
        run = maf.get_runs(FAMILY_SOURCE).index.values[0]

        temp_dir_itself = TemporaryDirectory()
        temp_dir = temp_dir_itself.name

        dest_fnames = maf.download_runs(run, dest_dir=temp_dir)
        self.assertTrue(path.exists(dest_fnames[0]))

        temp_dir_itself.cleanup()

    def test_get_metric_subsets(self):
        metric_subsets = maf.get_metric_subsets(METRIC_SUBSET_SOURCE)
        self.assertIsInstance(metric_subsets, pd.DataFrame)
        self.assertIn("metric subset", metric_subsets.index.names)
        self.assertIn("metric", metric_subsets.index.names)

        columns = ("style", "invert", "mag")
        for column in columns:
            self.assertIn(column, metric_subsets.columns)

    def test_get_metric_summaries(self):
        summary = maf.get_metric_summaries(summary_source=SUMMARY_SOURCE)
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertEqual(summary.columns.name, "metric")
        self.assertEqual(summary.index.name, "run")

        my_summary = maf.get_metric_summaries(
            runs=["baseline_v3.0_10yrs", "baseline_v3.4_10yrs"],
            metrics=[
                "Rms Max normairmass All sky all bands HealpixSlicer",
                "Median Max normairmass All sky all bands HealpixSlicer",
                "Max Max normairmass All sky all bands HealpixSlicer",
            ],
            summary_source=summary,
        )
        self.assertEqual(len(my_summary), 2)
        self.assertEqual(len(my_summary.columns), 3)

        srd_sim = maf.get_metric_summaries(
            metric_subsets="SRD",
            summary_source=summary,
            metric_subset_source=METRIC_SUBSET_SOURCE,
        )
        self.assertGreater(len(srd_sim.columns), 3)
        self.assertLess(len(srd_sim.columns), len(summary.columns))

        srdnvis_sim = maf.get_metric_summaries(
            metric_subsets=["SRD", "N Visits"],
            summary_source=summary,
            metric_subset_source=METRIC_SUBSET_SOURCE,
        )
        self.assertGreater(len(srdnvis_sim.columns), len(srd_sim.columns))
        self.assertLess(len(srdnvis_sim.columns), len(summary.columns))

    def test_get_family_descriptions(self):
        families = maf.get_family_descriptions(FAMILY_SOURCE)
        self.assertIsInstance(families, pd.DataFrame)
        self.assertEqual(families.index.name, "family")

    def test_describe_families(self):
        if "IPython" in sys.modules:
            with (
                patch("IPython.display.display_markdown") as _,
                patch("IPython.display.HTML") as _,
                patch("IPython.display") as _,
            ):
                self.perform_describe_families_test()
        else:
            with patch("builtins.print") as _:
                self.perform_describe_families_test()

    @unittest.skip("Skipping; family descriptions out of sync with summary")
    def perform_describe_families_test(self):
        families = maf.get_family_descriptions(FAMILY_SOURCE)
        disp_families = families[:2]

        fig, ax = maf.describe_families(families)
        self.assertIsNone(fig)
        self.assertIsNone(ax)

        all_metric_subsets = maf.get_metric_subsets(METRIC_SUBSET_SOURCE)
        summary = maf.get_metric_summaries(summary_source=SUMMARY_SOURCE)

        table_metric_subset = all_metric_subsets.loc["SRD"]
        fig, ax = maf.describe_families(
            disp_families, summary=summary, table_metric_subset=table_metric_subset
        )
        self.assertIsNone(fig)
        self.assertIsNone(ax)

        plot_metric_subset = all_metric_subsets.loc["N Visits"]
        fig, ax = maf.describe_families(disp_families, summary=summary, plot_metric_subset=plot_metric_subset)

    def test_create_metric_subset(self):
        metrics = ["Urania", "Thalia", "Calliope", "Terpsichore"]
        metric_subset_name = "Muses"
        metric_subset = maf.create_metric_subset(metric_subset_name, metrics)
        self.assertSequenceEqual(metrics, metric_subset.metric.tolist())
        self.assertSequenceEqual(
            metric_subset.columns.tolist(),
            ["metric", "short_name", "style", "invert", "mag"],
        )
        self.assertSequenceEqual(metric_subset.index.names, ["metric subset", "metric"])


run_tests_now = __name__ == "__main__"
if run_tests_now:
    unittest.main()
