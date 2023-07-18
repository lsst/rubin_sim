import shutil
import unittest

from rubin_sim.maf.db import ResultsDb
from rubin_sim.maf.run_comparison import combine_result_dbs


class TestGather(unittest.TestCase):
    def setUp(self):
        # Make two results database to access
        results_db = ResultsDb(out_dir="temp1")
        results_db.update_metric(
            "test_metric_name",
            "test_slicer_name",
            "temp",
            "test_sql_constraint",
            "test_metric_info_label",
            "test_metric_datafile",
        )

        results_db.update_summary_stat(1, "test_summary_name", summary_value=100.0)

        results_db = ResultsDb(out_dir="temp2")
        results_db.update_metric(
            "test_metric_name",
            "test_slicer_name",
            "temp",
            "test_sql_constraint",
            "test_metric_info_label",
            "test_metric_datafile",
        )

        results_db.update_summary_stat(1, "test_summary_name", summary_value=200.0)

    def tearDown(self):
        shutil.rmtree("temp1")
        shutil.rmtree("temp2")

    def test_gather_summaries(self):
        # Gather the results into a single db
        df = combine_result_dbs(run_dirs=["temp1", "temp2"])
        # Check that values are correct
        assert df.loc["temp1"].max() == 100.0
        assert df.loc["temp2"].max() == 200.0


if __name__ == "__main__":
    unittest.main()
