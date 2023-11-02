import os
import sqlite3
import unittest
from tempfile import TemporaryDirectory

import pandas as pd

import rubin_sim.maf.db as db
import rubin_sim.maf.metric_bundles as metric_bundles
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.plots as plots
import rubin_sim.maf.slicers as slicers


class TestScalarHourglassPlots(unittest.TestCase):
    def setUp(self):
        self._data_dir_itself = TemporaryDirectory()
        opsdb_fname = create_sample_visit_db(self._data_dir_itself.name)
        self._out_dir_itself = TemporaryDirectory()
        self.out_dir = self._out_dir_itself.name
        self.results_db = db.ResultsDb(out_dir=self.out_dir)
        self.ops_db = db.OpsimDatabase(opsdb_fname)
        self.slicer = slicers.TimeIntervalSlicer(interval_seconds=3600)
        self.metric = metrics.MedianMetric("seeingFwhmEff")

    @unittest.skip("This is now throwing astropy.utils.iers.iers.IERSRangeError, but only sometimes?")
    def test_month_plot(self):
        sql_constraint = ""
        metric_bundle = metric_bundles.MetricBundle(self.metric, self.slicer, sql_constraint)
        bundle_dict = {1: metric_bundle}
        bundle_group = metric_bundles.MetricBundleGroup(
            bundle_dict,
            self.ops_db,
            out_dir=self.out_dir,
            results_db=self.results_db,
        )
        _ = bundle_group.run_all()
        plot_func = plots.YearHourglassPlot(2023)
        _ = metric_bundle.plot(plotFunc=plot_func)


class TestTimeUseHourglassPlots(unittest.TestCase):
    def setUp(self):
        self._data_dir_itself = TemporaryDirectory()
        opsdb_fname = create_sample_visit_db(self._data_dir_itself.name)

        self._out_dir_itself = TemporaryDirectory()
        self.out_dir = self._out_dir_itself.name
        self.results_db = db.ResultsDb(out_dir=self.out_dir)
        self.ops_db = db.OpsimDatabase(opsdb_fname)
        self.slicer = slicers.BlockIntervalSlicer()
        self.metric = metrics.UseMetric()

    @unittest.skip("XXX-seems like this was erroneously passing before?")
    def test_year_plot(self):
        sql_constraint = ""
        metric_bundle = metric_bundles.MetricBundle(self.metric, self.slicer, sql_constraint)
        bundle_dict = {1: metric_bundle}
        bundle_group = metric_bundles.MetricBundleGroup(
            bundle_dict,
            self.ops_db,
            out_dir=self.out_dir,
            results_db=self.results_db,
        )
        _ = bundle_group.run_all()
        plot_func = plots.YearHourglassUsePlot(2023)
        _ = metric_bundle.plot(plotFunc=plot_func)


def create_sample_visit_db(dir_name):
    fname = os.path.join(dir_name, "opsim.sqlite3")

    sample_visits_dict = {
        "observationStartMJD": {
            75831: 59974.1403819843,
            88243: 59991.13314155574,
            84877: 59986.16509162588,
            73474: 59971.038614223085,
            91024: 59994.3667217889,
            94567: 60001.38094520713,
            88867: 59992.06039742178,
            91397: 59995.17212771153,
            88282: 59991.151247547146,
            82396: 59983.0460827712,
        },
        "visitTime": {
            75831: 34.0,
            88243: 34.0,
            84877: 34.0,
            73474: 34.0,
            91024: 34.0,
            94567: 34.0,
            88867: 34.0,
            91397: 34.0,
            88282: 34.0,
            82396: 34.0,
        },
        "seeingFwhmEff": {
            75831: 0.9580660562409485,
            88243: 1.2816139136535794,
            84877: 0.8711626732163685,
            73474: 1.138078694389459,
            91024: 0.6851416920629385,
            94567: 0.9526705179882091,
            88867: 1.0749841095670385,
            91397: 0.9092889658087193,
            88282: 1.3069907982221605,
            82396: 0.8034796448061345,
        },
        "filter": {
            75831: "y",
            88243: "g",
            84877: "z",
            73474: "r",
            91024: "r",
            94567: "y",
            88867: "r",
            91397: "r",
            88282: "g",
            82396: "r",
        },
        "note": {
            75831: "blob, zy, b",
            88243: "blob, ug, b",
            84877: "blob, iz, b",
            73474: "greedy",
            91024: "greedy",
            94567: "greedy",
            88867: "blob, ur, b",
            91397: "blob, ri, a",
            88282: "blob, gr, a",
            82396: "DD:ECDFS",
        },
    }

    visits = pd.DataFrame(sample_visits_dict)
    with sqlite3.connect(fname) as con:
        visits.to_sql("observations", con=con)

    return fname


run_tests_now = __name__ == "__main__"
if run_tests_now:
    unittest.main()
