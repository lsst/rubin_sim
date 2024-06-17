import glob
import os
import shutil
import tempfile
import unittest

from rubin_scheduler.data import get_data_dir
from rubin_scheduler.utils.code_utilities import sims_clean_up

import rubin_sim.maf.db as db
import rubin_sim.maf.maps as maps
import rubin_sim.maf.metric_bundles as metric_bundles
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.stackers as stackers

TEST_DB = "example_v3.4_0yrs.db"


class TestMetricBundle(unittest.TestCase):
    @classmethod
    def tearDown_class(cls):
        sims_clean_up()

    def setUp(self):
        self.out_dir = tempfile.mkdtemp(prefix="TMB")

    def test_out(self):
        """
        Check that the metric bundle can generate the expected output
        """
        nside = 8
        slicer = slicers.HealpixSlicer(nside=nside)
        metric = metrics.MeanMetric(col="airmass")
        sql = 'filter="r"'
        stacker1 = stackers.HourAngleStacker()
        stacker2 = stackers.GalacticStacker()
        map = maps.GalCoordsMap()

        metric_b = metric_bundles.MetricBundle(
            metric, slicer, sql, stacker_list=[stacker1, stacker2], maps_list=[map]
        )
        database = os.path.join(get_data_dir(), "tests", TEST_DB)

        results_db = db.ResultsDb(out_dir=self.out_dir)

        bgroup = metric_bundles.MetricBundleGroup(
            {0: metric_b}, database, out_dir=self.out_dir, results_db=results_db
        )
        bgroup.run_all()
        bgroup.plot_all()
        bgroup.write_all()

        out_thumbs = glob.glob(os.path.join(self.out_dir, "thumb*"))
        out_npz = glob.glob(os.path.join(self.out_dir, "*.npz"))
        out_pdf = glob.glob(os.path.join(self.out_dir, "*.pdf"))

        # By default, make 2 plots for healpix
        assert len(out_thumbs) == 2
        assert len(out_pdf) == 2
        assert len(out_npz) == 1

    def tearDown(self):
        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)


if __name__ == "__main__":
    unittest.main()
