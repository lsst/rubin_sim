import os
import shutil
import tempfile
import unittest

import numpy as np

import rubin_sim.maf.batches as batches
import rubin_sim.maf.db as db
import rubin_sim.maf.metric_bundles as metric_bundles
from rubin_sim.data import get_data_dir
from rubin_sim.maf.slicers import MoObjSlicer
from rubin_sim.utils.code_utilities import sims_clean_up


class TestBatches(unittest.TestCase):
    @classmethod
    def tearDown_class(cls):
        sims_clean_up()

    def setUp(self):
        self.out_dir = tempfile.mkdtemp(prefix="TMB")

    @unittest.skipUnless(
        os.path.isdir(os.path.join(get_data_dir(), "maf")),
        "Skip these batches unless MAF data present, required for setup",
    )
    def testload_them_all(self):
        ack = batches.altazHealpix()
        assert ack is not None
        ack = batches.altazLambert()
        assert ack is not None
        ack = batches.standard_summary()
        assert ack is not None
        ack = batches.standard_metrics("night")
        assert ack is not None
        ack = batches.filtersPerNight()
        assert ack is not None
        ack = batches.filtersWholeSurvey()
        assert ack is not None
        ack = batches.glanceBatch()
        assert ack is not None
        ack = batches.hourglassPlots()
        assert ack is not None
        ack = batches.slewBasics()
        assert ack is not None
        ack = batches.timeGaps()
        assert ack is not None
        ack = batches.metadataBasics("airmass")
        assert ack is not None
        ack = batches.metadataBasicsAngle("rotskyPos")
        assert ack is not None
        ack = batches.metadataMaps("fiveSigmaDepth")
        assert ack is not None

    @unittest.skipUnless(
        os.path.isdir(os.path.join(get_data_dir(), "maf")),
        "Skip these batches unless MAF data present, required for setup",
    )
    def test_moving_objects_batches(self):
        slicer = MoObjSlicer()
        ack = batches.quick_discovery_batch(slicer)
        assert ack is not None
        ack = batches.discovery_batch(slicer)
        assert ack is not None
        ack = batches.characterization_inner_batch(slicer)
        assert ack is not None
        ack = batches.characterization_outer_batch(slicer)
        assert ack is not None

    def test_moving_fractions(self):
        bdict = {}
        bundle = metric_bundles.create_empty_mo_metric_bundle()
        bundle.run_name = "dummy_name"
        bdict["days"] = bundle
        bdict["days"].slicer = MoObjSlicer()
        bdict["days"].slicer.slice_points["H"] = np.arange(50)
        # Should probably fill in some data as well to make sure
        # more parts of the code get exercised, but this is a start.
        ack = batches.run_completeness_summary(bdict, 0.0, [10, 20], None, None)
        assert ack is not None
        ack = batches.run_fraction_summary(bdict, 0.0, None, None)
        assert ack is not None
        ack = batches.plot_fractions(bdict, out_dir=self.out_dir)
        assert ack is not None
        # This batch takes a single bundle because it plots that single
        # bundle with multiple summary interpretations of the metric
        ack = batches.plot_single(bundle, out_dir=self.out_dir)
        assert ack is not None
        ack = batches.plot_activity(bdict)
        assert ack is not None

    @unittest.skipUnless(
        os.path.isdir(os.path.join(get_data_dir(), "maf")),
        "Skipping scienceRadarBatch test because operating without full MAF test data",
    )
    def test_science_radar(self):
        # Loading the science radar batch requires reading a significant
        # set of input files
        # This test is skipped if running with the lighter set of test data.
        # batch requires reading a lot of input files for lightcurves
        ack = batches.science_radar_batch()
        assert ack is not None

    @unittest.skipUnless(
        os.path.isdir(os.path.join(get_data_dir(), "maf")),
        "Skipping glance test because operating without full MAF test data",
    )
    def test_glance(self):
        ack = batches.glanceBatch()
        database = os.path.join(get_data_dir(), "tests", "example_dbv1.7_0yrs.db")
        results_db = db.ResultsDb(out_dir=self.out_dir)
        bgroup = metric_bundles.MetricBundleGroup(ack, database, out_dir=self.out_dir, results_db=results_db)
        bgroup.run_all()

    def tearDown(self):
        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)


if __name__ == "__main__":
    unittest.main()
