import unittest

import numpy as np
import pandas as pd
import os

from rubin_sim.maf.metrics.mdwarf_flares_metrics import MDwarfCountMetric, MDwarfMeasureMetric
from rubin_sim.maf import slicers
from rubin_sim.maf import metricBundles


def test_metric(self, metric):
    """
    This can test either m dwarf metric with
    simplistic data slices.

    Parameters
    ----------
    metric: MDwarfCountMetric or MDwarfMeasureMetric
    """
    # check that if there are no points near t0 then it's false
    filters = ["u", "g", "r"]
    time_rel = [-10, -9, -8]
    snr = [100, 100, 100]
    flux = [1000, 1001, 2000]
    fluxerr = [1, 1, 1]
    test_data_slice_too_early = [
        [
            pd.DataFrame(
                {"filter": filters, "time_rel": time_rel, "snr": snr, "flux": flux, "fluxerr": fluxerr}
            )
        ]
    ]
    assert not metric.run(self, test_data_slice_too_early, None)["metric_value"]

    # check that the pre flare average is computed correctly
    # if we give it some constant points
    filters = ["g", "g", "g", "g", "g", "g"]
    time_rel = [-10, -9, -8, 0.01, 0.02, 0.03]
    snr = [100, 100, 100, 100, 100, 100]
    flux = [1000, 1001, 999, 2000, 2000, 2000]
    fluxerr = [1, 1, 1, 1, 1, 1]
    test_data_slice_average = [
        [
            pd.DataFrame(
                {"filter": filters, "time_rel": time_rel, "snr": snr, "flux": flux, "fluxerr": fluxerr}
            )
        ]
    ]
    assert np.isclose(
        metric.run(self, test_data_slice_average, None)["pre_flare_average"].nominal_value, 1000
    )

    # check that it returns false if we have a densely sampled flat line
    filters = ["g"] * 100
    time_rel = np.linspace(-10, 10, 100)
    snr = np.full(100, 100)
    flux = [10000] * 100
    fluxerr = [1] * 100
    test_data_slice_flat = [
        [
            pd.DataFrame(
                {"filter": filters, "time_rel": time_rel, "snr": snr, "flux": flux, "fluxerr": fluxerr}
            )
        ]
    ]
    assert not metric.run(self, test_data_slice_flat, None)["metric_value"]

    # check that it returns true if we have a densely sampled transient
    filters = ["g"] * 100
    time_rel = np.linspace(-10, 10, 100)
    snr = np.full(100, 100)
    flux = ([10000] * 50) + ([20000] * 50)
    fluxerr = [1] * 100
    test_data_slice_transient = [
        [
            pd.DataFrame(
                {"filter": filters, "time_rel": time_rel, "snr": snr, "flux": flux, "fluxerr": fluxerr}
            )
        ]
    ]
    assert metric.run(self, test_data_slice_transient, None)["metric_value"]

    # check that it returns false if we have a dense transient that's too noisy
    filters = ["g"] * 100
    time_rel = np.linspace(-10, 10, 100)
    snr = np.full(100, 100)
    flux = ([10000] * 50) + ([20000] * 50)
    fluxerr = [10000] * 100
    test_data_slice_noisy_transient = [
        [
            pd.DataFrame(
                {"filter": filters, "time_rel": time_rel, "snr": snr, "flux": flux, "fluxerr": fluxerr}
            )
        ]
    ]
    assert not metric.run(self, test_data_slice_noisy_transient, None)["metric_value"]


def test_with_lcl_slicer(self, metric):
    """loads a data file from lightcurvelynx output
    and runs it through the slicer."""
    parent_dir = os.path.dirname(os.path.realpath(__file__))

    sim_data = os.path.join(parent_dir, "../data/mdwarf_test_data.parquet")
    out_dir = os.path.join(parent_dir, "../data")  # can't make this None
    results_db = None
    slicer = slicers.LynxBaseSlicer()
    metric = metric(col="lightcurve")
    bundle = metricBundles.MetricBundle(metric, slicer, constraint="")
    group = metricBundles.MetricBundleGroup(
        [bundle],
        sim_data,
        out_dir=out_dir,
        results_db=results_db,
        db_table="lightcurves",
    )
    group.run_all()

    # should remove extra files but i am anxious about
    # deleting too much by accident
    # so for now i will leave it as is
    return


class TestMDwarfMetric(unittest.TestCase):
    """
    Tests the M Dwarf Metrics
    """

    def test_simple_tests(self):
        for metric in [MDwarfCountMetric, MDwarfMeasureMetric]:
            test_metric(self, metric)

    def test_single_point(self):
        """
        Test that if we only have a single point in the flare it
        returns True for the Count metric and does not
        return True for the Measure metric
        """
        filters = ["g"] * 100
        time_rel = np.hstack((np.linspace(-20, -10, 99), [0.1]))
        snr = np.full(100, 100)
        flux = ([10000] * 99) + ([20000] * 1)
        fluxerr = [10] * 100
        test_data_slice_onepoint = [
            [
                pd.DataFrame(
                    {"filter": filters, "time_rel": time_rel, "snr": snr, "flux": flux, "fluxerr": fluxerr}
                )
            ]
        ]
        assert MDwarfCountMetric.run(self, test_data_slice_onepoint, None)["metric_value"]
        assert not MDwarfMeasureMetric.run(self, test_data_slice_onepoint, None)["metric_value"]

    def test_with_slicer(self):
        for metric in [MDwarfCountMetric, MDwarfMeasureMetric]:
            test_with_lcl_slicer(self, metric)


if __name__ == "__main__":
    unittest.main()
