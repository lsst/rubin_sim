# Tests to verify the functionality of the LightCurveLynx interface within
# the Rubin Sim MAF framework. These require the lightcurvelynx library be
# installed, so they will be skipped if the library is not available.

import numpy as np
import unittest

from rubin_sim.maf.metrics.lynx_metrics import LynxDetectionMetric
from rubin_sim.maf.slicers.lynx_slicers import LynxSamplerSlicer

try:
    from lightcurvelynx.models.basic_models import SinWaveModel

    HAS_LIBRARY = True
except ImportError:
    HAS_LIBRARY = False


class TestLightCurveLynxInterface(unittest.TestCase):
    """The test cases for the LightCurveLynx interface that require the library to be installed."""

    def setUp(self):
        # If the library is not available, skip the tests.
        if not HAS_LIBRARY:
            self.skipTest("lightcurvelynx is not installed")

        # Create fake 'matched' data with the columns we need.
        self.single_query_data = {
            "observationStartMJD": np.array([0.0, 1.0, 2.0, 3.0]),
            "fieldRA": np.array([15.0, 15.0, 15.0, 15.0]),
            "fieldDec": np.array([-10.0, -10.0, -10.0, -10.0]),
            "filter": np.array(["r", "g", "r", "r"]),
            "seeingFwhmEff": [1.12] * 4,
            "skyBrightness": [20.0] * 4,
            "visitExposureTime": [29.2] * 4,
            "numExposures": [1] * 4,
            "airmass": [1.0] * 4,
        }

        # Create a toy model.
        self.model = SinWaveModel(
            brightness=2000.0,
            amplitude=200.0,
            frequency=0.01,
            t0=0.0,
            ra=200.5,
            dec=-50.0,
            node_label="sin_wave_model",
        )

    def test_lynx_sample_slicer(self):
        """Test we can create a LynxSamplerSlicer."""
        states = self.model.sample_parameters(num_samples=5)
        ra_vals = states["sin_wave_model"]["ra"]
        dec_vals = states["sin_wave_model"]["dec"]

        # Create the slicer directly from the states.
        slicer = LynxSamplerSlicer(states, self.model)
        self.assertIsInstance(slicer, LynxSamplerSlicer)
        self.assertTrue(slicer.slice_points["lynx_model"] is self.model)
        self.assertEqual(slicer.slice_points["lynx_params"].shape, (5,))

        for idx, sample in enumerate(slicer.slice_points["lynx_params"]):
            self.assertTrue(np.allclose(sample["sin_wave_model"]["ra"], ra_vals[idx]))
            self.assertTrue(np.allclose(sample["sin_wave_model"]["dec"], dec_vals[idx]))

        # Create a slicer from the model directly, letting it sample its own states.
        slicer_from_model = LynxSamplerSlicer(self.model.sample_parameters(num_samples=10), self.model)
        self.assertIsInstance(slicer_from_model, LynxSamplerSlicer)
        self.assertTrue(slicer_from_model.slice_points["lynx_model"] is self.model)
        self.assertEqual(slicer_from_model.slice_points["lynx_params"].shape, (10,))

    def test_lynx_detection_metric(self):
        """Test that metric name is set appropriately automatically
        and when explicitly passed.
        """
        sampled_state = self.model.sample_parameters(num_samples=1)
        slice_point = {"lynx_model": self.model, "lynx_params": sampled_state}

        # At SNR > 5, we expect at least one detection.
        metric = LynxDetectionMetric(threshold=5.0)
        result = metric.run(self.single_query_data, slice_point=slice_point)
        self.assertIsInstance(result, (int, np.integer))
        self.assertGreater(result, 0)

        # At SNR > 0, we expect everything to be a detection.
        metric = LynxDetectionMetric(threshold=0.0)
        result = metric.run(self.single_query_data, slice_point=slice_point)
        self.assertIsInstance(result, (int, np.integer))
        self.assertEqual(result, 4)

    def test_lynx_detection_metric_with_precomputed_lightcurve(self):
        """Test that the LynxDetectionMetric works correctly when the 'lightcurve' data is precomputed and
        provided in the slice_point. We should return the pandas DataFrame directly."""
        # Create a fake model and sample a single state.
        sampled_state = self.model.sample_parameters(num_samples=1)

        # Precompute the light curve data as a Pandas DataFrame. Pandas is installed
        # if LightCurveLynx is available.
        import pandas as pd

        lightcurve = pd.DataFrame(
            {
                "mjd": self.single_query_data["observationStartMJD"],
                "flux": np.array([1000.0, 10.0, 1000.0, 500.0]),
                "fluxerr": np.array([10.0, 20.0, 20.0, 20.0]),
            }
        )
        slice_point = {"lynx_model": self.model, "lynx_params": sampled_state, "lightcurve": lightcurve}

        # At SNR > 1, we should get exactly 3 detections.
        metric = LynxDetectionMetric(threshold=1.0)
        result = metric.run(self.single_query_data, slice_point=slice_point)
        self.assertIsInstance(result, (int, np.integer))
        self.assertEqual(result, 3)


class TestLightCurveLynxInterfaceNoInstall(unittest.TestCase):
    """Test cases for the LightCurveLynx interface when the library is not installed."""

    def test_lynx_detection_metric_with_precomputed_lightcurve_no_install(self):
        """Test that the LynxDetectionMetric works correctly when the 'lightcurve' data is precomputed and
        provided in the slice_point. We should return the pandas DataFrame directly."""
        single_query_data = {
            "observationStartMJD": np.array([0.0, 1.0, 2.0, 3.0]),
            "fieldRA": np.array([15.0, 15.0, 15.0, 15.0]),
            "fieldDec": np.array([-10.0, -10.0, -10.0, -10.0]),
            "filter": np.array(["r", "g", "r", "r"]),
            "seeingFwhmEff": [1.12] * 4,
            "skyBrightness": [20.0] * 4,
            "visitExposureTime": [29.2] * 4,
            "numExposures": [1] * 4,
            "airmass": [1.0] * 4,
        }

        # Precompute the light curve data as a dictionary.
        lightcurve_data = {
            "mjd": single_query_data["observationStartMJD"],
            "flux": np.array([1000.0, 10.0, 1000.0, 500.0]),
            "fluxerr": np.array([10.0, 20.0, 20.0, 20.0]),
        }

        # At SNR > 1, we should get exactly 3 detections.
        metric = LynxDetectionMetric(threshold=1.0)
        result = metric.run(single_query_data, slice_point={"lightcurve": lightcurve_data})
        self.assertIsInstance(result, (int, np.integer))
        self.assertEqual(result, 3)


if __name__ == "__main__":
    unittest.main()
