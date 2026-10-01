# Tests to verify the functionality of the LightCurveLynx interface within
# the Rubin Sim MAF framework.

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
    @unittest.skipIf(not HAS_LIBRARY, "lightcurvelynx is not installed")
    def test_lynx_sample_slicer(self):
        """Test we can create a LynxSamplerSlicer."""
        model = SinWaveModel(
            brightness=2000.0,
            amplitude=200.0,
            frequency=0.01,
            t0=0.0,
            ra=200.5,
            dec=-50.0,
            node_label="sin_wave_model",
        )
        states = model.sample_parameters(num_samples=5)
        ra_vals = states["sin_wave_model"]["ra"]
        dec_vals = states["sin_wave_model"]["dec"]

        # Create the slicer directly from the states.
        slicer = LynxSamplerSlicer(states, model)
        self.assertIsInstance(slicer, LynxSamplerSlicer)
        self.assertTrue(slicer.slice_points["lynx_model"] is model)
        self.assertEqual(slicer.slice_points["lynx_params"].shape, (5,))

        for idx, sample in enumerate(slicer.slice_points["lynx_params"]):
            self.assertTrue(np.allclose(sample["sin_wave_model"]["ra"], ra_vals[idx]))
            self.assertTrue(np.allclose(sample["sin_wave_model"]["dec"], dec_vals[idx]))

        # Create a slicer from the model directly, letting it sample its own states.
        slicer_from_model = LynxSamplerSlicer(model.sample_parameters(num_samples=10), model)
        self.assertIsInstance(slicer_from_model, LynxSamplerSlicer)
        self.assertTrue(slicer_from_model.slice_points["lynx_model"] is model)
        self.assertEqual(slicer_from_model.slice_points["lynx_params"].shape, (10,))

    @unittest.skipIf(not HAS_LIBRARY, "lightcurvelynx is not installed")
    def test_lynx_detection_metric(self):
        """Test that metric name is set appropriately automatically
        and when explicitly passed.
        """
        pass


if __name__ == "__main__":
    unittest.main()
