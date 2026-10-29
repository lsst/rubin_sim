import numpy as np
import unittest

try:
    import lightcurvelynx
except ImportError:
    lightcurvelynx = None

from rubin_sim.maf.slicers import LynxSamplerSlicer


def _create_basic_opssim_data(num_points):
    """Create a data set that has all the rows we need for simulation."""
    band_choices = ["u", "g", "r"]
    data = {
        "observationStartMJD": np.arange(60000, 60000 + num_points).tolist(),
        "fieldRA": np.full(num_points, 200.5).tolist(),
        "fieldDec": np.full(num_points, -10.0).tolist(),
        "rotSkyPos": np.full(num_points, 0.0).tolist(),
        "band": [band_choices[i % len(band_choices)] for i in range(num_points)],
        "visitExposureTime": np.full(num_points, 30.0).tolist(),
        "numExposures": np.full(num_points, 2).tolist(),
        "seeingFwhmEff": np.full(num_points, 0.8).tolist(),
        "skyBrightness": np.full(num_points, 21.0).tolist(),
        "airmass": np.full(num_points, 1.2).tolist(),
    }
    return data


@unittest.skipIf(lightcurvelynx is None, "lightcurvelynx not installed")
class TestLynxSampler(unittest.TestCase):

    def setUp(self):
        from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
        from lightcurvelynx.models.basic_models import SinWaveModel

        # Create a basic LightCurveLynx model for simulation.
        self.model = SinWaveModel(
            brightness=2000.0,
            amplitude=200.0,
            frequency=NumpyRandomFunc("uniform", low=0.01, high=0.1),
            t0=NumpyRandomFunc("uniform", low=0.0, high=10.0),
            ra=NumpyRandomFunc("normal", loc=200.5, scale=0.01),
            dec=NumpyRandomFunc("normal", loc=-50.0, scale=0.01),
            node_label="sin_wave_model",
        )

        # Create a basic data set with everything we need to run a simulation.
        self.data = _create_basic_opssim_data(num_points=10)

    def test_iteration_on_states(self):
        """Test iteration goes through the expected samples given the states."""
        states = self.model.sample_parameters(num_samples=20)

        # Set up a data set and the slicer.
        slicer = LynxSamplerSlicer(samples=states, model=self.model)
        slicer.setup_slicer(self.data)

        # Check that we get the pointer to the same model and
        # the correct state at each sample.
        for idx, sample in enumerate(slicer):
            assert sample["slice_point"]["lynx_model"] is self.model
            
            given_state_i = states.extract_single_sample(idx)
            iter_state_i = sample["slice_point"]["lynx_params"]
            assert given_state_i == iter_state_i

    def test_iteration_on_num_samples(self):
        """Test iteration goes through the expected samples given the number of samples."""
        from lightcurvelynx.graph_state import GraphState

        # Set up a data set and the slicer.
        slicer = LynxSamplerSlicer(samples=15, model=self.model)
        slicer.setup_slicer(self.data)

        # Check that we get the expected number of samples.
        assert len(list(slicer)) == 15
        for sample in slicer:
            assert sample["slice_point"]["lynx_model"] is self.model
            assert isinstance(sample["slice_point"]["lynx_params"], GraphState)
