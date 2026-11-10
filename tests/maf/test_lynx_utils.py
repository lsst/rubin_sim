import numpy as np
import os
import tempfile
import unittest

try:
    import lightcurvelynx
except ImportError:
    lightcurvelynx = None

from rubin_sim.maf.metric_bundles.metric_bundle import MetricBundle
from rubin_sim.maf.metric_bundles.metric_bundle_group import MetricBundleGroup
from rubin_sim.maf.metrics.base_lynx_metrics import LynxObsCountMetric
from rubin_sim.maf.slicers.lynx_slicer import LynxSamplerSlicer
from rubin_sim.maf.slicers.lynx_slicer import LynxResultsSlicer


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


def _create_basic_lc_data(num_samples):
    """Create a basic light curve data set."""
    lc_data = {
        "mjd": np.arange(num_samples).tolist(),
        "flux": np.full(num_samples, 1010.0).tolist(),
        "fluxerr": np.full(num_samples, 10.0).tolist(),
        "flux_perfect": np.full(num_samples, 1000.0).tolist(),
    }
    return lc_data


class TestLynxResults(unittest.TestCase):
    """Test cases for slicers and metrics that read LightCurveLynx results."""

    def test_metric_eval(self):
        """Test that we can evaluate a metric that uses LightCurveLynx results."""
        # Create a slice point with the model and a single sampled state.
        slice_point = {"sid": 0}
        data_slice = [[_create_basic_lc_data(num_samples=15)]]

        # The count should match the number of observations in the data slice.
        metric = LynxObsCountMetric(from_results=True)
        assert metric.run(data_slice, slice_point) == 15

    @unittest.skipIf(lightcurvelynx is None, "lightcurvelynx not installed")
    def test_lynx_results_end_to_end(self):
        """Do an end-to-end test of using precomputed LightCurveLynx results."""
        from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
        from lightcurvelynx.models.basic_models import SinWaveModel
        from lightcurvelynx.obstable.opsim import OpSim
        from lightcurvelynx.simulate import simulate_lightcurves
        from lightcurvelynx.survey_info import SurveyInfo

        # Create a test model for the simulation.
        model = SinWaveModel(
            brightness=2000.0,
            amplitude=200.0,
            frequency=NumpyRandomFunc("uniform", low=0.01, high=0.1),
            t0=NumpyRandomFunc("uniform", low=0.0, high=10.0),
            ra=NumpyRandomFunc("normal", loc=200.5, scale=0.01),
            dec=NumpyRandomFunc("normal", loc=-50.0, scale=0.01),
            node_label="sin_wave_model",
        )

        # Create a basic data set with everything we need to run a simulation.
        ops_data = _create_basic_opssim_data(num_points=10)
        opsim = OpSim(ops_data)

        # Simulate light curves using the specified model and survey information.
        lightcurves = simulate_lightcurves(
            model,  # The model to simulate (including effects).
            10,  # The number of light curves to simulate.
            SurveyInfo(opsim),  # The survey information.
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save the resuts to a temporary file.
            fname = os.path.join(tmpdir, "test_results.parquet")
            lightcurves.to_parquet(fname)

            # Create the metric bundle and the metric bundle group
            bundle = MetricBundle(
                metric=LynxObsCountMetric(from_results=True),
                slicer=LynxResultsSlicer(),
                constraint="",
            )
            bundle_group = MetricBundleGroup([bundle], fname, out_dir=tmpdir)

            # Run the metric bundle group
            bundle_group.run_all()

            # Check that the metric bundle has metric values.
            assert len(bundle.metric_values) == 10
            assert np.all(bundle.metric_values >= 0)


@unittest.skipIf(lightcurvelynx is None, "lightcurvelynx not installed")
class TestLynxSampler(unittest.TestCase):
    """Test cases for slicers and metrics that dynamically call LightCurveLynx."""

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

    def test_metric_eval(self):
        """Test that we can evaluate a metric that uses LightCurveLynx sampling."""
        # Create a slice point with the model and a single sampled state.
        slice_point = {
            "sid": 0,
            "lynx_model": self.model,
            "lynx_params": self.model.sample_parameters(num_samples=1),
        }

        # The count should match the number of times we observe this point in
        # the fake OpSim.
        metric = LynxObsCountMetric(from_results=False)
        assert metric.run(self.data, slice_point) == 10
