import unittest

import numpy as np

import rubin_sim.maf.metrics as metrics


class TestHourglassmetric(unittest.TestCase):
    def test_hourglass_metric(self):
        """Test the hourglass metric"""
        names = ["observationStartMJD", "night", "filter"]
        types = [float, float, str]
        npts = 50
        data = np.zeros(npts, dtype=list(zip(names, types)))
        day0 = 59000
        data["observationStartMJD"] = np.arange(0, 10, 0.2)[:npts] + day0
        data["night"] = np.floor(data["observationStartMJD"] - day0)
        data["filter"] = "r"
        data["filter"][-1] = "g"
        slice_point = [0]
        metric = metrics.HourglassMetric()
        result = metric.run(data, slice_point)
        pernight = result["pernight"]
        perfilter = result["perfilter"]

        # All the gaps are larger than 2 min.
        assert np.size(perfilter) == 2 * data.size
        # Check that the format is right at least
        assert len(pernight.dtype.names) == 9


if __name__ == "__main__":
    unittest.main()
