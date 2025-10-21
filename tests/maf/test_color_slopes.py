import unittest

import numpy as np

import rubin_sim.maf.metrics as metrics


class TestSimpleMetrics(unittest.TestCase):
    def test_color_slope(self):
        names = ["night", "observationStartMJD", "filter", "fiveSigmaDepth"]
        types = [int, float, "<U1", float]

        data = np.zeros(4, dtype=list(zip(names, types)))

        # same filter, same night
        data["observationStartMJD"] = np.array([0, 0.25, 0.5, 0.55]) / 24
        data["filter"] = ["r", "r", "r", "r"]
        data["fiveSigmaDepth"] = 25.0

        csm = metrics.ColorSlopeMetric(color_length=1.0, slope_length=3.0)

        cs2n = metrics.ColorSlope2NightMetric(color_length=1.0, slope_length=15.0)
        assert csm.run(data) == 0
        assert cs2n.run(data) == 0

        # diff filter, same night
        # has color, but no slope
        data["observationStartMJD"] = np.array([0, 0.25, 0.5, 0.55]) / 24
        data["filter"] = ["r", "g", "r", "r"]

        assert csm.run(data) == 0
        assert cs2n.run(data) == 0

        # diff filter, same night
        # slope on 1st night, not second
        data["observationStartMJD"] = np.array([0, 0.25, 0.5, 3.55]) / 24
        data["filter"] = ["r", "g", "r", "r"]

        assert csm.run(data) == 1
        assert cs2n.run(data) == 0

        # diff filter, diff night
        # slope on 2nd night, not first
        data["night"] = [0, 0, 0, 1]
        data["observationStartMJD"] = np.array([0, 0.25, 0.5, 25]) / 24
        data["filter"] = ["r", "g", "r", "r"]

        assert csm.run(data) == 0
        assert cs2n.run(data) == 1

        # diff filter, diff night
        # slope on both nights
        data["night"] = [0, 0, 0, 1]
        data["observationStartMJD"] = np.array([0, 0.25, 3.5, 25]) / 24
        data["filter"] = ["r", "g", "r", "r"]

        assert csm.run(data) == 1
        assert cs2n.run(data) == 1

        # diff filter, diff night
        # slope on both nights, but no color
        data["night"] = [0, 0, 0, 1]
        data["observationStartMJD"] = np.array([0, 5.25, 3.5, 25]) / 24
        data["filter"] = ["r", "g", "r", "r"]

        assert csm.run(data) == 0
        assert cs2n.run(data) == 0


if __name__ == "__main__":
    unittest.main()
