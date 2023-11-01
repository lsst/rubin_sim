import unittest

import numpy as np

import rubin_sim.maf.metrics as metrics


class TestStringCount(unittest.TestCase):
    def testsc(self):
        metric = metrics.StringCountMetric()
        data = np.array(["a", "a", "b", "c", "", "", ""])
        dt = np.dtype([("filter", np.str_, 1)])
        data.dtype = dt
        result = metric.run(data)
        # Check that the metricValue is correct
        expected_results = {"a": 2, "b": 1, "c": 1, "blank": 3}
        for key in expected_results:
            assert result[key] == expected_results[key]

        # Check that the reduce functions got made and return expected result
        for key in expected_results:
            assert metric.reduce_funcs[key](result) == expected_results[key]


if __name__ == "__main__":
    unittest.main()
