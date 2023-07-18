# pylint: disable=too-many-arguments
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name

# imports
import unittest

import numpy as np
import pandas as pd

from rubin_sim.maf.slicers import BlockIntervalSlicer, TimeIntervalSlicer, VisitIntervalSlicer

# constants

SIMDATA = {
    "observationStartMJD": [
        60000.100,
        60000.101,
        60000.102,
        60000.103,
        60000.202,
        60000.203,
    ],
    "visitTime": [86, 86, 86, 86, 86, 86],
    "note": [
        "greedy",
        "blob, yy, a",
        "blob, yy, a",
        "DD:EDFS, b",
        "DD:EDFS, b",
        "DD:EDFS, b",
    ],
}

# interface functions

# classes


class TestTimeIntervalSlicer(unittest.TestCase):
    interval_seconds = 60

    def setUp(self):
        self.slicer = TimeIntervalSlicer(interval_seconds=self.interval_seconds)

    def test_setup_slicer(self):
        self.slicer.setup_slicer(SIMDATA)
        self.assertEqual(self.slicer.nslice, 24 * 60)
        slice_points = self.slicer.get_slice_points()
        nonempty_slices = [item[0] for item in self.slicer.sim_idxs.items() if len(item[1]) > 0]
        self.assertTrue(
            np.allclose(
                np.array(SIMDATA["observationStartMJD"]),
                slice_points["mjd"][nonempty_slices],
                atol=self.interval_seconds / (24 * 60 * 60.0),
            )
        )
        self.assertTrue(np.all(slice_points["duration"] == self.interval_seconds))


class TestBlockIntervalSlicer(unittest.TestCase):
    def setUp(self):
        self.slicer = BlockIntervalSlicer()

    def test_setup_slicer(self):
        self.slicer.setup_slicer(SIMDATA)
        self.assertEqual(self.slicer.nslice, 4)
        slice_points = self.slicer.get_slice_points()
        sim_idxs = self.slicer.sim_idxs
        visits = pd.DataFrame(SIMDATA)
        for sid in slice_points["sid"]:
            these_visits = visits.iloc[sim_idxs[sid]]
            self.assertTrue(np.all(these_visits["note"] == these_visits["note"].values[0]))


class TestVisitIntervalSlicer(unittest.TestCase):
    def setUp(self):
        self.slicer = VisitIntervalSlicer()

    def test_setup_slicer(self):
        self.slicer.setup_slicer(SIMDATA)
        self.assertEqual(self.slicer.nslice, len(SIMDATA["observationStartMJD"]))
        slice_points = self.slicer.get_slice_points()
        self.assertIn("sid", slice_points)
        self.assertIn("mjd", slice_points)
        self.assertIn("duration", slice_points)
        self.assertTrue(np.all(slice_points["duration"] == SIMDATA["visitTime"]))


# internal functions & classes

run_tests_now = __name__ == "__main__"
if run_tests_now:
    unittest.main()
