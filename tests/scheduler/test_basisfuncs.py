import unittest

import numpy as np

import rubin_sim.scheduler.basis_functions as basis_functions
from rubin_sim.scheduler.features import Conditions
from rubin_sim.scheduler.utils import empty_observation


class TestBasis(unittest.TestCase):
    def test_visit_repeat_basis_function(self):
        bf = basis_functions.VisitRepeatBasisFunction()

        indx = np.array([1000])

        # 30 minute step
        delta = 30.0 / 60.0 / 24.0

        # Add 1st observation, should still be zero
        obs = empty_observation()
        obs["filter"] = "r"
        obs["mjd"] = 59000.0
        conditions = Conditions()
        conditions.mjd = np.max(obs["mjd"])
        bf.add_observation(obs, indx=indx)
        self.assertEqual(np.max(bf(conditions)), 0.0)

        # Advance time so now we want a pair
        conditions.mjd += delta
        self.assertEqual(np.max(bf(conditions)), 1.0)

        # Now complete the pair and it should go back to zero
        bf.add_observation(obs, indx=indx)

        conditions.mjd += delta
        self.assertEqual(np.max(bf(conditions)), 0.0)

    def test_label(self):
        bf = basis_functions.VisitRepeatBasisFunction()
        self.assertIsInstance(bf.label(), str)

        bf = basis_functions.SlewtimeBasisFunction(nside=16)
        self.assertIsInstance(bf.label(), str)

    def test_visit_gap(self):
        visit_gap = basis_functions.VisitGap(note="test")

        conditions = Conditions()
        conditions.mjd = 59000.0

        # default is feasible
        assert visit_gap.check_feasibility(conditions=conditions)

        observation = empty_observation()
        observation["filter"] = "r"
        observation["note"] = "foo"
        observation["mjd"] = 59000.0

        visit_gap.add_observation(observation=observation)

        # observation with the wrong note
        assert visit_gap.check_feasibility(conditions=conditions)

        observation["note"] = "test"
        visit_gap.add_observation(observation=observation)

        # now observation with the correct note
        assert not visit_gap.check_feasibility(conditions=conditions)

        # check it becomes feasible again once enough time has passed
        conditions.mjd += 2.0 * visit_gap.gap

        assert visit_gap.check_feasibility(conditions=conditions)

    def test_visit_gap_with_filter(self):
        visit_gap = basis_functions.VisitGap(note="test", filter_names=["g"])

        conditions = Conditions()
        conditions.mjd = 59000.0

        # default is feasible
        assert visit_gap.check_feasibility(conditions=conditions)

        observation = empty_observation()
        observation["filter"] = "r"
        observation["note"] = "foo"
        observation["mjd"] = 59000.0

        visit_gap.add_observation(observation=observation)

        # observation with the wrong note
        assert visit_gap.check_feasibility(conditions=conditions)

        observation["note"] = "test"
        visit_gap.add_observation(observation=observation)

        # observation with the wrong filter
        assert visit_gap.check_feasibility(conditions=conditions)

        observation["filter"] = "g"
        visit_gap.add_observation(observation=observation)

        # observation with the correct note and filter
        assert not visit_gap.check_feasibility(conditions=conditions)

        # check it becomes feasible again once enough time has passed
        conditions.mjd += 2.0 * visit_gap.gap

        assert visit_gap.check_feasibility(conditions=conditions)

    def test_visit_gap_with_multiple_filters(self):
        visit_gap = basis_functions.VisitGap(note="test", filter_names=["g", "i"])

        conditions = Conditions()
        conditions.mjd = 59000.0

        # default is feasible
        assert visit_gap.check_feasibility(conditions=conditions)

        observation = empty_observation()
        observation["filter"] = "r"
        observation["note"] = "foo"
        observation["mjd"] = 59000.0

        visit_gap.add_observation(observation=observation)

        # observation with the wrong note
        assert visit_gap.check_feasibility(conditions=conditions)

        observation["note"] = "test"
        visit_gap.add_observation(observation=observation)

        # observation with the wrong filter
        assert visit_gap.check_feasibility(conditions=conditions)

        observation["filter"] = "g"
        observation["mjd"] += 1e-3
        visit_gap.add_observation(observation=observation)

        # observation with the correct note but only one filter
        assert visit_gap.check_feasibility(conditions=conditions)

        observation["filter"] = "i"
        observation["mjd"] += 1e-3
        visit_gap.add_observation(observation=observation)

        # observation with the correct note and both filters
        assert not visit_gap.check_feasibility(conditions=conditions)

        # make sure it is still not feasible after only the g observation gap
        # has passed
        conditions.mjd += visit_gap.gap + 1.1e-3

        # observation with the correct note and both filters
        assert not visit_gap.check_feasibility(conditions=conditions)

        # make sure it is feasible after both gaps have passed
        conditions.mjd += 1e-3

        assert visit_gap.check_feasibility(conditions=conditions)

    def test_sun_alt(self):
        sunaltbf = basis_functions.SunAltHighLimitBasisFunction(alt_limit=-15)
        conditions = Conditions()
        conditions.sun_alt = np.radians(-20)
        assert ~sunaltbf.check_feasibility(conditions)
        conditions.sun_alt = np.radians(-10)
        assert sunaltbf.check_feasibility(conditions)


if __name__ == "__main__":
    unittest.main()
