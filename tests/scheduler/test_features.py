import os
import unittest

import numpy as np

import rubin_sim.scheduler.features as features
from rubin_sim.data import get_data_dir
from rubin_sim.scheduler.model_observatory import ModelObservatory
from rubin_sim.scheduler.utils import empty_observation


class TestFeatures(unittest.TestCase):
    def test_pair_in_night(self):
        pin = features.PairInNight(gap_min=25.0, gap_max=45.0)
        self.assertEqual(np.max(pin.feature), 0.0)

        indx = np.array([1000])

        delta = 30.0 / 60.0 / 24.0

        # Add 1st observation, feature should still be zero
        obs = empty_observation()
        obs["filter"] = "r"
        obs["mjd"] = 59000.0
        pin.add_observation(obs, indx=indx)
        self.assertEqual(np.max(pin.feature), 0.0)

        # Add 2nd observation
        obs["mjd"] += delta
        pin.add_observation(obs, indx=indx)
        self.assertEqual(np.max(pin.feature), 1.0)

        obs["mjd"] += delta
        pin.add_observation(obs, indx=indx)
        self.assertEqual(np.max(pin.feature), 2.0)

    def test_conditions(self):
        observatory = ModelObservatory(seeing_db=os.path.join(get_data_dir(), "tests", "seeing.db"))
        conditions = observatory.return_conditions()
        self.assertIsInstance(repr(conditions), str)
        self.assertIsInstance(str(conditions), str)

        step_days = 1.0

        # Number of sidereal days in a standard day
        sidereal_hours_per_day = 24 * (24.0 / 23.9344696)
        initial_lmst = float(conditions.lmst)
        conditions.mjd = conditions.mjd + step_days
        new_lmst = float(conditions.lmst)
        self.assertAlmostEqual(
            new_lmst,
            (initial_lmst + step_days * sidereal_hours_per_day) % 24,
        )

        # Test that the string representation works
        _ = conditions.__str__()

        # Check that naked conditions work
        conditions_naked = features.Conditions()
        _ = conditions_naked.__str__()

    def test_note_last_observed(self):
        note_last_observed = features.NoteLastObserved(note="test")

        observation = empty_observation()
        observation["mjd"] = 59000.0

        note_last_observed.add_observation(observation=observation)

        assert note_last_observed.feature is None

        observation["note"] = "foo"

        note_last_observed.add_observation(observation=observation)
        assert note_last_observed.feature is None

        observation["note"] = "test"

        note_last_observed.add_observation(observation=observation)
        assert note_last_observed.feature == observation["mjd"]

    def test_note_last_observed_with_filter(self):
        note_last_observed = features.NoteLastObserved(
            note="test",
            filtername="r",
        )

        observation = empty_observation()
        observation["mjd"] = 59000.0

        note_last_observed.add_observation(observation=observation)

        assert note_last_observed.feature is None

        observation["note"] = "foo"

        note_last_observed.add_observation(observation=observation)
        assert note_last_observed.feature is None

        observation["note"] = "test"
        observation["filter"] = "g"

        note_last_observed.add_observation(observation=observation)
        assert note_last_observed.feature is None

        observation["note"] = "test"
        observation["filter"] = "r"

        note_last_observed.add_observation(observation=observation)
        assert note_last_observed.feature == observation["mjd"]


if __name__ == "__main__":
    unittest.main()
