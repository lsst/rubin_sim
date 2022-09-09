import os
import numpy as np
import pandas as pd
import unittest
from rubin_sim.data import get_data_dir
from rubin_sim.scheduler.schedulers import Core_scheduler
import rubin_sim.scheduler.basis_functions as basis_functions
import rubin_sim.scheduler.surveys as surveys
from rubin_sim.scheduler.utils import standard_goals
from rubin_sim.scheduler.modelObservatory import Model_observatory


class TestCoreSched(unittest.TestCase):
    def testsched(self):
        target_map = standard_goals()["r"]
        nside = 32

        bfs = []
        bfs.append(basis_functions.M5_diff_basis_function(nside=nside))
        bfs.append(basis_functions.Target_map_basis_function(target_map=target_map))
        weights = np.array([1.0, 1])
        survey = surveys.Greedy_survey(bfs, weights)
        scheduler = Core_scheduler([survey])

        observatory = Model_observatory(
            seeing_db=os.path.join(get_data_dir(), "tests", "seeing.db"),
        )

        # Check that we can update conditions
        scheduler.update_conditions(observatory.return_conditions())

        # Check that we can get an observation out
        obs = scheduler.request_observation()
        assert obs is not None

        # Check that we can flush the Queue
        scheduler.flush_queue()
        assert len(scheduler.queue) == 0

        # Check that we can add an observation
        scheduler.add_observation(obs)

        # Check dunder methods
        self.assertIsInstance(repr(scheduler), str)
        self.assertIsInstance(str(scheduler), str)

        # Check access methods
        these_basis_functions = scheduler.get_basis_functions([0, 0])
        healpix_maps = scheduler.get_healpix_maps([0, 0])

        # Check survey access methods
        reward_df = scheduler.make_reward_df(observatory.return_conditions())
        self.assertIsInstance(reward_df, pd.DataFrame)

        obs = scheduler.request_observation()
        surveys_df = scheduler.surveys_df(0)
        self.assertIsInstance(surveys_df, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
