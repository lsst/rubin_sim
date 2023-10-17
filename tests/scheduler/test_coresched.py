import os
import unittest

import numpy as np
import pandas as pd

import rubin_sim.scheduler.basis_functions as basis_functions
import rubin_sim.scheduler.surveys as surveys
from rubin_sim.data import get_data_dir
from rubin_sim.scheduler.model_observatory import ModelObservatory
from rubin_sim.scheduler.schedulers import CoreScheduler
from rubin_sim.scheduler.utils import generate_all_sky


class TestCoreSched(unittest.TestCase):
    def testsched(self):
        nside = 32
        # Just set up a very simple target map, dec limited, one filter
        sky_dict = generate_all_sky(nside, mask=-1)
        target_map = np.where(
            ((sky_dict["map"] >= 0) & (sky_dict["dec"] < 2) & (sky_dict["dec"] > -65)), 1, 0
        )

        bfs = []
        bfs.append(basis_functions.M5DiffBasisFunction(nside=nside))
        bfs.append(basis_functions.TargetMapBasisFunction(target_map=target_map, norm_factor=1))
        weights = np.array([1.0, 1])
        survey = surveys.GreedySurvey(bfs, weights)
        scheduler = CoreScheduler([survey])

        observatory = ModelObservatory(
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
        _ = scheduler.get_basis_functions([0, 0])
        _ = scheduler.get_healpix_maps([0, 0])

        # Check survey access methods
        reward_df = scheduler.make_reward_df(observatory.return_conditions())
        self.assertIsInstance(reward_df, pd.DataFrame)
        reward_df = scheduler.make_reward_df(observatory.return_conditions(), accum=False)
        self.assertIsInstance(reward_df, pd.DataFrame)

        obs = scheduler.request_observation()
        surveys_df = scheduler.surveys_df(0)
        self.assertIsInstance(surveys_df, pd.DataFrame)

        # Test we can record basis function values when requested
        recording_scheduler = CoreScheduler([survey], keep_rewards=True)
        recording_scheduler.update_conditions(observatory.return_conditions())
        obs = recording_scheduler.request_observation()
        self.assertIsInstance(recording_scheduler.queue_reward_df, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
