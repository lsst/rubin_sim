import os
import unittest

import pandas as pd

import rubin_sim.scheduler.basis_functions as basis_functions
import rubin_sim.scheduler.surveys as surveys
from rubin_sim.data import get_data_dir
from rubin_sim.scheduler.model_observatory import ModelObservatory


class TestSurveys(unittest.TestCase):
    def test_field_survey(self):
        nside = 32

        bfs = []
        bfs.append(basis_functions.M5DiffBasisFunction(nside=nside))
        survey = surveys.FieldSurvey(bfs, RA=90.0, dec=-30.0, reward_value=1)

        observatory = ModelObservatory(
            seeing_db=os.path.join(get_data_dir(), "tests", "seeing.db"),
        )

        # Check dunder methods
        self.assertIsInstance(repr(survey), str)
        self.assertIsInstance(str(survey), str)

        # Check survey access methods
        conditions = observatory.return_conditions()
        reward = survey.calc_reward_function(conditions)
        self.assertIsInstance(reward, float)
        reward_df = survey.reward_changes(conditions)
        reward_df = survey.make_reward_df(conditions)
        self.assertIsInstance(reward_df, pd.DataFrame)
        reward_df = survey.make_reward_df(conditions, accum=False)


if __name__ == "__main__":
    unittest.main()
