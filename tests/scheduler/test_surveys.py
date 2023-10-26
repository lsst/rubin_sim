import os
import unittest

import pandas as pd
import numpy as np
import healpy as hp

import rubin_sim.scheduler.basis_functions as basis_functions
import rubin_sim.scheduler.surveys as surveys
from rubin_sim.data import get_data_dir
from rubin_sim.scheduler.utils import set_default_nside
from rubin_sim.scheduler.model_observatory import ModelObservatory
from rubin_sim.scheduler.basis_functions import SimpleArrayBasisFunction


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

    def test_roi(self):
        random_seed = 6563
        infeasible_hpix = 123
        nside = set_default_nside()
        npix = hp.nside2npix(nside)
        rng = np.random.default_rng(seed=random_seed)
        num_bfs = 3
        bf_values = rng.random((num_bfs, npix))
        bf_values[:, infeasible_hpix] = -np.inf
        bfs = [SimpleArrayBasisFunction(values) for values in bf_values]

        observatory = ModelObservatory()
        conditions = observatory.return_conditions()

        # A few cases with an ROI with one valid healpix
        for i in range(3):
            hpix = rng.integers(npix)
            ra, decl = hp.pix2ang(nside, hpix, lonlat=True)
            survey = surveys.FieldSurvey(bfs, RA=ra, dec=decl, reward_value=1)
            reward_df = survey.make_reward_df(conditions)
            for value, max_basis_reward in zip(bf_values[:, hpix], reward_df["max_basis_reward"]):
                self.assertEqual(max_basis_reward, value)

        # One case with an ROI with only an infeasible healpix
        ra, decl = hp.pix2ang(nside, infeasible_hpix, lonlat=True)
        survey = surveys.FieldSurvey(bfs, RA=ra, dec=decl, reward_value=1)
        reward_df = survey.make_reward_df(conditions)
        for max_basis_reward in reward_df["max_basis_reward"]:
            self.assertEqual(max_basis_reward, -np.inf)

        for area in reward_df["basis_area"]:
            self.assertEqual(area, 0.0)

        for feasible in reward_df["feasible"]:
            self.assertFalse(feasible)

        # Make sure it still works as expected if no ROI is set
        weights = [1] * num_bfs
        survey = surveys.BaseMarkovSurvey(bfs, weights)
        for value, max_basis_reward in zip(bf_values.max(axis=1), reward_df["max_basis_reward"]):
            self.assertEqual(max_basis_reward, value)


if __name__ == "__main__":
    unittest.main()
