import os
import unittest

import numpy as np

from rubin_sim.data import get_data_dir
from rubin_sim.scheduler import sim_runner
from rubin_sim.scheduler.example import example_scheduler
from rubin_sim.scheduler.model_observatory import ModelObservatory
from rubin_sim.scheduler.utils import restore_scheduler, run_info_table, season_calc
from rubin_sim.utils import survey_start_mjd


class TestUtils(unittest.TestCase):
    @unittest.skipUnless(
        os.path.isfile(os.path.join(get_data_dir(), "scheduler/dust_maps/dust_nside_32.npz")),
        "Test data not available.",
    )
    def test_nside(self):
        """Test the example scheduler can be set to different nsides."""
        mjd_start = survey_start_mjd()
        _ = example_scheduler(mjd_start=mjd_start, nside=64)
        _ = example_scheduler(mjd_start=mjd_start, nside=8)

    @unittest.skipUnless(
        os.path.isfile(os.path.join(get_data_dir(), "scheduler/dust_maps/dust_nside_32.npz")),
        "Test data not available.",
    )
    def test_restore(self):
        """Test we can restore a scheduler properly"""
        # MJD set so it's in test data range
        mjd_start = survey_start_mjd()
        n_visit_limit = 3000

        scheduler = example_scheduler(mjd_start=mjd_start)

        mo = ModelObservatory(mjd_start=mjd_start)
        # Never load too many nights of sky
        mo.sky_model.load_length = 10.0
        mo, scheduler, observations = sim_runner(
            mo,
            scheduler,
            survey_length=30.0,
            verbose=False,
            filename=None,
            n_visit_limit=n_visit_limit,
        )

        # Won't be exact if we restart in the middle of a blob sequence
        # since the queue isn't reconstructed.
        # Also, any scripted observations that get generated
        # during the night (e.g., long gaps observations) will get lost,
        # so need to restart on a new night to ensure identical results.

        nd = np.zeros(observations.size)
        nd[1:] = np.diff(observations["night"])

        break_indx = np.min(np.where((observations["ID"] >= n_visit_limit / 2.0) & (nd != 0))[0])
        new_n_limit = n_visit_limit - break_indx

        new_mo = ModelObservatory(mjd_start=mjd_start)
        # Never load too much sky
        new_mo.sky_model.load_length = 10.0
        new_sched = example_scheduler(mjd_start=mjd_start)

        # Restore some of the observations
        new_sched, new_mo = restore_scheduler(break_indx - 1, new_sched, new_mo, observations, fast=False)

        # Simulate ahead and confirm that it behaves the same
        # as running straight through
        new_mo, new_sched, new_obs = sim_runner(
            new_mo,
            new_sched,
            survey_length=20.0,
            verbose=False,
            filename=None,
            n_visit_limit=new_n_limit,
        )

        # Check that observations taken after restart match those from before
        # Jenkins can be bad at comparing things, so if it thinks
        # they aren't the same, check column-by-column to double check
        if not np.all(new_obs == observations[break_indx:]):
            names = new_obs.dtype.names
            for name in names:
                # If it's a string
                if new_obs[name].dtype == "<U40":
                    assert np.all(new_obs[name] == observations[break_indx:][name])
                # Otherwise should be number-like
                else:
                    assert np.allclose(new_obs[name], observations[break_indx:][name])
        # Didn't need to go by column, the observations after restart
        # match the ones that were taken all at once.
        else:
            assert np.all(new_obs == observations[break_indx:])

        # And again, but this time using the fast array restore
        new_mo = ModelObservatory(mjd_start=mjd_start)
        new_mo.sky_model.load_length = 10.0
        new_sched = example_scheduler(mjd_start=mjd_start)
        new_sched, new_mo = restore_scheduler(break_indx - 1, new_sched, new_mo, observations, fast=True)
        # Simulate ahead and confirm that it behaves the same as
        # running straight through
        new_mo, new_sched, new_obs_fast = sim_runner(
            new_mo,
            new_sched,
            survey_length=20.0,
            verbose=False,
            filename=None,
            n_visit_limit=new_n_limit,
        )

        # Check that observations taken after restart match those from before
        # Jenkins can be bad at comparing things, so if it thinks
        # they aren't the same, check column-by-column to double check
        if not np.all(new_obs_fast == observations[break_indx:]):
            names = new_obs_fast.dtype.names
            for name in names:
                # If it's a string
                if new_obs_fast[name].dtype == "<U40":
                    assert np.all(new_obs_fast[name] == observations[break_indx:][name])
                # Otherwise should be number-like
                else:
                    assert np.allclose(new_obs_fast[name], observations[break_indx:][name])
        # Didn't need to go by column, the observations after restart
        # match the ones that were taken all at once.
        else:
            assert np.all(new_obs_fast == observations[break_indx:])

    def test_season(self):
        """
        Test that the season utils work as intended
        """
        night = 365.25 * 3.5
        plain = season_calc(night)
        assert plain == 3

        mod2 = season_calc(night, modulo=2)
        assert mod2 == 1

        mod3 = season_calc(night, modulo=3)
        assert mod3 == 0

        mod3 = season_calc(night, modulo=3, max_season=2)
        assert mod3 == -1

        mod3 = season_calc(night, modulo=3, max_season=2, offset=-365.25 * 2)
        assert mod3 == 1

        mod3 = season_calc(night, modulo=3, max_season=2, offset=-365.25 * 10)
        assert mod3 == -1

        mod3 = season_calc(night, modulo=3, offset=-365.25 * 10)
        assert mod3 == -1

    def test_run_info_table(self):
        """Test run_info_table gets information"""
        observatory = ModelObservatory(
            nside=8,
            mjd_start=survey_start_mjd(),
            seeing_db=os.path.join(get_data_dir(), "tests", "seeing.db"),
        )
        version_info = run_info_table(observatory)
        # Make a minimal set of keys that ought to be in the info table
        # Update these if the value they're stored as changes
        # (either from run_info_table or observatory.info)
        need_keys = [
            "rubin_sim.__version__",
            "hostname",
            "Date, ymd",
            "site_models",
            "skybrightness_pre",
        ]
        have_keys = list(version_info["Parameter"])
        for k in need_keys:
            self.assertTrue(k in have_keys)


if __name__ == "__main__":
    unittest.main()
