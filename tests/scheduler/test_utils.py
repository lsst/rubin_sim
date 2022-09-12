import unittest
import os
from rubin_sim.data import get_data_dir
from rubin_sim.scheduler.utils import season_calc
from rubin_sim.scheduler.model_observatory import Model_observatory
from rubin_sim.scheduler.utils import run_info_table


class TestFeatures(unittest.TestCase):
    def testSeason(self):
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
        observatory = Model_observatory(
            nside=8,
            mjd_start=59853.5,
            seeing_db=os.path.join(get_data_dir(), "tests", "seeing.db"),
        )
        versionInfo = run_info_table(observatory)
        # Make a minimal set of keys that probably ought to be in the info table
        # Update these if the value they're stored as changes (either from run_info_table or observatory.info)
        need_keys = [
            "rubin_sim.__version__",
            "hostname",
            "Date, ymd",
            "site_models",
            "skybrightness_pre",
        ]
        have_keys = list(versionInfo["Parameter"])
        for k in need_keys:
            self.assertTrue(k in have_keys)


if __name__ == "__main__":
    unittest.main()
