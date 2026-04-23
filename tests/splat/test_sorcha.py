import unittest

import rubin_sim.splat as splat
from rubin_sim.data import get_baseline, get_data_dir
import os


class TestSimple(unittest.TestCase):

    @unittest.skip("skipping until sorcha >1.1.1 is on conda.")
    def test_sorcha_wrapper(self):
        """ """
        baseline_file = get_baseline()

        # Same as default, but limit to night < 30
        query = (
            "SELECT observationId, observationStartMJD as observationStartMJD_TAI, "
            "visitTime, visitExposureTime, filter, seeingFwhmGeom as seeingFwhmGeom_arcsec, "
            "seeingFwhmEff as seeingFwhmEff_arcsec, fiveSigmaDepth as fieldFiveSigmaDepth_mag , "
            "fieldRA as fieldRA_deg, fieldDec as fieldDec_deg, rotSkyPos as "
            "fieldRotSkyPos_deg FROM observations where night < 30 order by observationId"
        )

        dd = get_data_dir()

        orbin_file = os.path.join(dd, "sorcha", "occ_rmax20_5k_kep.csv")
        colors_file = os.path.join(dd, "sorcha", "occ_rmax20_5k_param.csv")

        observations, stats = splat.solar_system.sorcha_wrapper(
            baseline_file, orbin_file, colors_file, query=query
        )

        assert observations.size > 0
        assert stats.size > 0


if __name__ == "__main__":

    unittest.main()
