import unittest

import numpy as np

import rubin_sim.utils as utils
from rubin_sim.scheduler.model_observatory import ModelObservatory


class KindaClouds:
    """Dummy class that always sets the clouds level to a unique float"""

    def __call__(self, mjd):
        return 0.23659


class ArbSeeing:
    """Dummy class to always return a specific seeing value"""

    def __call__(self, mjd):
        fwhm_500 = 1.756978
        return fwhm_500


class TestModelObservatory(unittest.TestCase):
    def test_replace(self):
        """test that we can replace default downtimes, seeing, and clouds"""

        mjd_start = utils.survey_start_mjd()
        mo_default = ModelObservatory(mjd_start=mjd_start)
        # Never load too many nights of sky
        mo_default.sky_model.load_length = 10.0
        cond_default = mo_default.return_conditions()

        # Define new downtimes
        downtimes = np.zeros(2, dtype=list(zip(["start", "end"], [float, float])))
        downtimes["start"] = np.array([1, 10]) + mjd_start
        downtimes["end"] = np.array([2, 11]) + mjd_start

        seeing_data = ArbSeeing()
        cloud_data = KindaClouds()

        mo_new = ModelObservatory(
            mjd_start=mjd_start,
            seeing_data=seeing_data,
            cloud_data=cloud_data,
            downtimes=downtimes,
        )
        # Never load too many nights of sky
        mo_new.sky_model.load_length = 10.0
        cond_new = mo_new.return_conditions()

        # Make sure the internal downtimes are different
        assert ~np.array_equal(mo_default.downtimes, mo_new.downtimes)

        # Make sure seeing is not the same
        diff = cond_default.fwhm_eff["r"] - cond_new.fwhm_eff["r"]
        assert np.nanmin(np.abs(diff)) > 0

        # Make sure cloudyness level is not the same
        assert cond_default.bulk_cloud != cond_new.bulk_cloud


if __name__ == "__main__":
    unittest.main()
