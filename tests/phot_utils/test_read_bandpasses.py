import os
import unittest

from rubin_scheduler.data import get_data_dir

from rubin_sim.phot_utils import Bandpass


class ReadBandPassTest(unittest.TestCase):
    """
    Tests for reading in bandpasses
    """

    def test_read(self):
        """
        Check that we can read things stored in the throughputs directory.
        """
        throughputs_dir = os.path.join(get_data_dir(), "throughputs")

        # select files to try and read
        files = [
            "2MASS/2MASS_Ks.dat",
            "WISE/WISE_w1.dat",
            "johnson/johnson_U.dat",
            "sdss/sdss_r.dat",
        ]
        for filename in files:
            bp = Bandpass()
            bp.read_throughput(os.path.join(throughputs_dir, filename))
