import os
import unittest

import numpy as np
import pandas as pd
from rubin_scheduler.data import get_data_dir

import rubin_sim.maf as maf


class TestSNmetrics(unittest.TestCase):
    def setUp(self):
        # Make sure we can read SN lightcurve and reference info
        if not os.path.isdir(os.path.join(get_data_dir(), "maf")):
            self.skipTest("Skipping SN tests because running unit tests without full rubin_scheduler.data.")
        # Read test point data
        testfile = os.path.join(get_data_dir(), "tests", "test_simData.hdf")
        if not os.path.isfile(testfile):
            raise FileExistsError("%s not found" % testfile)
        self.simdata = {}
        with pd.HDFStore(testfile, mode="r") as f:
            keys = f.keys()
            for k in keys:
                newkey = k.lstrip("/")
                self.simdata[newkey] = f.get(k).to_records()

    def test_nsn(self):
        # Set up a mock slicerPoint
        nside = 64
        slice_point = {"nside": nside, "ebv": 0.0}

        # Set up the metric
        metric = maf.SNNSNMetric(
            season=[-1],
            n_aft=8,
            n_bef=3,
            add_dust=False,
            hard_dust_cut=0.25,
            zmin=0.1,
            zmax=0.5,
            z_step=0.03,
            daymax_step=3,
            snr_min=1,
            zlim_coeff=0.95,
            sigma_c=0.04,
            bands="grizy",
            gamma_name="gamma_WFD.hdf5",
            verbose=False,
        )

        # Expected keys and results
        expected = {}
        # Muddle with m5 - very shallow
        expected["one_season_shallow"] = metric.badval
        # Very few visits
        expected["sparse_pointing"] = metric.badval
        # Very dense pointing
        expected["dense_pointing"] = np.rec.fromrecords([(5.158324, 0.276509)], names=["nSN", "zlim"])
        # These are of the same point on the sky, with/without DD fields -
        # they should be the same as the metric rejects DD visits internally
        # ---XXX except now the metric does
        # not reject visits based on notes, so not sure why these still pass

        expected["one_season_noDD"] = np.rec.fromrecords([(0.870031, 0.289703)], names=["nSN", "zlim"])
        expected["one_season_wDD"] = np.rec.fromrecords([(0.870031, 0.289703)], names=["nSN", "zlim"])
        # Here we muddle with the visit exposure time (without changing m5)
        expected["one_season_singleExp_20"] = np.rec.fromrecords([(0.879159, 0.29006)], names=["nSN", "zlim"])
        # Here we muddle with the number of visits per exposure,
        # without changing m5 or visitExpTime
        expected["one_season_singleExp_30"] = np.rec.fromrecords(
            [(0.870031, 0.289703)], names=["nSN", "zlim"]
        )

        verbose = True

        for k in self.simdata:
            res = metric.run(self.simdata[k], slice_point=slice_point)
            if verbose:
                print("")
                print(f"pointing test {k} with {len(self.simdata[k])} visits")
                try:
                    print(f"expected results {expected[k]}")
                except KeyError:
                    print(f"no expected results for {k}")
                print(f"calculated results {res}")
                print("")
            # self.assertEqual(expected[k], res)

        # And run for DD observations
        metric = maf.SNNSNMetric(
            season=[-1],
            n_aft=10,
            n_bef=4,
            add_dust=False,
            hard_dust_cut=0.25,
            zmin=0.1,
            zmax=0.5,
            z_step=0.03,
            daymax_step=3,
            snr_min=1,
            zlim_coeff=0.95,
            sigma_c=0.04,
            bands="grizy",
            gamma_name="gamma_DDF.hdf5",
            coadd_night=True,
            verbose=False,
        )
        for k in ["one_season_wDD"]:
            res = metric.run(self.simdata[k], slice_point=slice_point)
            if verbose:
                print("")
                print(f"pointing test {k} for {len(self.simdata[k])} DD-only visits")

                print(f"calculated results {res}")
                print("")


if __name__ == "__main__":
    unittest.main()
