import os
import unittest
import warnings

import numpy as np
from rubin_scheduler.data import get_data_dir

from rubin_sim.maf.metrics import SNCadenceMetric, SNSLMetric, SNSNRMetric
from rubin_sim.maf.utils.sn_utils import Lims, ReferenceData

m5_ref = dict(zip("ugrizy", [23.60, 24.83, 24.38, 23.92, 23.35, 22.44]))


def observations_band(day0=59000, daymin=59000, cadence=3.0, season_length=140.0, band="r"):
    # Define fake data
    names = [
        "observationStartMJD",
        "fieldRA",
        "fieldDec",
        "fiveSigmaDepth",
        "visitExposureTime",
        "numExposures",
        "visitTime",
        "season",
        "seeingFwhmEff",
        "seeingFwhmGeom",
        "pixRA",
        "pixDec",
    ]
    types = ["f8"] * len(names)
    names += ["night", "healpixID"]
    types += ["i2", "i2"]
    names += ["filter"]
    types += ["O"]

    daylast = daymin + season_length
    cadence = cadence
    dayobs = np.arange(daymin, daylast, cadence)
    npts = len(dayobs)
    data = np.zeros(npts, dtype=list(zip(names, types)))
    data["observationStartMJD"] = dayobs
    data["night"] = np.floor(data["observationStartMJD"] - day0 + 1)
    data["fiveSigmaDepth"] = m5_ref[band]
    data["visitExposureTime"] = 30.0
    data["numExposures"] = 1
    data["visitTime"] = 34
    data["filter"] = band
    data["seeingFwhmEff"] = 0.8
    data["seeingFwhmGeom"] = 0.8
    data["healpixID"] = 10
    data["pixRA"] = 0.0
    data["pixDec"] = 0.0
    return data


def observations_season(mjdmin=59000, cadence=3.0):
    bands = "grizy"
    nvisits = dict(zip(bands, [10, 20, 20, 26, 20]))
    rat = 34.0 / 3600.0 / 24.0
    shift_visits = {}
    shift_visits["g"] = 0
    shift_visits["r"] = rat * nvisits["g"]
    shift_visits["i"] = rat * nvisits["r"]
    shift_visits["z"] = rat * nvisits["i"]
    shift_visits["y"] = rat * nvisits["z"]

    # get data
    data = None
    season_length = 180
    shift = 30.0 / (3600.0 * 24)
    for band in bands:
        mjd = mjdmin + shift_visits[band]
        for i in range(nvisits[band]):
            mjd += shift
            dat = observations_band(daymin=mjd, season_length=season_length, cadence=cadence, band=band)
            if data is None:
                data = dat
            else:
                data = np.concatenate((data, dat))

    return data


def fake_data(band, season=1):
    # Define fake data
    names = [
        "observationStartMJD",
        "fieldRA",
        "fieldDec",
        "fiveSigmaDepth",
        "visitExposureTime",
        "numExposures",
        "visitTime",
        "season",
        "seeingFwhmEff",
        "seeingFwhmGeom",
        "airmass",
        "sky",
        "moonPhase",
        "pixRA",
        "pixDec",
    ]

    types = ["f8"] * len(names)
    names += ["night"]
    types += ["i2"]
    names += ["healpixID"]
    types += ["i2"]
    names += ["filter"]
    types += ["O"]

    dayobs = [
        59948.31957176,
        59959.2821412,
        59970.26134259,
        59973.25978009,
        59976.26383102,
        59988.20670139,
        59991.18412037,
        60004.1853588,
        60032.08975694,
        60045.11981481,
        60047.98747685,
        60060.02083333,
        60071.986875,
        60075.96452546,
    ]
    day0 = np.min(dayobs)
    npts = len(dayobs)
    data = np.zeros(npts, dtype=list(zip(names, types)))
    data["observationStartMJD"] = dayobs
    data["night"] = np.floor(data["observationStartMJD"] - day0 + 1)
    data["fiveSigmaDepth"] = m5_ref[band]
    data["visitExposureTime"] = 15.0
    data["numExposures"] = 2
    data["visitTime"] = 2.0 * 15.0
    data["season"] = season
    data["filter"] = band
    data["seeingFwhmEff"] = 0.0
    data["seeingFwhmGeom"] = 0.0
    data["airmass"] = 1.2
    data["sky"] = 20.0
    data["moonPhase"] = 0.5
    data["pixRA"] = 0.0
    data["pixDec"] = 0.0
    data["healpixID"] = 1

    return data


class TestSNmetrics(unittest.TestCase):
    def setUp(self):
        if not os.path.isdir(os.path.join(get_data_dir(), "maf")):
            self.skipTest("Skipping SN tests because running unit tests without full rubin_sim.data.")

    @unittest.skip("The SNCadenceMetric is not used")
    def test_sn_cadence_metric(self):
        """Test the SN cadence metric"""

        # Load up the files from sims_maf_contrib if possible
        sims_maf_contrib_dir = os.path.join(get_data_dir(), "maf")
        if sims_maf_contrib_dir is not None:
            # Load required SN info to run the metric
            band = "r"
            SNR = dict(zip("griz", [30.0, 40.0, 30.0, 20.0]))  # SNR for WFD
            mag_range = [21.0, 25.5]  # WFD mag range
            dt_range = [0.5, 30.0]  # WFD dt range
            li_files = [os.path.join(sims_maf_contrib_dir, "Li_SNCosmo_-2.0_0.2.npy")]
            mag_to_flux_files = [os.path.join(sims_maf_contrib_dir, "Mag_to_Flux_SNCosmo.npy")]
            lim_sn = Lims(
                li_files,
                mag_to_flux_files,
                band,
                SNR[band],
                mag_range=mag_range,
                dt_range=dt_range,
            )

            # Define fake data
            names = [
                "observationStartMJD",
                "fieldRA",
                "fieldDec",
                "fiveSigmaDepth",
                "visitExposureTime",
                "numExposures",
                "visitTime",
            ]
            types = ["f8"] * len(names)
            names += ["night"]
            types += ["i2"]
            names += ["filter"]
            types += ["O"]

            day0 = 59000
            daylast = day0 + 250
            cadence = 5
            dayobs = np.arange(day0, daylast, cadence)
            npts = len(dayobs)
            data = np.zeros(npts, dtype=list(zip(names, types)))
            data["observationStartMJD"] = dayobs
            data["night"] = np.floor(data["observationStartMJD"] - day0)
            data["fiveSigmaDepth"] = m5_ref[band]
            data["visitExposureTime"] = 15.0
            data["numExposures"] = 2
            data["visitTime"] = 2.0 * 15.0
            data["filter"] = band

            # Run the metric with these fake data
            slice_point = {"nside": 64, "ebv": 0}
            metric = SNCadenceMetric(lim_sn=lim_sn, coadd=False)
            result = metric.run(data, slice_point)

            # And the result should be...
            result_ref = 0.3743514

            assert np.abs(result - result_ref) < 1.0e-5

        else:
            warnings.warn("skipping SN test because no rubin_sim.data set")

    @unittest.skip("This metric is not used")
    def test_snsnr_metric(self):
        """Test the SN SNR metric"""

        sims_maf_contrib_dir = os.path.join(get_data_dir(), "maf")
        if sims_maf_contrib_dir is not None:
            # Load required SN info to run the metric
            band = "r"
            z = 0.3
            season = 1.0
            li_files = [os.path.join(sims_maf_contrib_dir, "Li_SNCosmo_-2.0_0.2.npy")]
            mag_to_flux_files = [os.path.join(sims_maf_contrib_dir, "Mag_to_Flux_SNCosmo.npy")]

            names_ref = ["SNCosmo"]
            coadd = False

            lim_sn = ReferenceData(li_files, mag_to_flux_files, band, z)

            # Define fake data
            names = [
                "observationStartMJD",
                "fieldRA",
                "fieldDec",
                "fiveSigmaDepth",
                "visitExposureTime",
                "numExposures",
                "visitTime",
                "season",
            ]
            types = ["f8"] * len(names)
            names += ["night"]
            types += ["i2"]
            names += ["filter"]
            types += ["O"]

            dayobs = [
                59948.31957176,
                59959.2821412,
                59970.26134259,
                59973.25978009,
                59976.26383102,
                59988.20670139,
                59991.18412037,
                60004.1853588,
                60032.08975694,
                60045.11981481,
                60047.98747685,
                60060.02083333,
                60071.986875,
                60075.96452546,
            ]
            day0 = np.min(dayobs)
            npts = len(dayobs)
            data = np.zeros(npts, dtype=list(zip(names, types)))

            data["observationStartMJD"] = dayobs
            data["night"] = np.floor(data["observationStartMJD"] - day0)
            data["fiveSigmaDepth"] = m5_ref[band]
            data["visitExposureTime"] = 15.0
            data["numExposures"] = 2
            data["visitTime"] = 2.0 * 15.0
            data["season"] = season
            data["filter"] = band

            # Run the metric with these fake data
            slice_point = {"nside": 64, "ebv": 0.0}
            metric = SNSNRMetric(lim_sn=lim_sn, coadd=coadd, names_ref=names_ref, season=season, z=z)

            result = metric.run(data, slice_point)

            # And the result should be...
            result_ref = 0.4830508474576271

            assert np.abs(result - result_ref) < 1.0e-5
        else:
            warnings.warn("skipping SN test because no rubin_sim.data set")

    def test_snsl_metric(self):
        """Test the SN SL metric"""

        # load some fake data
        data = None
        bands = "griz"
        cadence = dict(zip(bands, [2, 1, 2, 1]))
        for band in bands:
            for i in range(cadence[band]):
                fakes = fake_data(band)
                if data is None:
                    data = fakes
                else:
                    data = np.concatenate((data, fakes))

        # metric instance

        metric = SNSLMetric(
            nfilters_min=4,
            min_season_obs=5,
            m5mins={"u": 22.7, "g": 24.1, "r": 23.7, "i": 23.1, "z": 22.2, "y": 21.4},
        )

        # run the metric
        n_sl = metric.run(data, slice_point={"nside": 64, "ra": 0.0, "ebv": 0.0})

        # and the result should be
        # Changing the reference value because we have new coadd and mag limits
        # Change again to switch to per-season calc rather than average
        # Change again to reflect updated calculation
        # (due to counting year = 365.25 days, not 360)
        n_sl_ref = 1.4569195613987307e-06
        assert np.abs(n_sl - n_sl_ref) < 1.0e-8


if __name__ == "__main__":
    unittest.main()
