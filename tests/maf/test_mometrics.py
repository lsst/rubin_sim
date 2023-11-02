import unittest

import numpy as np
import pandas as pd

import rubin_sim.maf.metrics as metrics


class TestMoMetrics1(unittest.TestCase):
    def setUp(self):
        # Set up some sso_obs data to test the metrics on.
        # Note that sso_obs is a numpy recarray.
        # The expected set of columns in sso_obs is:
        # cols = ['observationStartMJD', 'night', 'fieldRA', 'fieldDec',
        #        'rotSkyPos', 'filter',
        #        'visitExposureTime', 'seeingFwhmGeom', 'fiveSigmaDepth',
        #        'solarElong',
        #        'delta', 'ra', 'dec', 'magV', 'time', 'dradt', 'ddecdt',
        #        'phase', 'solarelon',
        #        'velocity', 'magFilter', 'dmagColor', 'dmagTrail',
        #        'dmagDetect']
        # And stackers will often add
        # addCols = ['appMag', 'magLimit', 'snr', 'vis']

        # Test metrics using sso_obs for a particular object.
        times = np.array(
            [0.1, 0.2, 0.3, 1.1, 1.3, 5.1, 7.1, 7.2, 7.3, 10.1, 10.2, 10.3, 13.1, 13.5],
            dtype="float",
        )
        sso_obs = np.recarray(
            [len(times)],
            dtype=(
                [
                    ("time", "<f8"),
                    ("ra", "<f8"),
                    ("dec", "<f8"),
                    ("appMag", "<f8"),
                    ("observationStartMJD", "<f8"),
                    ("night", "<f8"),
                    ("magLimit", "<f8"),
                    ("SNR", "<f8"),
                    ("vis", "<f8"),
                ]
            ),
        )

        sso_obs["time"] = times
        sso_obs["observationStartMJD"] = times
        sso_obs["night"] = np.floor(times)
        sso_obs["ra"] = np.arange(len(times))
        sso_obs["dec"] = np.arange(len(times))
        sso_obs["appMag"] = np.zeros(len(times), dtype="float") + 24.0
        sso_obs["magLimit"] = np.zeros(len(times), dtype="float") + 25.0
        sso_obs["SNR"] = np.zeros(len(times), dtype="float") + 5.0
        sso_obs["vis"] = np.zeros(len(times), dtype="float") + 1
        sso_obs["vis"][0:5] = 0
        self.sso_obs = sso_obs
        self.orb = None
        self.hval = 8.0

    def testn_obs_metric(self):
        n_obs_metric = metrics.NObsMetric(snr_limit=5)
        n_obs = n_obs_metric.run(self.sso_obs, self.orb, self.hval)
        self.assertEqual(n_obs, len(self.sso_obs["time"]))
        n_obs_metric = metrics.NObsMetric(snr_limit=10)
        n_obs = n_obs_metric.run(self.sso_obs, self.orb, self.hval)
        self.assertEqual(n_obs, 0)
        n_obs_metric = metrics.NObsMetric(snr_limit=None)
        n_obs = n_obs_metric.run(self.sso_obs, self.orb, self.hval)
        self.assertEqual(n_obs, len(self.sso_obs["time"]) - 5)

    def testn_obs_no_singles_metric(self):
        n_obs_no_singles_metric = metrics.NObsNoSinglesMetric(snr_limit=5)
        n_obs = n_obs_no_singles_metric.run(self.sso_obs, self.orb, self.hval)
        self.assertEqual(n_obs, len(self.sso_obs["time"]) - 1)

    def test_n_nights_metric(self):
        n_nights_metric = metrics.NNightsMetric(snr_limit=5)
        nnights = n_nights_metric.run(self.sso_obs, self.orb, self.hval)
        self.assertEqual(nnights, len(np.unique(self.sso_obs["night"])))
        n_nights_metric = metrics.NNightsMetric(snr_limit=None)
        nnights = n_nights_metric.run(self.sso_obs, self.orb, self.hval)
        self.assertEqual(nnights, len(np.unique(self.sso_obs["night"])) - 2)

    def test_arc_metric(self):
        arc_metric = metrics.ObsArcMetric(snr_limit=5)
        arc = arc_metric.run(self.sso_obs, self.orb, self.hval)
        self.assertEqual(
            arc,
            self.sso_obs["observationStartMJD"][-1] - self.sso_obs["observationStartMJD"][0],
        )
        arc_metric = metrics.ObsArcMetric(snr_limit=None)
        arc = arc_metric.run(self.sso_obs, self.orb, self.hval)
        self.assertEqual(
            arc,
            self.sso_obs["observationStartMJD"][-1] - self.sso_obs["observationStartMJD"][5],
        )

    def test_activity_over_period_metric(self):
        # cometary orbit format ok
        orb = np.recarray(
            1,
            dtype=(
                [
                    ("objId", (str, 20)),
                    ("q", float),
                    ("e", float),
                    ("inc", float),
                    ("Omega", float),
                    ("argPeri", float),
                    ("tPeri", float),
                    ("epoch", float),
                    ("H", float),
                    ("g", float),
                ]
            ),
        )

        orb["objId"] = "NESC00001HYj"
        orb["q"] = 1.00052
        orb["e"] = 0.028514
        orb["inc"] = 0.502477
        orb["Omega"] = 50.989131
        orb["argPeri"] = 55.091685
        orb["tPeri"] = 61046.627194 - 59850
        orb["epoch"] = 60973.799216 - 59850
        orb["H"] = 35.526041
        orb["g"] = 0.15
        o = pd.DataFrame(orb)
        activity_period_metric = metrics.ActivityOverPeriodMetric(bin_size=360, snr_limit=5)
        activity = activity_period_metric.run(self.sso_obs, o.iloc[0], self.hval)
        self.assertEqual(activity, 1.0)
        activity_period_metric = metrics.ActivityOverPeriodMetric(bin_size=720, snr_limit=5)
        activity = activity_period_metric.run(self.sso_obs, o.iloc[0], self.hval)
        self.assertEqual(activity, 1.0)
        activity_period_metric = metrics.ActivityOverPeriodMetric(bin_size=10, snr_limit=5)
        activity = activity_period_metric.run(self.sso_obs, o.iloc[0], self.hval)
        self.assertLess(activity, 0.03)
        # different type of orbit - currently should fail quietly
        orb = np.recarray(
            1,
            dtype=(
                [
                    ("objId", (str, 20)),
                    ("a", float),
                    ("e", float),
                    ("inc", float),
                    ("Omega", float),
                    ("argPeri", float),
                    ("meanAnomaly", float),
                    ("epoch", float),
                    ("H", float),
                    ("g", float),
                ]
            ),
        )
        orb["objId"] = "NESC00001HYj"
        orb["a"] = 1.029886
        orb["e"] = 0.028514
        orb["inc"] = 0.502477
        orb["Omega"] = 50.989131
        orb["argPeri"] = 55.091685
        orb["meanAnomaly"] = 291.321814
        orb["epoch"] = 60973.799216 - 59850
        orb["H"] = 35.526041
        orb["g"] = 0.15
        o = pd.DataFrame(orb)
        activity_period_metric = metrics.ActivityOverPeriodMetric(bin_size=360, snr_limit=5)
        activity = activity_period_metric.run(self.sso_obs, o.iloc[0], self.hval)
        self.assertEqual(activity, 1.0)
        activity_period_metric = metrics.ActivityOverPeriodMetric(bin_size=180, snr_limit=5)
        activity = activity_period_metric.run(self.sso_obs, o.iloc[0], self.hval)
        self.assertEqual(activity, 0.5)

    def tearDown(self):
        del self.sso_obs
        del self.orb
        del self.hval


class TestDiscoveryMetrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(61331)
        # Test metrics using sso_obs for a particular object.
        times = np.array(
            [0.1, 0.2, 0.9, 1.1, 1.3, 5.1, 7.1, 7.2, 7.5, 10.1, 10.2, 13.1, 13.5],
            dtype="float",
        )
        sso_obs = np.recarray(
            [len(times)],
            dtype=(
                [
                    ("time", "<f8"),
                    ("ra", "<f8"),
                    ("dec", "<f8"),
                    ("ec_lon", "<f8"),
                    ("ec_lat", "<f8"),
                    ("solar_elong", "<f8"),
                    ("appMag", "<f8"),
                    ("observationStartMJD", "<f8"),
                    ("night", "<f8"),
                    ("magLimit", "<f8"),
                    ("velocity", "<f8"),
                    ("SNR", "<f8"),
                    ("vis", "<f8"),
                    ("magFilter", "<f8"),
                    ("fiveSigmaDepth", "<f8"),
                    ("seeingFwhmGeom", "<f8"),
                    ("visitExposureTime", "<f8"),
                    ("dmagDetect", "<f8"),
                ]
            ),
        )

        sso_obs["time"] = times
        sso_obs["observationStartMJD"] = times
        sso_obs["night"] = np.floor(times)
        sso_obs["ra"] = np.arange(len(times))
        sso_obs["dec"] = np.arange(len(times)) + 5
        sso_obs["ec_lon"] = sso_obs["ra"] + 10
        sso_obs["ec_lat"] = sso_obs["dec"] + 20
        sso_obs["solar_elong"] = sso_obs["ra"] + 30
        sso_obs["appMag"] = np.zeros(len(times), dtype="float") + 24.0
        sso_obs["magFilter"] = np.zeros(len(times), dtype="float") + 24.0
        sso_obs["fiveSigmaDepth"] = np.zeros(len(times), dtype="float") + 25.0
        sso_obs["dmagDetect"] = np.zeros(len(times), dtype="float")
        sso_obs["magLimit"] = np.zeros(len(times), dtype="float") + 25.0
        sso_obs["SNR"] = np.zeros(len(times), dtype="float") + 5.0
        sso_obs["vis"] = np.zeros(len(times), dtype="float") + 1
        sso_obs["vis"][0:5] = 0
        sso_obs["velocity"] = rng.rand(len(times))
        sso_obs["seeingFwhmGeom"] = np.ones(len(times), "float")
        sso_obs["visitExposureTime"] = np.ones(len(times), "float") * 24.0
        self.sso_obs = sso_obs
        self.orb = np.recarray([len(times)], dtype=([("H", "<f8")]))
        self.orb["H"] = np.zeros(len(times), dtype="float") + 8
        self.hval = 8

    def test_discovery_metric(self):
        disc_metric = metrics.DiscoveryMetric(
            n_obs_per_night=2,
            t_min=0.0,
            t_max=0.3,
            n_nights_per_window=3,
            t_window=9,
            snr_limit=5,
        )
        metric_value = disc_metric.run(self.sso_obs, self.orb, self.hval)
        child = metrics.DiscoveryNObsMetric(disc_metric)
        n_obs = child.run(self.sso_obs, self.orb, self.hval, metric_value)
        self.assertEqual(n_obs, 8)
        # child = metrics.DiscoveryNObsMetric(disc_metric, i=1)
        # n_obs = child.run(self.sso_obs, self.orb, self.hval, metric_value)
        # self.assertEqual(n_obs, 7)
        child = metrics.DiscoveryNChancesMetric(disc_metric)
        nchances = child.run(self.sso_obs, self.orb, self.hval, metric_value)
        self.assertEqual(nchances, 2)
        child = metrics.DiscoveryTimeMetric(disc_metric)
        time = child.run(self.sso_obs, self.orb, self.hval, metric_value)
        self.assertEqual(time, self.sso_obs["observationStartMJD"][0])
        # child = metrics.DiscoveryTimeMetric(disc_metric, i=1)
        # time = child.run(self.sso_obs, self.orb, self.hval, metric_value)
        # self.assertEqual(time, self.sso_obs["observationStartMJD"][3])
        child = metrics.DiscoveryRadecMetric(disc_metric)
        ra, dec = child.run(self.sso_obs, self.orb, self.hval, metric_value)
        self.assertEqual(ra, 0)
        self.assertEqual(dec, 5)
        child = metrics.DiscoveryEclonlatMetric(disc_metric)
        lon, lat, solar_elong = child.run(self.sso_obs, self.orb, self.hval, metric_value)
        self.assertEqual(lon, 10)
        self.assertEqual(lat, 25)

        disc_metric3 = metrics.MagicDiscoveryMetric(n_obs=5, t_window=2, snr_limit=5)
        magic = disc_metric3.run(self.sso_obs, self.orb, self.hval)
        self.assertEqual(magic, 1)
        disc_metric3 = metrics.MagicDiscoveryMetric(n_obs=3, t_window=1, snr_limit=5)
        magic = disc_metric3.run(self.sso_obs, self.orb, self.hval)
        self.assertEqual(magic, 2)
        disc_metric3 = metrics.MagicDiscoveryMetric(n_obs=4, t_window=4, snr_limit=5)
        magic = disc_metric3.run(self.sso_obs, self.orb, self.hval)
        self.assertEqual(magic, 4)

    def test_high_velocity_metric(self):
        rng = np.random.RandomState(8123)
        vel_metric = metrics.HighVelocityMetric(psf_factor=1.0, snr_limit=5)
        metric_value = vel_metric.run(self.sso_obs, self.orb, self.hval)
        self.assertEqual(metric_value, 0)
        self.sso_obs["velocity"][0:2] = 1.5
        metric_value = vel_metric.run(self.sso_obs, self.orb, self.hval)
        self.assertEqual(metric_value, 2)
        vel_metric = metrics.HighVelocityMetric(psf_factor=2.0, snr_limit=5)
        metric_value = vel_metric.run(self.sso_obs, self.orb, self.hval)
        self.assertEqual(metric_value, 0)
        self.sso_obs["velocity"][0:2] = rng.rand(1)

    def test_high_velocity_nights_metric(self):
        vel_metric = metrics.HighVelocityNightsMetric(psf_factor=1.0, n_obs_per_night=1, snr_limit=5)
        metric_value = vel_metric.run(self.sso_obs, self.orb, self.hval)
        self.assertEqual(metric_value, 0)
        self.sso_obs["velocity"][0:2] = 1.5
        metric_value = vel_metric.run(self.sso_obs, self.orb, self.hval)
        self.assertEqual(metric_value, self.sso_obs["observationStartMJD"][0])
        self.sso_obs["velocity"][0:2] = np.random.rand(1)


class TestKnownObjectMetrics(unittest.TestCase):
    def setUp(self):
        self.t1 = 53371
        self.t2 = 57023
        self.t3 = 59580
        times = np.arange(self.t1 - 365 * 2, self.t3 + 365 * 3, 1)
        cols = ["MJD(UTC)", "RA", "Dec", "magV", "Elongation", "appMagV"]
        dtype = []
        for c in cols:
            dtype.append((c, "<f8"))
        sso_obs = np.recarray([len(times)], dtype=dtype)

        sso_obs["MJD(UTC)"] = times
        sso_obs["RA"] = np.arange(len(times))
        sso_obs["Dec"] = np.arange(len(times))
        sso_obs["magV"] = np.zeros(len(times), dtype="float") + 20.0
        sso_obs["Elongation"] = np.zeros(len(times), dtype=float) + 180.0
        self.hval = 0.0
        sso_obs["appMagV"] = sso_obs["magV"] + self.hval
        self.orb = None
        self.sso_obs = sso_obs

    def test_known_objects_metric(self):
        known_object_metric = metrics.KnownObjectsMetric(
            t_switch1=self.t1,
            eff1=1.0,
            v_mag_thresh1=20.5,
            t_switch2=self.t2,
            eff2=1.0,
            v_mag_thresh2=20.5,
            t_switch3=self.t3,
            eff3=1.0,
            v_mag_thresh3=20.5,
            eff4=1.0,
            v_mag_thresh4=22,
        )
        m_val = known_object_metric.run(self.sso_obs, self.orb, self.hval)
        self.assertEqual(m_val, self.sso_obs["MJD(UTC)"].min())
        known_object_metric = metrics.KnownObjectsMetric(
            t_switch1=self.t1,
            eff1=1.0,
            v_mag_thresh1=15.0,
            t_switch2=self.t2,
            eff2=1.0,
            v_mag_thresh2=20.5,
            t_switch3=self.t3,
            eff3=1.0,
            v_mag_thresh3=20.5,
            eff4=1.0,
            v_mag_thresh4=22,
        )
        m_val = known_object_metric.run(self.sso_obs, self.orb, self.hval)
        self.assertEqual(m_val, self.t1)
        known_object_metric = metrics.KnownObjectsMetric(
            t_switch1=self.t1,
            eff1=0.0,
            v_mag_thresh1=20.5,
            t_switch2=self.t2,
            eff2=1.0,
            v_mag_thresh2=20.5,
            t_switch3=self.t3,
            eff3=1.0,
            v_mag_thresh3=20.5,
            eff4=1.0,
            v_mag_thresh4=22,
        )
        m_val = known_object_metric.run(self.sso_obs, self.orb, self.hval)
        self.assertEqual(m_val, self.t1)
        known_object_metric = metrics.KnownObjectsMetric(
            t_switch1=self.t1,
            eff1=1.0,
            v_mag_thresh1=10,
            t_switch2=self.t2,
            eff2=1.0,
            v_mag_thresh2=10.5,
            t_switch3=self.t3,
            eff3=1.0,
            v_mag_thresh3=20.5,
            eff4=1.0,
            v_mag_thresh4=22,
        )
        m_val = known_object_metric.run(self.sso_obs, self.orb, self.hval)
        self.assertEqual(m_val, self.t2)
        known_object_metric = metrics.KnownObjectsMetric(
            t_switch1=self.t1,
            eff1=1.0,
            v_mag_thresh1=10,
            t_switch2=self.t2,
            eff2=1.0,
            v_mag_thresh2=10.5,
            t_switch3=self.t3,
            eff3=1.0,
            v_mag_thresh3=10.5,
            eff4=1.0,
            v_mag_thresh4=22,
        )
        m_val = known_object_metric.run(self.sso_obs, self.orb, self.hval)
        self.assertEqual(m_val, self.t3)
        known_object_metric = metrics.KnownObjectsMetric(
            t_switch1=self.t1,
            eff1=1.0,
            v_mag_thresh1=10,
            t_switch2=self.t2,
            eff2=1.0,
            v_mag_thresh2=10.5,
            t_switch3=self.t3,
            eff3=1.0,
            v_mag_thresh3=10.5,
            eff4=1.0,
            v_mag_thresh4=10.5,
        )
        m_val = known_object_metric.run(self.sso_obs, self.orb, self.hval)
        self.assertEqual(m_val, known_object_metric.badval)


if __name__ == "__main__":
    unittest.main()
