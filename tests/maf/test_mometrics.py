import numpy as np
import pandas as pd
import unittest
import rubin_sim.maf.metrics as metrics


class TestMoMetrics1(unittest.TestCase):
    def setUp(self):
        # Set up some ssoObs data to test the metrics on.
        # Note that ssoObs is a numpy recarray.
        # The expected set of columns in ssoObs is:
        # cols = ['observationStartMJD', 'night', 'fieldRA', 'fieldDec', 'rotSkyPos', 'filter',
        #        'visitExposureTime', 'seeingFwhmGeom', 'fiveSigmaDepth', 'solarElong',
        #        'delta', 'ra', 'dec', 'magV', 'time', 'dradt', 'ddecdt', 'phase', 'solarelon',
        #        'velocity', 'magFilter', 'dmagColor', 'dmagTrail', 'dmagDetect']
        # And stackers will often add
        # addCols = ['appMag', 'magLimit', 'snr', 'vis']

        # Test metrics using ssoObs for a particular object.
        times = np.array(
            [0.1, 0.2, 0.3, 1.1, 1.3, 5.1, 7.1, 7.2, 7.3, 10.1, 10.2, 10.3, 13.1, 13.5],
            dtype="float",
        )
        ssoObs = np.recarray(
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

        ssoObs["time"] = times
        ssoObs["observationStartMJD"] = times
        ssoObs["night"] = np.floor(times)
        ssoObs["ra"] = np.arange(len(times))
        ssoObs["dec"] = np.arange(len(times))
        ssoObs["appMag"] = np.zeros(len(times), dtype="float") + 24.0
        ssoObs["magLimit"] = np.zeros(len(times), dtype="float") + 25.0
        ssoObs["SNR"] = np.zeros(len(times), dtype="float") + 5.0
        ssoObs["vis"] = np.zeros(len(times), dtype="float") + 1
        ssoObs["vis"][0:5] = 0
        self.ssoObs = ssoObs
        self.orb = None
        self.Hval = 8.0

    def testNObsMetric(self):
        nObsMetric = metrics.NObsMetric(snr_limit=5)
        nObs = nObsMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(nObs, len(self.ssoObs["time"]))
        nObsMetric = metrics.NObsMetric(snr_limit=10)
        nObs = nObsMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(nObs, 0)
        nObsMetric = metrics.NObsMetric(snr_limit=None)
        nObs = nObsMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(nObs, len(self.ssoObs["time"]) - 5)

    def testNObsNoSinglesMetric(self):
        nObsNoSinglesMetric = metrics.NObsNoSinglesMetric(snr_limit=5)
        nObs = nObsNoSinglesMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(nObs, len(self.ssoObs["time"]) - 1)

    def testNNightsMetric(self):
        nNightsMetric = metrics.NNightsMetric(snr_limit=5)
        nnights = nNightsMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(nnights, len(np.unique(self.ssoObs["night"])))
        nNightsMetric = metrics.NNightsMetric(snr_limit=None)
        nnights = nNightsMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(nnights, len(np.unique(self.ssoObs["night"])) - 2)

    def testArcMetric(self):
        arcMetric = metrics.ObsArcMetric(snr_limit=5)
        arc = arcMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(
            arc,
            self.ssoObs["observationStartMJD"][-1]
            - self.ssoObs["observationStartMJD"][0],
        )
        arcMetric = metrics.ObsArcMetric(snr_limit=None)
        arc = arcMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(
            arc,
            self.ssoObs["observationStartMJD"][-1]
            - self.ssoObs["observationStartMJD"][5],
        )

    def testActivityOverPeriodMetric(self):
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
        activityPeriodMetric = metrics.ActivityOverPeriodMetric(
            binsize=360, snr_limit=5
        )
        activity = activityPeriodMetric.run(self.ssoObs, o.iloc[0], self.Hval)
        self.assertEqual(activity, 1.0)
        activityPeriodMetric = metrics.ActivityOverPeriodMetric(
            binsize=720, snr_limit=5
        )
        activity = activityPeriodMetric.run(self.ssoObs, o.iloc[0], self.Hval)
        self.assertEqual(activity, 1.0)
        activityPeriodMetric = metrics.ActivityOverPeriodMetric(binsize=10, snr_limit=5)
        activity = activityPeriodMetric.run(self.ssoObs, o.iloc[0], self.Hval)
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
        activityPeriodMetric = metrics.ActivityOverPeriodMetric(
            binsize=360, snr_limit=5
        )
        activity = activityPeriodMetric.run(self.ssoObs, o.iloc[0], self.Hval)
        self.assertEqual(activity, 1.0)
        activityPeriodMetric = metrics.ActivityOverPeriodMetric(
            binsize=180, snr_limit=5
        )
        activity = activityPeriodMetric.run(self.ssoObs, o.iloc[0], self.Hval)
        self.assertEqual(activity, 0.5)

    def tearDown(self):
        del self.ssoObs
        del self.orb
        del self.Hval


class TestDiscoveryMetrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(61331)
        # Test metrics using ssoObs for a particular object.
        times = np.array(
            [0.1, 0.2, 0.9, 1.1, 1.3, 5.1, 7.1, 7.2, 7.5, 10.1, 10.2, 13.1, 13.5],
            dtype="float",
        )
        ssoObs = np.recarray(
            [len(times)],
            dtype=(
                [
                    ("time", "<f8"),
                    ("ra", "<f8"),
                    ("dec", "<f8"),
                    ("ecLon", "<f8"),
                    ("ecLat", "<f8"),
                    ("solarElong", "<f8"),
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

        ssoObs["time"] = times
        ssoObs["observationStartMJD"] = times
        ssoObs["night"] = np.floor(times)
        ssoObs["ra"] = np.arange(len(times))
        ssoObs["dec"] = np.arange(len(times)) + 5
        ssoObs["ecLon"] = ssoObs["ra"] + 10
        ssoObs["ecLat"] = ssoObs["dec"] + 20
        ssoObs["solarElong"] = ssoObs["ra"] + 30
        ssoObs["appMag"] = np.zeros(len(times), dtype="float") + 24.0
        ssoObs["magFilter"] = np.zeros(len(times), dtype="float") + 24.0
        ssoObs["fiveSigmaDepth"] = np.zeros(len(times), dtype="float") + 25.0
        ssoObs["dmagDetect"] = np.zeros(len(times), dtype="float")
        ssoObs["magLimit"] = np.zeros(len(times), dtype="float") + 25.0
        ssoObs["SNR"] = np.zeros(len(times), dtype="float") + 5.0
        ssoObs["vis"] = np.zeros(len(times), dtype="float") + 1
        ssoObs["vis"][0:5] = 0
        ssoObs["velocity"] = rng.rand(len(times))
        ssoObs["seeingFwhmGeom"] = np.ones(len(times), "float")
        ssoObs["visitExposureTime"] = np.ones(len(times), "float") * 24.0
        self.ssoObs = ssoObs
        self.orb = np.recarray([len(times)], dtype=([("H", "<f8")]))
        self.orb["H"] = np.zeros(len(times), dtype="float") + 8
        self.Hval = 8

    def testDiscoveryMetric(self):
        discMetric = metrics.DiscoveryMetric(
            nObsPerNight=2,
            tMin=0.0,
            tMax=0.3,
            nNightsPerWindow=3,
            tWindow=9,
            snr_limit=5,
        )
        metricValue = discMetric.run(self.ssoObs, self.orb, self.Hval)
        child = metrics.Discovery_N_ObsMetric(discMetric, i=0)
        nobs = child.run(self.ssoObs, self.orb, self.Hval, metricValue)
        self.assertEqual(nobs, 8)
        child = metrics.Discovery_N_ObsMetric(discMetric, i=1)
        nobs = child.run(self.ssoObs, self.orb, self.Hval, metricValue)
        self.assertEqual(nobs, 7)
        child = metrics.Discovery_N_ChancesMetric(discMetric)
        nchances = child.run(self.ssoObs, self.orb, self.Hval, metricValue)
        self.assertEqual(nchances, 2)
        child = metrics.Discovery_TimeMetric(discMetric, i=0)
        time = child.run(self.ssoObs, self.orb, self.Hval, metricValue)
        self.assertEqual(time, self.ssoObs["observationStartMJD"][0])
        child = metrics.Discovery_TimeMetric(discMetric, i=1)
        time = child.run(self.ssoObs, self.orb, self.Hval, metricValue)
        self.assertEqual(time, self.ssoObs["observationStartMJD"][3])
        child = metrics.Discovery_RADecMetric(discMetric, i=0)
        ra, dec = child.run(self.ssoObs, self.orb, self.Hval, metricValue)
        self.assertEqual(ra, 0)
        self.assertEqual(dec, 5)
        child = metrics.Discovery_EcLonLatMetric(discMetric, i=0)
        lon, lat, solarElong = child.run(self.ssoObs, self.orb, self.Hval, metricValue)
        self.assertEqual(lon, 10)
        self.assertEqual(lat, 25)

        discMetric3 = metrics.MagicDiscoveryMetric(nObs=5, tWindow=2, snr_limit=5)
        magic = discMetric3.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(magic, 1)
        discMetric3 = metrics.MagicDiscoveryMetric(nObs=3, tWindow=1, snr_limit=5)
        magic = discMetric3.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(magic, 2)
        discMetric3 = metrics.MagicDiscoveryMetric(nObs=4, tWindow=4, snr_limit=5)
        magic = discMetric3.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(magic, 4)

    def testHighVelocityMetric(self):
        rng = np.random.RandomState(8123)
        velMetric = metrics.HighVelocityMetric(psfFactor=1.0, snr_limit=5)
        metricValue = velMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(metricValue, 0)
        self.ssoObs["velocity"][0:2] = 1.5
        metricValue = velMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(metricValue, 2)
        velMetric = metrics.HighVelocityMetric(psfFactor=2.0, snr_limit=5)
        metricValue = velMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(metricValue, 0)
        self.ssoObs["velocity"][0:2] = rng.rand(1)

    def testHighVelocityNightsMetric(self):
        velMetric = metrics.HighVelocityNightsMetric(
            psfFactor=1.0, nObsPerNight=1, snr_limit=5
        )
        metricValue = velMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(metricValue, 0)
        self.ssoObs["velocity"][0:2] = 1.5
        metricValue = velMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(metricValue, self.ssoObs["observationStartMJD"][0])
        self.ssoObs["velocity"][0:2] = np.random.rand(1)


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
        ssoObs = np.recarray([len(times)], dtype=dtype)

        ssoObs["MJD(UTC)"] = times
        ssoObs["RA"] = np.arange(len(times))
        ssoObs["Dec"] = np.arange(len(times))
        ssoObs["magV"] = np.zeros(len(times), dtype="float") + 20.0
        ssoObs["Elongation"] = np.zeros(len(times), dtype=float) + 180.0
        self.Hval = 0.0
        ssoObs["appMagV"] = ssoObs["magV"] + self.Hval
        self.orb = None
        self.ssoObs = ssoObs

    def testKnownObjectsMetric(self):
        knownObjectMetric = metrics.KnownObjectsMetric(
            tSwitch1=self.t1,
            eff1=1.0,
            vMagThresh1=20.5,
            tSwitch2=self.t2,
            eff2=1.0,
            vMagThresh2=20.5,
            tSwitch3=self.t3,
            eff3=1.0,
            vMagThresh3=20.5,
            eff4=1.0,
            vMagThresh4=22,
        )
        mVal = knownObjectMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(mVal, self.ssoObs["MJD(UTC)"].min())
        knownObjectMetric = metrics.KnownObjectsMetric(
            tSwitch1=self.t1,
            eff1=1.0,
            vMagThresh1=15.0,
            tSwitch2=self.t2,
            eff2=1.0,
            vMagThresh2=20.5,
            tSwitch3=self.t3,
            eff3=1.0,
            vMagThresh3=20.5,
            eff4=1.0,
            vMagThresh4=22,
        )
        mVal = knownObjectMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(mVal, self.t1)
        knownObjectMetric = metrics.KnownObjectsMetric(
            tSwitch1=self.t1,
            eff1=0.0,
            vMagThresh1=20.5,
            tSwitch2=self.t2,
            eff2=1.0,
            vMagThresh2=20.5,
            tSwitch3=self.t3,
            eff3=1.0,
            vMagThresh3=20.5,
            eff4=1.0,
            vMagThresh4=22,
        )
        mVal = knownObjectMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(mVal, self.t1)
        knownObjectMetric = metrics.KnownObjectsMetric(
            tSwitch1=self.t1,
            eff1=1.0,
            vMagThresh1=10,
            tSwitch2=self.t2,
            eff2=1.0,
            vMagThresh2=10.5,
            tSwitch3=self.t3,
            eff3=1.0,
            vMagThresh3=20.5,
            eff4=1.0,
            vMagThresh4=22,
        )
        mVal = knownObjectMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(mVal, self.t2)
        knownObjectMetric = metrics.KnownObjectsMetric(
            tSwitch1=self.t1,
            eff1=1.0,
            vMagThresh1=10,
            tSwitch2=self.t2,
            eff2=1.0,
            vMagThresh2=10.5,
            tSwitch3=self.t3,
            eff3=1.0,
            vMagThresh3=10.5,
            eff4=1.0,
            vMagThresh4=22,
        )
        mVal = knownObjectMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(mVal, self.t3)
        knownObjectMetric = metrics.KnownObjectsMetric(
            tSwitch1=self.t1,
            eff1=1.0,
            vMagThresh1=10,
            tSwitch2=self.t2,
            eff2=1.0,
            vMagThresh2=10.5,
            tSwitch3=self.t3,
            eff3=1.0,
            vMagThresh3=10.5,
            eff4=1.0,
            vMagThresh4=10.5,
        )
        mVal = knownObjectMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(mVal, knownObjectMetric.badval)


if __name__ == "__main__":
    unittest.main()
