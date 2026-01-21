import os
import unittest
from io import StringIO
from tempfile import TemporaryDirectory
from typing import ClassVar

import pandas as pd
import testing.postgresql
from lsst.resources import ResourcePath

import rubin_sim.sim_archive.prenightindex
from rubin_sim.sim_archive import vseqarchive, vseqmetadata
from rubin_sim.sim_archive.sim_archive import (
    NoMatchingSimulationsFoundError,
    fetch_obsloctap_visits,
    fetch_sim_for_nights,
    fetch_sim_stats_for_night,
    find_latest_prenight_sim_for_nights,
)
from rubin_sim.sim_archive.tempdb import LocalOnlyPostgresql

TEST_VISITS = pd.read_csv(
    StringIO("""
observationStartMJD fieldRA   fieldDec   band rotSkyPos  visitExposureTime night target_name
61376.271368        51.536172 -12.851173 r    122.606347 29.2              1     target1
61376.271816        52.370715 -9.768649  r    124.357953 29.2              1     target2
61376.272265        55.094497 -9.088026  r    130.639271 29.2              1     target3
61377.272712        54.314320 -12.171543 r    130.159006 29.2              1     target4
61377.273163        57.080518 -11.470472 r    136.801494 29.2              1     target5
"""),
    sep=r"\s+",
)

TEST_METADATA_DB_SCHEMA = "test"


class TestSimArchiveUtils(unittest.TestCase):
    temp_dir: ClassVar[TemporaryDirectory]
    test_archive_url: ClassVar[str]
    test_archive: ClassVar[ResourcePath]
    test_database: ClassVar[LocalOnlyPostgresql]
    metadata_db_kwargs: ClassVar[dict]
    vsarch: ClassVar[vseqmetadata.VisitSequenceArchiveMetadata]

    @classmethod
    def setUpClass(cls) -> None:
        try:
            testing.postgresql.find_program("postgres", ["bin"])
        except RuntimeError:
            raise unittest.SkipTest("PostgreSQL not found, skipping visit sequence archive tests")

        cls.temp_dir = TemporaryDirectory()
        cls.test_archive_url = "file://" + cls.temp_dir.name + "/archive/"
        cls.test_archive = ResourcePath(cls.test_archive_url)

        cls.test_database = LocalOnlyPostgresql(base_dir=cls.temp_dir.name)

        cls.vsarch = vseqmetadata.VisitSequenceArchiveMetadata(
            metadata_db_kwargs=cls.test_database.psycopg2_dsn(),
            metadata_db_schema=TEST_METADATA_DB_SCHEMA,
        )
        cls.vsarch.create_schema_in_database()

        # Update module-level variables in vseqmetadata to point to
        # the test database.
        vseqmetadata.VSARCHIVE_PGDATABASE = cls.test_database.psycopg2_dsn()["database"]
        vseqmetadata.VSARCHIVE_PGHOST = cls.test_database.psycopg2_dsn()["host"]
        vseqmetadata.VSARCHIVE_PGUSER = cls.test_database.psycopg2_dsn()["user"]
        vseqmetadata.VSARCHIVE_PGPORT = cls.test_database.psycopg2_dsn()["port"]
        vseqmetadata.VSARCHIVE_PGSCHEMA = TEST_METADATA_DB_SCHEMA

        # Create two simple simulations using TEST_VISITS data

        # Keep track of the simulations we create
        cls.sim_uuids = []

        sim_uuid = cls.vsarch.record_simulation_metadata(
            TEST_VISITS,
            "Test simonyi simulation",
            first_day_obs="2026-12-01",
            last_day_obs="2026-12-02",
            telescope="simonyi",
        )
        cls.sim_uuids.append(sim_uuid)
        cls.vsarch.tag(sim_uuid, "prenight", "nominal", "ideal")

        sim_uuid = cls.vsarch.record_simulation_metadata(
            TEST_VISITS,
            "Test auxtel simulation",
            first_day_obs="2026-12-01",
            last_day_obs="2026-12-02",
            telescope="auxtel",
        )
        cls.sim_uuids.append(sim_uuid)
        cls.vsarch.tag(sim_uuid, "prenight", "nominal", "ideal")

        # Add visits to both simulations
        with TemporaryDirectory() as temp_dir:
            visits_file = os.path.join(temp_dir, "visits.h5")
            TEST_VISITS.to_hdf(visits_file, key="observations")
            vseqarchive.add_file(cls.vsarch, cls.sim_uuids[0], visits_file, "visits", cls.test_archive)
            vseqarchive.add_file(cls.vsarch, cls.sim_uuids[1], visits_file, "visits", cls.test_archive)

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            if cls.test_database is not None:
                cls.test_database.stop()
        finally:
            cls.temp_dir.cleanup()

    def setUp(self) -> None:
        self.start_environ: dict = {}
        self.start_environ.update(os.environ)
        os.environ["LSST_DISABLE_BUCKET_VALIDATION"] = "1"

    def tearDown(self) -> None:
        for key in os.environ:
            if key not in self.start_environ:
                del os.environ[key]
            else:
                if os.environ[key] != self.start_environ[key]:
                    os.environ[key] = self.start_environ[key]

    def test_find_latest_prenight_sim_for_nights(self) -> None:
        # Find a simulation covering the first night
        result = find_latest_prenight_sim_for_nights("2026-12-01", "2026-12-01")
        self.assertIsInstance(result, dict)
        # Check that expected columns are a subset of columns present
        expected_columns = {"visitseq_uuid", "visitseq_url", "files"}
        present_columns = set(result.keys())
        self.assertTrue(
            expected_columns.issubset(present_columns),
            f"Expected columns {expected_columns} are not a subset of present columns {present_columns}",
        )
        self.assertEqual(result["visitseq_uuid"], self.sim_uuids[0])

    def test_fetch_sim_for_nights_default(self) -> None:
        # Test with default which_sim (should use the simonyi sim)
        visits = fetch_sim_for_nights("2026-12-01", "2026-12-01")
        self.assertIsInstance(visits, pd.DataFrame)
        self.assertEqual(len(visits), 3)

    def test_throws_no_sim_exception(self) -> None:
        # Throw the informative exception when no simulations are found.
        self.assertRaises(NoMatchingSimulationsFoundError, fetch_sim_for_nights, "2045-12-01", "2045-12-01")

    def test_fetch_sim_for_nights_with_dict(self) -> None:
        # Test with a dict that selects the second (auxtel) simulation
        visits = fetch_sim_for_nights(
            "2026-12-01", "2026-12-01", which_sim={"tags": ("prenight",), "telescope": "auxtel"}
        )
        self.assertIsInstance(visits, pd.DataFrame)
        self.assertEqual(len(visits), 3)

    def test_fetch_obsloctap_visits(self) -> None:
        # Test fetching visits for a specific night
        visits = fetch_obsloctap_visits("2026-12-01", nights=1, max_simulation_age=365000)
        self.assertIsInstance(visits, pd.DataFrame)
        # Should have the expected columns
        expected_columns = (
            "observationStartMJD",
            "fieldRA",
            "fieldDec",
            "rotSkyPos",
            "band",
            "visitExposureTime",
            "night",
            "target_name",
        )
        for col in expected_columns:
            self.assertIn(col, visits.columns)

        self.assertEqual(len(visits), 3)

    def test_fetch_sim_stats_for_night(self) -> None:
        # Test fetching stats for a night
        rubin_sim.sim_archive.prenightindex.MAX_AGE = 365000
        stats = fetch_sim_stats_for_night("2026-12-01")
        self.assertIsInstance(stats, dict)
        # Should contain at least the nominal_visits key
        self.assertIn("nominal_visits", stats)
        # The number of visits should match TEST_VISITS
        self.assertEqual(stats["nominal_visits"], 3)
