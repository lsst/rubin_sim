import os
import unittest
from datetime import date
from io import StringIO
from tempfile import TemporaryDirectory
from typing import ClassVar
from uuid import UUID

import pandas as pd
import testing.postgresql
from lsst.resources import ResourcePath

from rubin_sim.sim_archive import vseqarchive, vseqmetadata
from rubin_sim.sim_archive.prenightindex import (
    get_prenight_index_from_bucket,
    get_prenight_index_from_database,
    get_sim_index_info,
    get_sim_uuid,
)
from rubin_sim.sim_archive.tempdb import LocalOnlyPostgresql

TEST_VISITS = pd.read_csv(
    StringIO("""
observationStartMJD fieldRA   fieldDec   band rotSkyPos  visitExposureTime night target_name
61376.271368        51.536172 -12.851173 r    122.606347 29.2              1     target1
61376.271816        52.370715 -9.768649  r    124.357953 29.2              1     target2
61377.272265        55.094497 -9.088026  r    130.639271 29.2              1     target3
61377.272712        54.314320 -12.171543 r    130.159006 29.2              1     target4
61377.273163        57.080518 -11.470472 r    136.801494 29.2              1     target5
"""),
    sep=r"\s+",
)

TEST_METADATA_DB_SCHEMA = "test"


class TestPrenightIndex(unittest.TestCase):
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

        # Create a simple simulation using TEST_VISITS data
        cls.sim_uuid = cls.vsarch.record_simulation_metadata(
            TEST_VISITS,
            "Test simonyi simulation",
            first_day_obs="2026-12-01",
            last_day_obs="2026-12-02",
            telescope="simonyi",
        )
        # Store the sim creation date for our test
        cls.sim_creation_date = date(2026, 12, 1)
        cls.vsarch.tag(cls.sim_uuid, "prenight", "nominal", "ideal")

        # Add visits to the simulation
        with TemporaryDirectory() as temp_dir:
            visits_file = os.path.join(temp_dir, "visits.h5")
            TEST_VISITS.to_hdf(visits_file, key="observations")
            vseqarchive.add_file(cls.vsarch, cls.sim_uuid, visits_file, "visits", cls.test_archive)

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

    def test_get_prenight_index_from_database(self) -> None:
        # Test the get_prenight_index_from_database function
        result = get_prenight_index_from_database("2026-12-01", telescope="simonyi")

        # Should return a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Should contain the simulation we created
        self.assertIn(self.sim_uuid, result.index)

        # Should have one row for our simulation
        self.assertEqual(len(result), 1)

        # Check that the simulation has the expected tags
        sim_row = result.loc[self.sim_uuid]
        self.assertIn("prenight", sim_row["tags"])
        self.assertIn("nominal", sim_row["tags"])
        self.assertIn("ideal", sim_row["tags"])

    def test_get_prenight_index_from_database_with_different_telescope(self) -> None:
        # Test with auxtel telescope (should return empty DataFrame since
        # we only created simonyi sim)
        result = get_prenight_index_from_database("2026-12-01", telescope="auxtel")

        # Should return an empty DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)

    def test_get_prenight_index_from_database_with_integer_dayobs(self) -> None:
        # Test with integer day_obs
        result = get_prenight_index_from_database(20261201, telescope="simonyi")

        # Should return a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn(self.sim_uuid, result.index)
        self.assertEqual(len(result), 1)

    def test_get_sim_uuid(self) -> None:
        # Test the get_sim_uuid function
        # We should be able to find our test simulation
        # Using today's date as requested
        today = date.today()
        result = get_sim_uuid(20261201, today, 1)

        # Should return a UUID
        self.assertIsInstance(result, UUID)

        # Should match the UUID we created
        self.assertEqual(result, self.sim_uuid)

        # Test with date object for day_obs
        result2 = get_sim_uuid(date(2026, 12, 1), today, 1)
        self.assertEqual(result2, self.sim_uuid)

        # Test with str for day_obs
        result3 = get_sim_uuid("20261201", today, 1)
        self.assertEqual(result3, self.sim_uuid)

        # Test with iso str for day_obs
        result4 = get_sim_uuid("2026-12-01", today, 1)
        self.assertEqual(result4, self.sim_uuid)

        # Test that it raises ValueError for non-existent simulation
        # (wrong date)
        with self.assertRaises(ValueError):
            get_sim_uuid(20261201, date(2022, 12, 2), 1)

        # Test that it raises ValueError for non-existent daily_id
        with self.assertRaises(ValueError):
            get_sim_uuid(20261201, today, 2)

    def test_get_sim_index_info(self) -> None:
        # Test the get_sim_index_info function
        # Get the index information for our test simulation
        result = get_sim_index_info(20261201, self.sim_uuid)

        # Should return a pandas Series
        self.assertIsInstance(result, pd.Series)

        # Should contain the expected UUID
        self.assertEqual(str(result.name), str(self.sim_uuid))

        # Should have the expected tags
        self.assertIn("prenight", result["tags"])
        self.assertIn("nominal", result["tags"])
        self.assertIn("ideal", result["tags"])

        # Test that it raises ValueError for non-existent UUID
        with self.assertRaises(ValueError):
            get_sim_index_info(20261201, UUID("12345678-1234-1234-1234-123456789012"))

    def test_get_prenight_index_from_bucket(self) -> None:
        # Test the get_prenight_index_from_bucket function
        # Create a temporary directory to simulate a bucket
        with TemporaryDirectory() as temp_dir:
            # Create a mock prenight index JSON file similar to the sample data
            mock_index_data = {
                "6c242afb-edd1-4cea-9f8c-80e0a18b4b75": {
                    "sim_creation_day_obs": "2026-04-11",
                    "daily_id": 2,
                    "visitseq_label": "Nominal start and overhead, ideal conditions",
                    "telescope": "simonyi",
                    "first_day_obs": "2026-04-11",
                    "last_day_obs": "2026-04-13",
                    "creation_time": "2026-04-11T14:13:55.024Z",
                    "scheduler_version": "3.21.1",
                    "sim_runner_kwargs": None,
                    "conda_env_sha256": "48bcf84e41a741ee67fe644b1ed8d5858d81a7ecfe012473fe2e2f0f3fc05095",
                    "parent_visitseq_uuid": "94fd43ff-5034-43cd-ac48-6461cdca7979",
                    "parent_last_day_obs": "2026-04-10",
                    "tags": ["ideal", "nominal", "prenight"],
                    "comments": {},
                    "files": {},
                },
                "b9405aaf-dfe8-4508-ad90-cb37527dbc27": {
                    "sim_creation_day_obs": "2026-04-11",
                    "daily_id": 3,
                    "visitseq_label": "Nominal start and overhead, ideal conditions 2",
                    "telescope": "simonyi",
                    "first_day_obs": "2026-04-11",
                    "last_day_obs": "2026-04-13",
                    "creation_time": "2026-04-11T14:21:27.332Z",
                    "scheduler_version": "3.21.1",
                    "sim_runner_kwargs": None,
                    "conda_env_sha256": "48bcf84e41a741ee67fe644b1ed8d5858d81a7ecfe012473fe2e2f0f3fc05095",
                    "parent_visitseq_uuid": "94fd43ff-5034-43cd-ac48-6461cdca7979",
                    "parent_last_day_obs": "2026-04-10",
                    "tags": ["ideal", "nominal", "prenight", "rewards"],
                    "comments": {},
                    "files": {},
                },
            }

            # Create a mock bucket structure
            prenight_index_path = ResourcePath(temp_dir)
            year = "2026"
            month = "4"
            telescope = "simonyi"
            isodate = "2026-04-11"

            # Create the directory structure
            mock_bucket_path = (
                prenight_index_path.join(telescope)
                .join(year)
                .join(month)
                .join(f"{telescope}_prenights_for_{isodate}.json")
            )

            # Create the directory structure manually
            import os

            os.makedirs(os.path.dirname(mock_bucket_path.ospath), exist_ok=True)

            # Write the mock data to a JSON file
            import json

            with open(mock_bucket_path.ospath, "w") as f:
                json.dump(mock_index_data, f)

            # Test the function
            result = get_prenight_index_from_bucket(
                "2026-04-11", telescope="simonyi", prenight_index_path=temp_dir
            )

            # Should return a DataFrame
            self.assertIsInstance(result, pd.DataFrame)

            # Should have the expected number of rows (2 from our mock data)
            self.assertEqual(len(result), 2)

            # Should have the expected UUIDs in the index
            expected_uuids = ["6c242afb-edd1-4cea-9f8c-80e0a18b4b75", "b9405aaf-dfe8-4508-ad90-cb37527dbc27"]
            for uuid in expected_uuids:
                self.assertIn(uuid, result.index)

            # Check that the first row has expected data
            first_row = result.loc["6c242afb-edd1-4cea-9f8c-80e0a18b4b75"]
            self.assertEqual(first_row["sim_creation_day_obs"], "2026-04-11")
            self.assertEqual(first_row["daily_id"], 2)
            self.assertIn("ideal", first_row["tags"])
            self.assertIn("nominal", first_row["tags"])
            self.assertIn("prenight", first_row["tags"])

            # Check that the second row has expected data
            second_row = result.loc["b9405aaf-dfe8-4508-ad90-cb37527dbc27"]
            self.assertEqual(second_row["sim_creation_day_obs"], "2026-04-11")
            self.assertEqual(second_row["daily_id"], 3)
            self.assertIn("ideal", second_row["tags"])
            self.assertIn("nominal", second_row["tags"])
            self.assertIn("prenight", second_row["tags"])
