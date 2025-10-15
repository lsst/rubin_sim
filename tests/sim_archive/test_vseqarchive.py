import datetime
import hashlib
import io
import os
import unittest
from getpass import getuser
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import ClassVar, cast
from uuid import UUID, uuid1

import click
import click.testing
import numpy as np
import pandas as pd
import testing.postgresql
import yaml
from astropy.time import Time
from lsst.resources import ResourcePath
from psycopg2 import sql
from rubin_scheduler.scheduler.utils import SchemaConverter

from rubin_sim.sim_archive import vseqarchive, vseqmetadata
from rubin_sim.sim_archive.prototype import export_sim_to_prototype_sim_archive
from rubin_sim.sim_archive.tempdb import LocalOnlyPostgresql

TEST_METADATA_DB_SCHEMA = "test"

# Test visits using consdb column names
TEST_VISITS = pd.read_csv(
    StringIO(
        """
obs_start_mjd       s_ra      s_dec band  sky_rotation exp_time   altitude     azimuth
64632.271368  51.536172 -12.851173    r    122.606347     29.2  67.439834  317.602598
64632.271816  52.370715  -9.768649    r    124.357953     29.2  65.262546  323.291472
64632.272265  55.094497  -9.088026    r    130.639271     29.2  65.900938  329.433578
64633.272712  54.314320 -12.171543    r    130.159006     29.2  67.624221  321.348869
64633.273163  57.080518 -11.470472    r    136.801494    29.2.  68.327156  328.146833
"""
    ),
    sep=r"\s+",
)

# Test visits using opsim column names
# Taken from baseline, but used instead of baseline
# so it works when get_baseline() fails.
TEST_OBS = pd.DataFrame(
    {
        "ID": {0: 0, 1: 1, 2: 2},
        "RA": {0: 4.565221270744618, 1: 4.587617254424366, 2: 4.569874353050399},
        "dec": {0: -0.47877497696518434, 1: -0.3942901024801995, 2: -0.34708159948758066},
        "mjd": {0: 60980.00162606769, 1: 60980.00187693634, 2: 60980.002122816804},
        "flush_by_mjd": {0: 60980.02490014931, 1: 60980.02490014931, 2: 60980.02490014931},
        "exptime": {0: 15.0, 1: 15.0, 2: 15.0},
        "band": {0: "r", 1: "r", 2: "r"},
        "filter": {0: "r", 1: "r", 2: "r"},
        "rotSkyPos": {0: 5.876022662133265, 1: 5.919075559769473, 2: 5.94587053925843},
        "rotSkyPos_desired": {0: 0.0, 1: 0.0, 2: 0.0},
        "nexp": {0: 1, 1: 1, 2: 1},
        "airmass": {0: 1.9425330274137829, 1: 2.0097732394514645, 2: 2.1437619807371933},
        "FWHM_500": {0: 0.9789617075932842, 1: 0.9789617075932842, 2: 0.9789617075932842},
        "FWHMeff": {0: 1.7561043378017294, 1: 1.7923280611783998, 2: 1.8630958886242825},
        "FWHM_geometric": {0: 1.4955177656730214, 1: 1.5252936662886447, 2: 1.5834648204491601},
        "skybrightness": {0: 18.571033643475822, 1: 18.499742358339535, 2: 18.4280549931739},
        "night": {0: 0, 1: 0, 2: 0},
        "slewtime": {0: 106.51934924053808, 1: 5.675050952948157, 2: 5.244072029689301},
        "visittime": {0: 16.0, 1: 16.0, 2: 16.0},
        "slewdist": {0: 1.0476600064430734, 1: 0.08688615327418507, 2: 0.05002065991954836},
        "fivesigmadepth": {0: 21.929073206570653, 1: 21.863266843112314, 2: 21.768922123808583},
        "alt": {0: 0.5428304424974331, 1: 0.5289410180743699, 2: 0.49345760977076153},
        "az": {0: 4.437241068620033, 1: 4.536614524865317, 2: 4.57781450588731},
        "pa": {0: 1.9277474700579063, 1: 1.9698101688642318, 2: 1.997014567506373},
        "pseudo_pa": {0: 1.9306710506086011, 1: 1.9726118749454766, 2: 1.999754703197624},
        "clouds": {0: 0.0, 1: 0.0, 2: 0.0},
        "moonAlt": {0: 1.1813900553282524, 1: 1.1818643717117499, 2: 1.1823244654562768},
        "sunAlt": {0: -0.21560611790270862, 1: -0.2168497474147377, 2: -0.21806791322669622},
        "scheduler_note": {0: "twilight_near_sun, 0", 1: "twilight_near_sun, 0", 2: "twilight_near_sun, 0"},
        "target_name": {0: "bulgy", 1: "bulgy", 2: "bulgy"},
        "target_id": {0: 0, 1: 1, 2: 2},
        "lmst": {0: 22.024928004140648, 1: 22.03096533614179, 2: 22.03688262412642},
        "rotTelPos": {0: 0.7670373688600254, 1: 0.7659252955606934, 2: 0.7662731443238844},
        "rotTelPos_backup": {0: 0.0, 1: 0.0, 2: 0.0},
        "moonAz": {0: 0.4062715059631942, 1: 0.40266683110724966, 2: 0.3991218282595135},
        "sunAz": {0: 4.275439688438064, 1: 4.274507556583111, 2: 4.273593129782555},
        "sunRA": {0: 3.771398634838603, 1: 3.7714029138801326, 2: 3.7714071078422475},
        "sunDec": {0: -0.2499715857671971, 1: -0.24997299288585925, 2: -0.24997437202179135},
        "moonRA": {0: 5.912319257908991, 1: 5.912350571271447, 2: 5.912381259109779},
        "moonDec": {0: -0.1677616241703147, 1: -0.16773479483758946, 2: -0.1677084921417354},
        "moonDist": {0: 1.296308274027283, 1: 1.2808773106616722, 2: 1.3009217682807237},
        "solarElong": {0: 0.7714313357065409, 1: 0.7844179510467449, 2: 0.7670707340378583},
        "moonPhase": {0: 65.73037246219155, 1: 65.73146495686613, 2: 65.73253572916387},
        "cummTelAz": {0: -1.8461922459167281, 1: -1.7467049213662156, 2: -1.7050034487399},
        "observation_reason": {0: "pairs_i_5.0", 1: "pairs_i_5.0", 2: "pairs_i_5.0"},
        "science_program": {0: "None", 1: "None", 2: "None"},
        "cloud_extinction": {0: 0.0, 1: 0.0, 2: 0.0},
        "note": {0: "", 1: "", 2: ""},
    }
)


class TestVisitSequenceArchive(unittest.TestCase):
    temp_dir: ClassVar[TemporaryDirectory]
    test_archive_url: ClassVar[str]
    test_archive: ClassVar[ResourcePath]
    test_database: ClassVar[LocalOnlyPostgresql]
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

    def num_rows_in_table(self, archive: vseqarchive.VisitSequenceArchiveMetadata, table: str) -> int:
        result = archive.query(
            sql.SQL("SELECT COUNT(*) FROM {}.{};").format(
                sql.Identifier(TEST_METADATA_DB_SCHEMA), sql.Identifier(table)
            ),
            {},
        )
        assert isinstance(result[0][0], int)
        num_rows: int = result[0][0]
        return num_rows

    def num_rows_for_visitseq(
        self, archive: vseqarchive.VisitSequenceArchiveMetadata, table: str, visitseq_uuid: UUID
    ) -> int:
        data = {"visitseq_uuid": visitseq_uuid}
        result = archive.query(
            sql.SQL("SELECT COUNT(*) FROM {}.{} WHERE visitseq_uuid={};").format(
                sql.Identifier(TEST_METADATA_DB_SCHEMA),
                sql.Identifier(table),
                sql.Placeholder("visitseq_uuid"),
            ),
            data,
        )
        assert isinstance(result[0][0], int)
        num_rows: int = result[0][0]
        return num_rows

    def test_record_visitseq_metadata(self) -> None:
        # Remember how many simulations there were before we add one
        original_num_sequences = self.num_rows_in_table(self.vsarch, "visitseq")

        visits = TEST_VISITS
        label = f"Test on {Time.now().iso}"
        vseq_uuid = self.vsarch.record_visitseq_metadata(visits, label, table="visitseq")
        assert isinstance(vseq_uuid, UUID)

        num_matching_seqs = self.num_rows_for_visitseq(self.vsarch, "visitseq", vseq_uuid)
        assert num_matching_seqs == 1

        # Verify that we added only one
        new_num_sequences = self.num_rows_in_table(self.vsarch, "visitseq")
        assert new_num_sequences == (original_num_sequences + 1)

    def test_update_visitseq_metadata(self) -> None:
        visits = TEST_VISITS
        label = f"Test on {Time.now().iso}"
        vseq_uuid = self.vsarch.record_visitseq_metadata(visits, label, table="visitseq")
        assert isinstance(vseq_uuid, UUID)

        initial_visitseq_metadata = self.vsarch.get_visitseq_metadata(vseq_uuid)
        assert initial_visitseq_metadata.first_day_obs is None
        self.vsarch.update_visitseq_metadata(vseq_uuid, "first_day_obs", "2020-01-01")
        visitseq_metadata_1 = self.vsarch.get_visitseq_metadata(vseq_uuid)
        assert visitseq_metadata_1.first_day_obs == datetime.date(2020, 1, 1)

    def test_get_set_visitseq_url(self) -> None:
        with NamedTemporaryFile() as temp_file:
            test_url = ResourcePath(temp_file.name).geturl()

        visits = TEST_VISITS
        label = f"Test on {Time.now().iso}"
        vseq_uuid = self.vsarch.record_visitseq_metadata(visits, label, table="visitseq", url=test_url)
        return_url = self.vsarch.get_visitseq_url(vseq_uuid)
        assert return_url == test_url

        with NamedTemporaryFile() as temp_file:
            new_test_url = ResourcePath(temp_file.name).geturl()
        self.vsarch.set_visitseq_url(vseq_uuid, new_test_url)
        return_url = self.vsarch.get_visitseq_url(vseq_uuid)
        assert return_url != test_url
        assert return_url == new_test_url

    def test_record_simulation_metadata_simple(self) -> None:
        # Remember how many simulations there were before we add one
        original_num_sequences = self.num_rows_in_table(self.vsarch, "simulations")

        visits = TEST_VISITS
        label = f"Test on {Time.now().iso}"
        vseq_uuid = self.vsarch.record_simulation_metadata(visits, label)
        assert isinstance(vseq_uuid, UUID)

        num_matching_seqs = self.num_rows_for_visitseq(self.vsarch, "simulations", vseq_uuid)
        assert num_matching_seqs == 1

        # Verify that we added only one
        new_num_sequences = self.num_rows_in_table(self.vsarch, "simulations")
        assert new_num_sequences == (original_num_sequences + 1)

    def test_record_simulation_metadata_long(self) -> None:
        # Remember how many simulations there were before we add one
        original_num_sequences = self.num_rows_in_table(self.vsarch, "simulations")

        visits = TEST_VISITS
        label = f"Test on {Time.now().iso}"
        first_day_obs = "2025-01-01"
        sim_runner_kwargs = {"foo": 5, "bar": [1, 2, 3]}

        vseq_uuid = self.vsarch.record_simulation_metadata(
            visits, label, first_day_obs=first_day_obs, sim_runner_kwargs=sim_runner_kwargs
        )
        assert isinstance(vseq_uuid, UUID)

        num_matching_seqs = self.num_rows_for_visitseq(self.vsarch, "simulations", vseq_uuid)
        assert num_matching_seqs == 1

        # Verify that we added only one
        new_num_sequences = self.num_rows_in_table(self.vsarch, "simulations")
        assert new_num_sequences == (original_num_sequences + 1)

    def test_get_visitseq_metadata(self) -> None:
        visits = TEST_VISITS
        label = f"Test on {Time.now().iso}"
        vseq_uuid = self.vsarch.record_visitseq_metadata(visits, label, table="visitseq")
        visit_seq = self.vsarch.get_visitseq_metadata(vseq_uuid)
        assert visit_seq["visitseq_label"] == label

    def test_record_completed_metadata(self) -> None:
        # Remember how many simulations there were before we add one
        original_num_sequences = self.num_rows_in_table(self.vsarch, "completed")

        visits = TEST_VISITS
        label = f"Test on {Time.now().iso}"
        vseq_uuid = self.vsarch.record_completed_metadata(
            visits, label, query="SELECT * FROM foo WHERE bar='baz';"
        )
        assert isinstance(vseq_uuid, UUID)

        num_matching_seqs = self.num_rows_for_visitseq(self.vsarch, "completed", vseq_uuid)
        assert num_matching_seqs == 1

        # Verify that we added only one
        new_num_sequences = self.num_rows_in_table(self.vsarch, "completed")
        assert new_num_sequences == (original_num_sequences + 1)

    def test_record_mixed_metadata(self) -> None:
        visits = TEST_VISITS
        label = f"Test on {Time.now().iso}"
        last_early_day_obs = 20251201
        first_late_day_obs = 20251202
        early_parent_uuid = uuid1()
        late_parent_uuid = uuid1()
        vseq_uuid = self.vsarch.record_mixed_metadata(
            visits, label, last_early_day_obs, first_late_day_obs, early_parent_uuid, late_parent_uuid
        )
        visit_seq = self.vsarch.get_visitseq_metadata(vseq_uuid, "mixedvisitseq")
        assert visit_seq["last_early_day_obs"] == datetime.date(2025, 12, 1)
        assert visit_seq["first_late_day_obs"] == datetime.date(2025, 12, 2)
        assert visit_seq["early_parent_uuid"] == early_parent_uuid
        assert visit_seq["late_parent_uuid"] == late_parent_uuid

    def test_tagging(self) -> None:
        visits = TEST_VISITS
        label = f"Test on {Time.now().iso}"
        vseq_uuid = self.vsarch.record_visitseq_metadata(visits, label, table="visitseq")

        result = self.vsarch.is_tagged(vseq_uuid, "test1")
        assert not result

        self.vsarch.tag(vseq_uuid, "test1")
        result = self.vsarch.is_tagged(vseq_uuid, "test1")
        assert result

        self.vsarch.tag(vseq_uuid, "test2", "test3")
        result = self.vsarch.is_tagged(vseq_uuid, "test2")
        result = self.vsarch.is_tagged(vseq_uuid, "test3")
        assert result

        self.vsarch.untag(vseq_uuid, "test2")
        assert not self.vsarch.is_tagged(vseq_uuid, "test2")
        assert self.vsarch.is_tagged(vseq_uuid, "test3")

        self.vsarch.untag(vseq_uuid, "test3")
        self.vsarch.untag(vseq_uuid, "test1")

    def test_comments(self) -> None:
        visits = TEST_VISITS
        label = f"Test on {Time.now().iso}"
        vseq_uuid = self.vsarch.record_visitseq_metadata(visits, label, table="visitseq")

        # There should be no comments on our newly created visitseq
        # to start with. What happens when we ask for comments?
        initial_comments = self.vsarch.get_comments(vseq_uuid)
        assert len(initial_comments) == 0

        # Test without an author
        comment1 = "This is a first comment."
        self.vsarch.comment(vseq_uuid, comment1)
        comments_after_one = self.vsarch.get_comments(vseq_uuid)
        assert len(comments_after_one) == 1
        assert comment1 in comments_after_one.comment.values

        # Test with an author
        comment2 = "This is a test comment"
        author = "Unit Tester"
        self.vsarch.comment(vseq_uuid, comment2, author=author)
        comments_after_two = self.vsarch.get_comments(vseq_uuid)
        assert len(comments_after_two) == 2
        assert comment1 in comments_after_two.comment.values
        assert comment2 in comments_after_two.comment.values
        assert author in comments_after_two.author.values

    def test_register_and_get_file_url(self) -> None:
        # Simple test: register a file, and see if we can get the URL back.
        with NamedTemporaryFile() as temp_file:
            file_content = os.urandom(100)
            file_sha256 = bytes.fromhex(hashlib.sha256(file_content).hexdigest())
            file_rp = ResourcePath(temp_file.name)
            file_url = file_rp.geturl()
            file_type = "test"
            test_uuid = uuid1()
            self.vsarch.register_file(test_uuid, file_type, file_sha256, file_url)
            returned_url = self.vsarch.get_file_url(test_uuid, file_type)
            assert returned_url == file_url

            returned_sha256 = self.vsarch.get_file_sha256(test_uuid, file_type)
            assert returned_sha256 == file_sha256
        # Make sure we cannot register a file (same UUID and type) twice
        # accidentally.
        with NamedTemporaryFile() as temp_file:
            new_file_content = os.urandom(100)
            new_file_sha256 = bytes.fromhex(hashlib.sha256(new_file_content).hexdigest())
            new_file_rp = ResourcePath(temp_file.name)
            new_file_url = new_file_rp.geturl()
            file_type = "test"
            with self.assertRaises(ValueError):
                self.vsarch.register_file(test_uuid, file_type, new_file_sha256, new_file_url)

            returned_url = self.vsarch.get_file_url(test_uuid, file_type)
            assert returned_url != new_file_url
            assert returned_url == file_url

            # Now, insist we want to replace the old value.
            self.vsarch.register_file(test_uuid, file_type, new_file_sha256, new_file_url, update=True)
            returned_url = self.vsarch.get_file_url(test_uuid, file_type)
            assert returned_url == new_file_url
            assert returned_url != file_url

        # Verify that passing the ResourcePath instead of a URL works
        with NamedTemporaryFile() as temp_file:
            file_content = os.urandom(100)
            file_sha256 = bytes.fromhex(hashlib.sha256(file_content).hexdigest())
            file_rp = ResourcePath(temp_file.name)
            file_type = "test"
            test_uuid = uuid1()
            self.vsarch.register_file(test_uuid, file_type, file_sha256, file_rp)
            returned_url = self.vsarch.get_file_url(test_uuid, file_type)
            assert returned_url == file_rp.geturl()

        # Test behavior when there are no matching files
        test_uuid = uuid1()
        with self.assertRaises(ValueError) as assert_raises_context:
            returned_url = self.vsarch.get_file_url(test_uuid, "test")

        assert assert_raises_context.exception.args[0].startswith("No URLs found for test for visitseq")

    def test_nightly_stats(self) -> None:
        visits = TEST_VISITS.copy()
        times = Time(np.floor(visits.obs_start_mjd - 0.5), format="mjd").to_datetime()
        assert isinstance(times, np.ndarray)
        visits = visits.assign(day_obs=pd.Series(times).dt.date)

        test_uuid = uuid1()
        stats_df = vseqarchive.compute_nightly_stats(visits)
        self.vsarch.insert_nightly_stats(test_uuid, stats_df)
        returned_stats_df = self.vsarch.query_nightly_stats(test_uuid)
        assert len(returned_stats_df) > 0
        assert len(stats_df) == len(returned_stats_df)

    def test_record_conda_env(self) -> None:
        conda_env_hash, conda_env_json = vseqarchive.compute_conda_env()
        self.vsarch.record_conda_env(conda_env_hash, conda_env_json)
        assert self.vsarch.conda_env_is_saved(conda_env_hash)
        assert conda_env_hash is not None

    def test_archive_file(self) -> None:
        test_file_type = "testbytes"
        visits = TEST_VISITS
        label = f"Test on {Time.now().iso}"
        vseq_uuid = self.vsarch.record_visitseq_metadata(visits, label, table="visitseq")
        content = os.urandom(100)

        with NamedTemporaryFile() as temp_file:
            file_name = temp_file.name
            temp_file.write(content)
            temp_file.flush()
            sent_location = vseqarchive.add_file(
                self.vsarch, vseq_uuid, file_name, test_file_type, self.test_archive
            )

        found_location = self.vsarch.get_file_url(vseq_uuid, test_file_type)
        assert found_location == sent_location.geturl()

        found_content = ResourcePath(uri=found_location).read()
        assert found_content == content

    def test_archive_visits(self) -> None:
        # Saving with the file type of visits takes a different
        # path through the code, so test it separately.
        test_file_type = "visits"
        visits = TEST_VISITS
        label = f"Test on {Time.now().iso}"
        vseq_uuid = self.vsarch.record_visitseq_metadata(visits, label, table="visitseq")

        with NamedTemporaryFile() as temp_file:
            visits.to_hdf(temp_file.name, key="observations")
            file_name = temp_file.name
            sent_location = vseqarchive.add_file(
                self.vsarch, vseq_uuid, file_name, test_file_type, self.test_archive
            )

        found_location = self.vsarch.get_file_url(vseq_uuid, test_file_type)
        assert found_location == sent_location.geturl()

        visits_found_location = self.vsarch.get_visitseq_url(vseq_uuid)
        assert visits_found_location == sent_location.geturl()

        reread_visits = vseqarchive.get_visits(found_location)
        pd.testing.assert_frame_equal(reread_visits, visits)

        max_obs_start_mjd = visits.obs_start_mjd.max()
        max_visits = vseqarchive.get_visits(
            found_location, query=f"obs_start_mjd > {max_obs_start_mjd - 0.0001}"
        )
        assert len(max_visits) == 1

    def test_sims_on_nights(self) -> None:
        # Make a tag for just this execution of this test:
        test_tag = str(uuid1())

        first_day_obs = "2025-12-01"
        last_day_obs = "2025-12-03"
        tags = (test_tag,)

        # Verify that we start off with no sims matching
        # our test tag.
        result = self.vsarch.sims_on_nights(first_day_obs, last_day_obs, tags)
        assert len(result) == 0

        # Add a sim with our tag, and verify we see it
        sim_uuids = []
        sim_uuid = self.vsarch.record_simulation_metadata(
            TEST_VISITS,
            f"Test {test_tag} on {Time.now().iso}",
            telescope="simonyi",
            first_day_obs="2025-12-01",
            last_day_obs="2025-12-03",
        )
        sim_uuids.append(sim_uuid)
        self.vsarch.tag(sim_uuid, *tags)
        result = self.vsarch.sims_on_nights(first_day_obs, last_day_obs, tags)
        assert len(result) == 1
        assert result.visitseq_uuid[0] == sim_uuid

        # Make sure we do not get it back if we ask for a tag it
        # does not have
        result = self.vsarch.sims_on_nights(first_day_obs, last_day_obs, tags + ("extra",))
        assert len(result) == 0

        # Add another sim with a wider time range, and make sure
        # we get both back.
        sim_uuid = self.vsarch.record_simulation_metadata(
            TEST_VISITS,
            f"Test {test_tag} on {Time.now().iso}",
            telescope="simonyi",
            first_day_obs="2025-11-15",
            last_day_obs="2025-12-30",
        )
        sim_uuids.append(sim_uuid)
        self.vsarch.tag(sim_uuid, *tags)
        result = self.vsarch.sims_on_nights(first_day_obs, last_day_obs, tags)
        assert set(sim_uuids) == set(result.visitseq_uuid)

        # Shift the time range we request to get only the second
        # we added
        result = self.vsarch.sims_on_nights("2025-12-10", tags=tags)
        assert len(result) == 1
        assert sim_uuids[-1] == result.visitseq_uuid[0]

        # Make sure we do not see this if we ask for auxtel
        result = self.vsarch.sims_on_nights(first_day_obs, last_day_obs, tags, telescope="auxtel")
        assert len(result) == 0

    def test_proto_export_import(self) -> None:
        # ***** Create a test simulation
        label = f"Test on {Time.now().iso}"
        first_day_obs = "2025-01-01"
        last_day_obs = "2025-01-05"
        sim_runner_kwargs = {"foo": 5, "bar": [1, 2, 3]}

        sim_uuid = self.vsarch.record_simulation_metadata(
            TEST_VISITS,
            label,
            first_day_obs=first_day_obs,
            last_day_obs=last_day_obs,
            sim_runner_kwargs=sim_runner_kwargs,
        )
        in_tags = ["prototest1", "proto"]
        self.vsarch.tag(sim_uuid, *in_tags)

        with NamedTemporaryFile() as temp_file:
            TEST_VISITS.to_hdf(temp_file.name, key="observations")
            file_name = temp_file.name
            vseqarchive.add_file(self.vsarch, sim_uuid, file_name, "visits", self.test_archive)

        # Include an attached file
        content = os.urandom(100)
        test_file_type = "testbytes"
        with NamedTemporaryFile() as temp_file:
            file_name = temp_file.name
            temp_file.write(content)
            temp_file.flush()
            vseqarchive.add_file(self.vsarch, sim_uuid, file_name, test_file_type, self.test_archive)

        # ***** Send it to a prototype archive
        proto_archive_url = "file://" + self.temp_dir.name + "/proto/"
        proto_sim_rp = export_sim_to_prototype_sim_archive(self.vsarch, sim_uuid, proto_archive_url)
        assert proto_sim_rp.isdir()
        assert proto_sim_rp.exists()
        proto_sim_index = proto_sim_rp.ospath.split("/")[-2]
        assert proto_sim_index.isdigit()
        today = datetime.datetime.now().date().isoformat()
        assert proto_sim_rp.ospath.split("/")[-3] == today

        metadata_yaml_rp = proto_sim_rp.join("sim_metadata.yaml")
        assert metadata_yaml_rp.exists()
        read_metadata = yaml.safe_load(metadata_yaml_rp.read().decode("UTF-8"))

        assert set(read_metadata["tags"]) == set(in_tags)
        assert read_metadata["label"] == label
        assert read_metadata["uuid"] == str(sim_uuid)
        read_bytes = ResourcePath(read_metadata["files"]["testbytes"]["url"]).read()
        assert read_bytes == content

        # ***** Transfer it back out of the prototype archive into
        #       a new entry in the vseqarchive
        new_sim_uuid = self.vsarch.import_sim_from_prototype_sim_archive(
            self.test_archive_url, today, proto_sim_index, proto_archive_url
        )
        round_trip_metadata = self.vsarch.get_visitseq_metadata(new_sim_uuid, "simulations_extra")
        assert round_trip_metadata["files"]["testbytes"] == read_metadata["files"]["testbytes"]["url"]
        assert set(["from_prototype_sim_archive"] + in_tags) == set(round_trip_metadata["tags"])

    def run_click_command(self, command: list[str]) -> str:
        # Wrapper around click's testing tools that sets up
        # the environment to point at the tests and runs
        # a command in that environment.

        dsn = self.test_database.psycopg2_dsn()
        env = {
            "VSARCHIVE_PGDATABASE": dsn["database"],
            "VSARCHIVE_PGHOST": dsn["host"],
            "VSARCHIVE_PGPORT": str(dsn["port"]),
            "VSARCHIVE_PGUSER": getuser(),
            "VSARCHIVE_PGSCHEMA": TEST_METADATA_DB_SCHEMA,
        }

        runner = click.testing.CliRunner()
        result = runner.invoke(vseqarchive.vseqarchive, command, env=env)

        # Verify the command ran successfully.
        self.assertEqual(result.exit_code, 0, msg=result.output)
        return result.output

    def test_cli(self) -> None:
        # Create a temporary HDF5 file with a ``visits`` key
        # and test recording it.
        with NamedTemporaryFile(suffix=".h5") as temp_file:
            # Make a dummy data file

            TEST_VISITS.to_hdf(temp_file.name, key="observations", mode="w")

            table_name = "simulations"
            label = f"CLI test {Time.now().iso}"
            url = ResourcePath(temp_file.name).geturl()
            record_command = [
                "record-visitseq-metadata",
                table_name,
                temp_file.name,
                label,
                "--telescope",
                "simonyi",
                "--url",
                url,
            ]

            output = self.run_click_command(record_command)

            # Extract the UUID that was printed.
            uuid_str = output.strip()
            visitseq_uuid = UUID(uuid_str)

            # Verify that the stored label matches what we supplied.
            stored_metadata = self.vsarch.get_visitseq_metadata(visitseq_uuid, table=table_name)
            self.assertEqual(stored_metadata["visitseq_label"], label)

            # Archive the visits themselves
            archive_visits_command = [
                "archive-file",
                uuid_str,
                temp_file.name,
                "visits",
                "--archive-base",
                self.test_archive_url,
            ]
            output = self.run_click_command(archive_visits_command)
            archived_url = output.strip()
            assert archived_url.startswith(self.test_archive_url)
            assert archived_url.endswith(Path(temp_file.name).name)

            # Compute and add nightly statistics
            nightly_stats_command = [
                "add-nightly-stats",
                uuid_str,
                temp_file.name,
            ]
            output = self.run_click_command(nightly_stats_command)
            stats_from_output = pd.read_csv(io.StringIO(output), sep="\t")
            assert len(stats_from_output) > 0

            query_nightly_stats_command = ["query-nightly-stats", uuid_str]
            query_output = self.run_click_command(query_nightly_stats_command)
            returned_stats = pd.read_csv(io.StringIO(query_output), sep="\t")
            assert len(returned_stats) == len(stats_from_output)

        # Get the file out of the archive, and test it against
        # what we sent.
        with TemporaryDirectory() as temp_dir:
            visits_fname = str(Path(temp_dir).joinpath("visits.h5"))
            get_visits_command = ["get-file", uuid_str, "visits", visits_fname]
            self.run_click_command(get_visits_command)
            read_visits = pd.read_hdf(visits_fname, "observations")
            assert isinstance(read_visits, pd.DataFrame)
            pd.testing.assert_frame_equal(read_visits, TEST_VISITS)

        # Attach a different file and see if we can get it back
        test_content = "I am some test content."
        with TemporaryDirectory() as temp_dir:
            origin_fname = str(Path(temp_dir).joinpath("origin.txt"))
            with open(origin_fname, "w") as origin_fp:
                origin_fp.write(test_content)
            arch_file_command = [
                "archive-file",
                uuid_str,
                origin_fname,
                "test",
                "--archive-base",
                self.test_archive_url,
            ]
            self.run_click_command(arch_file_command)
            dest_fname = str(Path(temp_dir).joinpath("destination.txt"))
            get_file_command = ["get-file", uuid_str, "test", dest_fname]
            self.run_click_command(get_file_command)
            with open(dest_fname) as dest_fp:
                returned_content = dest_fp.read()
            assert returned_content == test_content

        # Test getting the URL back from the metadata database
        get_url_command = [
            "get-visitseq-url",
            uuid_str,
        ]
        first_get_url_output = self.run_click_command(get_url_command)
        first_returned_url = first_get_url_output.strip()
        assert first_returned_url == archived_url

        # Test changing other metadata.
        # Use first_day_obs as an example
        assert stored_metadata.first_day_obs is None
        test_day_obs_datetime = cast(
            datetime.datetime,
            Time(TEST_VISITS.obs_start_mjd.min() - 0.5, format="mjd").datetime,
        )
        test_day_obs_str = test_day_obs_datetime.date().isoformat()
        update_first_day_obs_command = [
            "update-visitseq-metadata",
            uuid_str,
            "first_day_obs",
            test_day_obs_str,
        ]
        self.run_click_command(update_first_day_obs_command)
        updated_metadata = self.vsarch.get_visitseq_metadata(visitseq_uuid, table=table_name)
        assert updated_metadata.first_day_obs.isoformat() == test_day_obs_str

        # This will be needed when testing prenight index creation.
        update_last_day_obs_command = [
            "update-visitseq-metadata",
            uuid_str,
            "last_day_obs",
            test_day_obs_str,
        ]
        self.run_click_command(update_last_day_obs_command)

        #
        # Test tagging and untagging
        #

        sample_tags = ["testtag1", "testtag2", "testtag3"]

        # Check that a sample tag is not there
        is_tagged1_command = ["is-tagged", uuid_str, sample_tags[0]]
        is_tagged_output = self.run_click_command(is_tagged1_command)
        is_tagged = is_tagged_output.strip()
        assert is_tagged == "false"

        # add our sample tags, and verify that they are there
        tag_command = ["tag", uuid_str] + sample_tags
        self.run_click_command(tag_command)
        for tag in sample_tags:
            is_tagged_command = ["is-tagged", uuid_str, tag]
            assert self.run_click_command(is_tagged_command).strip() == "true"

        # drop a sample tag, and check that it is gone.
        untag_command = ["untag", uuid_str, sample_tags[0]]
        self.run_click_command(untag_command)
        is_tagged_command = ["is-tagged", uuid_str, sample_tags[0]]
        assert self.run_click_command(is_tagged_command).strip() == "false"

        #
        # Test adding comments
        #

        # Verify that there are no comments before we've added any.
        get_comments_command = ["get-comments", uuid_str]
        init_comments = self.run_click_command(get_comments_command)
        assert len(init_comments.strip()) == 0

        # Add a comment
        test_comment = "My first test comment."
        add_first_comment_command = ["comment", uuid_str, test_comment]
        self.run_click_command(add_first_comment_command)
        after_first_test_comment = self.run_click_command(get_comments_command)
        assert test_comment in after_first_test_comment

        next_test_comment = "My second test comment."
        add_second_comment_command = ["comment", uuid_str, next_test_comment]
        self.run_click_command(add_second_comment_command)
        after_second_test_comment = self.run_click_command(get_comments_command)
        assert test_comment in after_second_test_comment
        assert next_test_comment in after_second_test_comment

        # Make a temp to test prenight index creation.
        with TemporaryDirectory() as temp_dir:
            # Tag the sim as a prenight so it will be picked up
            tag_command = ["tag", uuid_str, "prenight"]
            self.run_click_command(tag_command)

            # Test making a prenight index
            prenight_index_base_rp = ResourcePath(temp_dir).join("prenight", forceDirectory=True)
            make_prenight_index_command = [
                "make-prenight-index",
                test_day_obs_str,
                "simonyi",
                "--destination",
                prenight_index_base_rp.geturl(),
            ]
            make_index_output = self.run_click_command(make_prenight_index_command)
            prenight_index_url = make_index_output.strip()
            prenight_index_df = pd.read_json(ResourcePath(prenight_index_url).ospath, orient="index")
            assert prenight_index_df.loc[uuid_str, "first_day_obs"] == test_day_obs_str

        # Test exporting a sim to a prototype archive,
        # and importing it again.

        # Make a temp archive for the test proto archive.
        with TemporaryDirectory() as temp_dir:
            proto_base_rp = ResourcePath(temp_dir).join("proto", forceDirectory=True)
            proto_base_url = proto_base_rp.geturl()

            # Write a test sim into the test proto archive
            output = self.run_click_command(
                ["export-proto", uuid_str, "--proto-sim-archive-url", proto_base_url]
            )
            exported_url = output.strip()

            expected_date = datetime.datetime.now(datetime.UTC).date().isoformat()
            expected_exported_url = proto_base_rp.join(expected_date).join("1", forceDirectory=True).geturl()
            assert exported_url == expected_exported_url

            # Import the test sim back from the proto archive
            output = self.run_click_command(
                [
                    "import-proto",
                    self.test_archive_url,
                    expected_date,
                    "1",
                    "--proto-sim-archive-url",
                    proto_base_url,
                ]
            )
            # Make sure we get back a valid UUID
            UUID(output.strip())

        # Test setting the URL to something else.
        new_url = f"{url}/new_extra_stuff"
        set_url_command = ["set-visitseq-url", uuid_str, new_url]
        self.run_click_command(set_url_command)
        new_get_url_output = self.run_click_command(get_url_command)
        new_returned_url = new_get_url_output.strip()
        assert new_returned_url != first_returned_url
        assert new_returned_url == new_url

    def test_record_conda_env_cli(self) -> None:
        # Separate this from the other CLI testing
        # because it is much slower than anything
        # else, and we may want to disable this
        # test independenty of the others.
        output = self.run_click_command(["record-conda-env"])
        hash_hex = output.strip()
        hash_sha256 = bytes.fromhex(hash_hex)

        query = sql.SQL(
            """
            SELECT package_version
            FROM {}.conda_packages
            WHERE conda_env_hash={} AND package_name='numpy'
        """
        ).format(sql.Identifier(TEST_METADATA_DB_SCHEMA), sql.Placeholder("hash_sha256"))
        data = {"hash_sha256": hash_sha256}
        found_versions = self.vsarch.query(query, data)
        assert len(found_versions) == 1
        found_version = found_versions[0][0]
        assert found_version == np.__version__

        # Associate it with a simulation, and query
        # it from the simulation
        # Begin by adding a simulation
        visits = TEST_VISITS
        label = f"Test on {Time.now().iso}"
        vseq_uuid = self.vsarch.record_simulation_metadata(visits, label)

        # Update the simulation to refer to the conda env
        update_sim_conda_env_command = [
            "update-visitseq-metadata",
            str(vseq_uuid),
            "conda_env_sha256",
            hash_hex,
        ]
        self.run_click_command(update_sim_conda_env_command)

        query = sql.SQL(
            """
            SELECT package_version
            FROM {}.simulation_packages
            WHERE package_name='numpy' AND visitseq_uuid={}
        """
        ).format(
            sql.Identifier(TEST_METADATA_DB_SCHEMA),
            sql.Placeholder("visitseq_uuid"),
        )
        package_query_data = {"visitseq_uuid": vseq_uuid}
        found_versions = self.vsarch.query(query, package_query_data)
        assert len(found_versions) == 1
        found_version = found_versions[0][0]
        assert found_version == np.__version__

    def test_opsim_cli(self) -> None:
        # Test that we can archive and retrieve visits
        # in the sqlite3 format.

        # Make a sample file in opsim sqlite3 database format
        schema_converter = SchemaConverter()
        db_file_name = str(Path(self.temp_dir.name).joinpath("test_opsim_cli_opsim.db"))
        schema_converter.obs2opsim(TEST_OBS, db_file_name)

        # Make a visit sequence and send the visits
        # attached to it.
        label = f"CLI test {Time.now().iso}"
        record_command = [
            "record-visitseq-metadata",
            "visitseq",
            db_file_name,
            label,
            "--telescope",
            "simonyi",
        ]
        uuid_str = self.run_click_command(record_command).strip()

        # Archive the visits from the sqlite3 file
        archive_visits_command = [
            "archive-file",
            uuid_str,
            db_file_name,
            "visits",
            "--archive-base",
            self.test_archive_url,
        ]
        self.run_click_command(archive_visits_command)

        # Get it back as an h5 file
        h5_file_name = str(Path(self.temp_dir.name).joinpath("test_opsim_cli_returned.h5"))
        get_h5_visits_command = ["get-file", uuid_str, "visits", h5_file_name]
        self.run_click_command(get_h5_visits_command)
        returned_visits_h5 = pd.read_hdf(h5_file_name, "observations")
        assert len(returned_visits_h5) == len(TEST_OBS)

        ret_db_file_name = str(Path(self.temp_dir.name).joinpath("test_opsim_cli_returned.db"))
        get_opsim_visits_command = ["get-file", uuid_str, "visits", ret_db_file_name]
        self.run_click_command(get_opsim_visits_command)
        ret_obs = schema_converter.opsim2obs(ret_db_file_name)
        assert len(ret_obs) == len(TEST_OBS)
