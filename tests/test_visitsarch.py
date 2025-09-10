import datetime
import hashlib
import os
import unittest
from io import StringIO
from tempfile import NamedTemporaryFile, TemporaryDirectory
from uuid import UUID, uuid1

import numpy as np
import pandas as pd
from astropy.time import Time
from lsst.resources import ResourcePath
from psycopg2 import sql

from rubin_sim import visitsarch

TEST_METADATA_SCHEMA = sql.Identifier("ehntest")
TEST_METADATA_DATABASE = {"database": "opsim_log", "host": "134.79.23.205", "schema": "ehntest"}

TEST_VISITS = pd.read_csv(
    StringIO(
        """
obs_start_mjd   	s_ra	            s_dec    	        band	sky_rotation        exp_time
64632.27136770076	51.536171530913606	-12.851173055354066	r	    122.60634698116121	29.2
64632.27181591137	52.370714702087625	-9.768648967698063	r	    124.35795331534635	29.2
64632.272264776075	55.09449662658469	-9.088025784959735	r	    130.6392705609067	29.2
64632.27271223557	54.31431950468007	-12.171542795692007	r	    130.1590058301397	29.2
64632.27316256726	57.08051812751406	-11.47047216067004	r	    136.8014944653928	29.2
"""
    ),
    sep=r"\s+",
)


class TestVisitSequenceArchive(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.test_archive = "file://" + self.temp_dir.name + "/archive/"
        self.vsarch = visitsarch.VisitSequenceArchive(
            metadata_db=TEST_METADATA_DATABASE, archive_url=self.test_archive
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def num_rows_in_table(self, archive: visitsarch.VisitSequenceArchive, table: str) -> int:
        result = archive.direct_metadata_query(
            sql.SQL("SELECT COUNT(*) FROM {}.{};").format(TEST_METADATA_SCHEMA, sql.Identifier(table)), {}
        )
        assert isinstance(result[0][0], int)
        num_rows: int = result[0][0]
        return num_rows

    def num_rows_for_visitseq(
        self, archive: visitsarch.VisitSequenceArchive, table: str, visitseq_uuid: UUID
    ) -> int:
        data = {"visitseq_uuid": visitseq_uuid}
        result = archive.direct_metadata_query(
            sql.SQL("SELECT COUNT(*) FROM {}.{} WHERE visitseq_uuid={};").format(
                TEST_METADATA_SCHEMA, sql.Identifier(table), sql.Placeholder("visitseq_uuid")
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
        self.vsarch.set_visitseq_url("visitseq", vseq_uuid, new_test_url)
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

    def test_record_nightly_stats(self) -> None:
        visits = TEST_VISITS.copy()
        visits["day_obs"] = pd.Series(
            Time(np.floor(visits.obs_start_mjd - 0.5), format="mjd").to_datetime()
        ).dt.date

        test_uuid = uuid1()
        stats_df = self.vsarch.record_nightly_stats(test_uuid, visits)
        assert len(stats_df) > 0

    def test_record_conda_env(self) -> None:
        conda_env_hash = self.vsarch.record_conda_env()
        assert self.vsarch.conda_env_is_saved(conda_env_hash)
        assert conda_env_hash is not None

    def test_write_file_to_archive(self) -> None:
        visits = TEST_VISITS
        label = f"Test on {Time.now().iso}"
        vseq_uuid = self.vsarch.record_visitseq_metadata(visits, label, table="visitseq")

        with NamedTemporaryFile() as temp_file:
            content = os.urandom(100)
            file_name = temp_file.name
            temp_file.write(content)
            temp_file.flush()
            archived_rp, _ = self.vsarch._write_file_to_archive(vseq_uuid, file_name)

        assert archived_rp.geturl().startswith(self.test_archive)
        reread_content = archived_rp.read()
        assert reread_content == content

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
            sent_location = self.vsarch.archive_file(vseq_uuid, file_name, test_file_type)

        found_location = self.vsarch.get_file_url(vseq_uuid, test_file_type)
        assert found_location == sent_location.geturl()

        found_content = ResourcePath(uri=found_location).read()
        assert found_content == content
