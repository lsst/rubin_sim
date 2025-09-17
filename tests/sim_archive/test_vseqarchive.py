import datetime
import hashlib
import io
import os
import unittest
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from uuid import UUID, uuid1

import click
import click.testing
import numpy as np
import pandas as pd
from astropy.time import Time
from lsst.resources import ResourcePath
from psycopg2 import sql
from rubin_scheduler.scheduler.utils import SchemaConverter

from rubin_sim.data import get_baseline
from rubin_sim.sim_archive import vseqarchive

TEST_METADATA_DB_SCHEMA = "ehntest"
TEST_METADATA_DB_KWARGS = {"database": "opsim_log", "host": "134.79.23.205"}

TEST_VISITS = pd.read_csv(
    StringIO(
        """
obs_start_mjd   	s_ra	            s_dec    	        band	sky_rotation        exp_time
64632.27136770076	51.536171530913606	-12.851173055354066	r	    122.60634698116121	29.2
64632.27181591137	52.370714702087625	-9.768648967698063	r	    124.35795331534635	29.2
64632.272264776075	55.09449662658469	-9.088025784959735	r	    130.6392705609067	29.2
64633.27271223557	54.31431950468007	-12.171542795692007	r	    130.1590058301397	29.2
64633.27316256726	57.08051812751406	-11.47047216067004	r	    136.8014944653928	29.2
"""
    ),
    sep=r"\s+",
)


class TestVisitSequenceArchive(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.test_archive_url = "file://" + self.temp_dir.name + "/archive/"
        self.test_archive = ResourcePath(self.test_archive_url)
        self.vsarch = vseqarchive.VisitSequenceArchiveMetadata(
            metadata_db_kwargs=TEST_METADATA_DB_KWARGS,
            metadata_db_schema=TEST_METADATA_DB_SCHEMA,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

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

    def run_click_command(self, command: list[str]) -> str:
        # Wrapper around click's testing tools that sets up
        # the environment to point at the tests and runs
        # a command in that environment.

        env = {
            "VSARCHIVE_PGDATABASE": TEST_METADATA_DB_KWARGS["database"],
            "VSARCHIVE_PGHOST": TEST_METADATA_DB_KWARGS["host"],
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

            table_name = "visitseq"
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
            get_visits_command = ["get-file", visits_fname, uuid_str, "visits"]
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
            get_file_command = ["get-file", dest_fname, uuid_str, "test"]
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

        # Test setting the URL to something else.
        new_url = f"{url}/new_extra_stuff"
        set_url_command = ["set-visitseq-url", uuid_str, new_url]
        self.run_click_command(set_url_command)
        new_get_url_output = self.run_click_command(get_url_command)
        new_returned_url = new_get_url_output.strip()
        assert new_returned_url != first_returned_url
        assert new_returned_url == new_url

        # Test changing other metadata.
        # Use first_day_obs as an example
        assert stored_metadata.first_day_obs is None
        test_day_obs_str = (
            Time(TEST_VISITS.obs_start_mjd.min() - 0.5, format="mjd").datetime.date().isoformat()
        )
        update_first_day_obs_command = [
            "update-visitseq-metadata",
            uuid_str,
            "first_day_obs",
            test_day_obs_str,
        ]
        self.run_click_command(update_first_day_obs_command)
        updated_metadata = self.vsarch.get_visitseq_metadata(visitseq_uuid, table=table_name)
        assert updated_metadata.first_day_obs.isoformat() == test_day_obs_str

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
        data = {"visitseq_uuid": vseq_uuid}
        found_versions = self.vsarch.query(query, data)
        assert len(found_versions) == 1
        found_version = found_versions[0][0]
        assert found_version == np.__version__

    def test_opsim_cli(self) -> None:
        # Test that we can archive and retrieve visits
        # in the sqlite3 format.

        # Make a sample file in opsim sqlite3 database format
        schema_converter = SchemaConverter()
        obs = schema_converter.opsim2obs(get_baseline())[:11]
        db_file_name = str(Path(self.temp_dir.name).joinpath("test_opsim_cli_opsim.db"))
        schema_converter.obs2opsim(obs, db_file_name)

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
        get_h5_visits_command = ["get-file", h5_file_name, uuid_str, "visits"]
        self.run_click_command(get_h5_visits_command)
        returned_visits_h5 = pd.read_hdf(h5_file_name, "observations")
        assert len(returned_visits_h5) == len(obs)

        ret_db_file_name = str(Path(self.temp_dir.name).joinpath("test_opsim_cli_returned.db"))
        get_opsim_visits_command = ["get-file", ret_db_file_name, uuid_str, "visits"]
        self.run_click_command(get_opsim_visits_command)
        ret_obs = schema_converter.opsim2obs(ret_db_file_name)
        assert len(ret_obs) == len(obs)
