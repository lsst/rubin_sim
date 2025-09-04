import unittest
from io import StringIO
from tempfile import TemporaryDirectory
from uuid import UUID

import pandas as pd
from astropy.time import Time
from psycopg2 import sql

from rubin_sim import visitsarch

TEST_METADATA_SCHEMA = sql.Identifier("ehntest")
TEST_METADATA_DATABASE = {"database": "opsim_log", "host": "134.79.23.205", "schema": "ehntest"}

TEST_VISITS = pd.read_csv(
    StringIO(
        """
observationStartMJD	fieldRA	            fieldDec	        band	rotSkyPos           visitExposureTime
64632.27136770076	51.536171530913606	-12.851173055354066	r	    122.60634698116121	29.2
64632.27181591137	52.370714702087625	-9.768648967698063	r	    124.35795331534635	29.2
64632.272264776075	55.09449662658469	-9.088025784959735	r	    130.6392705609067	29.2
64632.27271223557	54.31431950468007	-12.171542795692007	r	    130.1590058301397	29.2
64632.27316256726	57.08051812751406	-11.47047216067004	r	    136.8014944653928	29.2
"""
    ),
    sep=r"\s+",
)


class TestVisitSetArchive(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.test_archive = "file://" + self.temp_dir.name + "/"

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
        visit_seq_archive = visitsarch.VisitSequenceArchive(metadata_db=TEST_METADATA_DATABASE)

        # Remember how many simulations there were before we add one
        original_num_sequences = self.num_rows_in_table(visit_seq_archive, "visitseq")

        visits = TEST_VISITS
        label = f"Test on {Time.now().iso}"
        vseq_uuid = visit_seq_archive.record_visitseq_metadata(visits, label, table="visitseq")
        assert isinstance(vseq_uuid, UUID)

        num_matching_seqs = self.num_rows_for_visitseq(visit_seq_archive, "visitseq", vseq_uuid)
        assert num_matching_seqs == 1

        # Verify that we added only one
        new_num_sequences = self.num_rows_in_table(visit_seq_archive, "visitseq")
        assert new_num_sequences == (original_num_sequences + 1)

    def test_record_simulation_metadata_simple(self) -> None:
        visit_seq_archive = visitsarch.VisitSequenceArchive(metadata_db=TEST_METADATA_DATABASE)

        # Remember how many simulations there were before we add one
        original_num_sequences = self.num_rows_in_table(visit_seq_archive, "simulations")

        visits = TEST_VISITS
        label = f"Test on {Time.now().iso}"
        vseq_uuid = visit_seq_archive.record_simulation_metadata(visits, label)
        assert isinstance(vseq_uuid, UUID)

        num_matching_seqs = self.num_rows_for_visitseq(visit_seq_archive, "simulations", vseq_uuid)
        assert num_matching_seqs == 1

        # Verify that we added only one
        new_num_sequences = self.num_rows_in_table(visit_seq_archive, "simulations")
        assert new_num_sequences == (original_num_sequences + 1)

    def test_record_simulation_metadata_long(self) -> None:
        visit_seq_archive = visitsarch.VisitSequenceArchive(metadata_db=TEST_METADATA_DATABASE)

        # Remember how many simulations there were before we add one
        original_num_sequences = self.num_rows_in_table(visit_seq_archive, "simulations")

        visits = TEST_VISITS
        label = f"Test on {Time.now().iso}"
        first_day_obs = "2025-01-01"
        sim_runner_kwargs = {"foo": 5, "bar": [1, 2, 3]}

        vseq_uuid = visit_seq_archive.record_simulation_metadata(
            visits, label, first_day_obs=first_day_obs, sim_runner_kwargs=sim_runner_kwargs
        )
        assert isinstance(vseq_uuid, UUID)

        num_matching_seqs = self.num_rows_for_visitseq(visit_seq_archive, "simulations", vseq_uuid)
        assert num_matching_seqs == 1

        # Verify that we added only one
        new_num_sequences = self.num_rows_in_table(visit_seq_archive, "simulations")
        assert new_num_sequences == (original_num_sequences + 1)

    def test_record_completed_metadata(self) -> None:
        visit_seq_archive = visitsarch.VisitSequenceArchive(metadata_db=TEST_METADATA_DATABASE)

        # Remember how many simulations there were before we add one
        original_num_sequences = self.num_rows_in_table(visit_seq_archive, "completed")

        visits = TEST_VISITS
        label = f"Test on {Time.now().iso}"
        vseq_uuid = visit_seq_archive.record_completed_metadata(
            visits, label, query="SELECT * FROM foo WHERE bar='baz';"
        )
        assert isinstance(vseq_uuid, UUID)

        num_matching_seqs = self.num_rows_for_visitseq(visit_seq_archive, "completed", vseq_uuid)
        assert num_matching_seqs == 1

        # Verify that we added only one
        new_num_sequences = self.num_rows_in_table(visit_seq_archive, "completed")
        assert new_num_sequences == (original_num_sequences + 1)

    def test_tagging(self) -> None:
        visit_seq_archive = visitsarch.VisitSequenceArchive(metadata_db=TEST_METADATA_DATABASE)

        visits = TEST_VISITS
        label = f"Test on {Time.now().iso}"
        vseq_uuid = visit_seq_archive.record_visitseq_metadata(visits, label, table="visitseq")

        result = visit_seq_archive.is_tagged(vseq_uuid, "test1")
        assert not result

        visit_seq_archive.tag(vseq_uuid, "test1")
        result = visit_seq_archive.is_tagged(vseq_uuid, "test1")
        assert result

        visit_seq_archive.tag(vseq_uuid, "test2", "test3")
        result = visit_seq_archive.is_tagged(vseq_uuid, "test2")
        result = visit_seq_archive.is_tagged(vseq_uuid, "test3")
        assert result

        visit_seq_archive.untag(vseq_uuid, "test2")
        assert not visit_seq_archive.is_tagged(vseq_uuid, "test2")
        assert visit_seq_archive.is_tagged(vseq_uuid, "test3")

        visit_seq_archive.untag(vseq_uuid, "test3")
        visit_seq_archive.untag(vseq_uuid, "test1")

    def test_comments(self) -> None:
        visit_seq_archive = visitsarch.VisitSequenceArchive(metadata_db=TEST_METADATA_DATABASE)

        visits = TEST_VISITS
        label = f"Test on {Time.now().iso}"
        vseq_uuid = visit_seq_archive.record_visitseq_metadata(visits, label, table="visitseq")

        # There should be no comments on our newly created visitseq
        # to start with. What happens when we ask for comments?
        initial_comments = visit_seq_archive.get_comments(vseq_uuid)
        assert len(initial_comments) == 0

        # Test without an author
        comment1 = "This is a first comment."
        visit_seq_archive.comment(vseq_uuid, comment1)
        comments_after_one = visit_seq_archive.get_comments(vseq_uuid)
        assert len(comments_after_one) == 1
        assert comment1 in comments_after_one.comment.values

        # Test with an author
        comment2 = "This is a test comment"
        author = "Unit Tester"
        visit_seq_archive.comment(vseq_uuid, comment2, author=author)
        comments_after_two = visit_seq_archive.get_comments(vseq_uuid)
        assert len(comments_after_two) == 2
        assert comment1 in comments_after_two.comment.values
        assert comment2 in comments_after_two.comment.values
        assert author in comments_after_two.author.values
