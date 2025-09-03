import unittest
from io import StringIO
from tempfile import TemporaryDirectory
from uuid import UUID

import pandas as pd
from astropy.time import Time

from rubin_sim import visitsarch

TEST_METADATA_SCHEMA = "ehntest"
TEST_METADATA_DATABASE = {"database": "opsim_log", "host": "134.79.23.205", "schema": TEST_METADATA_SCHEMA}

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

    def test_record_visitseq_metadata(self) -> None:
        visit_seq_archive = visitsarch.VisitSequenceArchive(metadata_db=TEST_METADATA_DATABASE)

        # Remember how many simulations there were before we add one
        original_num_sequences = visit_seq_archive.direct_metadata_query(
            f"SELECT COUNT(*) FROM {TEST_METADATA_SCHEMA}.visitseq;"
        )[0][0]

        visits = TEST_VISITS
        label = f"Test on {Time.now().iso}"
        vseq_uuid = visit_seq_archive.record_visitseq_metadata(visits, label, table="visitseq")
        assert isinstance(vseq_uuid, UUID)

        matching_seqs = visit_seq_archive.direct_metadata_query(
            f"SELECT * FROM {TEST_METADATA_SCHEMA}.visitseq WHERE visitseq_uuid='{vseq_uuid}';"
        )
        assert len(matching_seqs) == 1

        # Verify that we added only one
        new_num_sequences = visit_seq_archive.direct_metadata_query(
            f"SELECT COUNT(*) FROM {TEST_METADATA_SCHEMA}.visitseq;"
        )[0][0]
        assert new_num_sequences == (original_num_sequences + 1)

    def test_record_simulation_metadata_simple(self) -> None:
        visit_seq_archive = visitsarch.VisitSequenceArchive(metadata_db=TEST_METADATA_DATABASE)

        # Remember how many simulations there were before we add one
        original_num_sequences = visit_seq_archive.direct_metadata_query(
            f"SELECT COUNT(*) FROM {TEST_METADATA_SCHEMA}.simulations;"
        )[0][0]

        visits = TEST_VISITS
        label = f"Test on {Time.now().iso}"
        vseq_uuid = visit_seq_archive.record_simulation_metadata(visits, label)
        assert isinstance(vseq_uuid, UUID)

        matching_seqs = visit_seq_archive.direct_metadata_query(
            f"SELECT * FROM {TEST_METADATA_SCHEMA}.simulations WHERE visitseq_uuid='{vseq_uuid}';"
        )
        assert len(matching_seqs) == 1

        # Verify that we added only one
        new_num_sequences = visit_seq_archive.direct_metadata_query(
            f"SELECT COUNT(*) FROM {TEST_METADATA_SCHEMA}.simulations;"
        )[0][0]
        assert new_num_sequences == (original_num_sequences + 1)

    def test_record_simulation_metadata_long(self) -> None:
        visit_seq_archive = visitsarch.VisitSequenceArchive(metadata_db=TEST_METADATA_DATABASE)

        # Remember how many simulations there were before we add one
        original_num_sequences = visit_seq_archive.direct_metadata_query(
            f"SELECT COUNT(*) FROM {TEST_METADATA_SCHEMA}.simulations;"
        )[0][0]

        visits = TEST_VISITS
        label = f"Test on {Time.now().iso}"
        first_day_obs = "2025-01-01"
        sim_runner_kwargs = {"foo": 5, "bar": [1, 2, 3]}

        vseq_uuid = visit_seq_archive.record_simulation_metadata(
            visits, label, first_day_obs=first_day_obs, sim_runner_kwargs=sim_runner_kwargs
        )
        assert isinstance(vseq_uuid, UUID)

        matching_seqs = visit_seq_archive.direct_metadata_query(
            f"SELECT * FROM {TEST_METADATA_SCHEMA}.simulations WHERE visitseq_uuid='{vseq_uuid}';"
        )
        assert len(matching_seqs) == 1

        # Verify that we added only one
        new_num_sequences = visit_seq_archive.direct_metadata_query(
            f"SELECT COUNT(*) FROM {TEST_METADATA_SCHEMA}.simulations;"
        )[0][0]
        assert new_num_sequences == (original_num_sequences + 1)

    def test_record_completed_metadata(self) -> None:
        visit_seq_archive = visitsarch.VisitSequenceArchive(metadata_db=TEST_METADATA_DATABASE)

        # Remember how many simulations there were before we add one
        original_num_sequences = visit_seq_archive.direct_metadata_query(
            f"SELECT COUNT(*) FROM {TEST_METADATA_SCHEMA}.completed;"
        )[0][0]

        visits = TEST_VISITS
        label = f"Test on {Time.now().iso}"
        vseq_uuid = visit_seq_archive.record_completed_metadata(
            visits, label, query="SELECT * FROM foo WHERE bar='baz';"
        )
        assert isinstance(vseq_uuid, UUID)

        matching_seqs = visit_seq_archive.direct_metadata_query(
            f"SELECT * FROM {TEST_METADATA_SCHEMA}.completed WHERE visitseq_uuid='{vseq_uuid.hex}';"
        )
        assert len(matching_seqs) == 1

        # Verify that we added only one
        new_num_sequences = visit_seq_archive.direct_metadata_query(
            f"SELECT COUNT(*) FROM {TEST_METADATA_SCHEMA}.completed;"
        )[0][0]
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
