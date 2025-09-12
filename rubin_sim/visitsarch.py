import hashlib
import json
import logging
import os
import subprocess
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Mapping, Tuple
from uuid import UUID

import click
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import psycopg2.pool
import sqlalchemy
from astropy.time import Time
from lsst.resources import ResourcePath, ResourcePathExpression
from psycopg2 import sql

VSARCHIVE_PGDATABASE = "opsim_log"
VSARCHIVE_PGHOST = "134.79.23.205"
ARCHIVE_URL: ResourcePathExpression = "test_archive"

JSON_DUMP_LIMIT = 4096

LOGGER = logging.getLogger(__name__)

psycopg2.extras.register_uuid()


def _dayobs_to_date(dayobs: str | date | int) -> date:
    match dayobs:
        case int():
            year = dayobs // 10000
            month = (dayobs // 100) % 100
            day = dayobs % 100
            dayobs = date(year, month, day)
        case str():
            dayobs = datetime.fromisoformat(dayobs).date()
        case _:
            assert isinstance(dayobs, date)

    return dayobs


def compute_visits_sha256(visits: pd.DataFrame) -> bytes:
    recs = visits.to_records()
    visitseq_hash = hashlib.sha256(str(recs.dtype).encode())
    visitseq_hash.update(np.ascontiguousarray(recs).data.tobytes())
    visitseq_sha256 = bytes.fromhex(visitseq_hash.hexdigest())
    return visitseq_sha256


class VisitSequenceArchive:
    """Interface to archive of sequences of visits.

    Parameters
    ----------
    metadata_db_kwargs: `Mapping`
        A dictionary or other mapping defining the connection
        parameters for connecting to the postgresql database
        that holds the sequence metadata. Keys are passed as keyword
        arguments to `psycopg2.pool.SimppleConnectionPool`.
    metadata_db_schema: `str`
        The schema in the database holding the metadata.
    archive_url: `ResourcePathExpression`
        The base location for the files to be stored in the archive,
        passed to `lsst.resources.ResourcePath`.

    Notes
    -----
    The visit sequence archive consists of two primary architectural
    components: an archive in which files associated with a sequence
    of visits are stored (including, optionally, the table of visits
    themselves); and postegresql database with tables with metadata
    about the sequneces of visits, including the
    provenance of the visits, the locations of the files associated
    with each sequence of visits, comments and tags associated with
    each sequence of visits, and basic statistics of the visits.
    """

    def __init__(
        self,
        metadata_db_kwargs: Mapping | None = None,
        metadata_db_schema: str = "ehntest",
        archive: ResourcePathExpression | ResourcePath | None = None,
    ):
        if not isinstance(metadata_db_kwargs, dict):
            metadata_db_kwargs = {} if metadata_db_kwargs is None else dict(metadata_db_kwargs)
        assert isinstance(metadata_db_kwargs, dict)

        if "database" not in metadata_db_kwargs:
            metadata_db_kwargs["database"] = os.environ.get("VSARCHIVE_PGDATABASE", VSARCHIVE_PGDATABASE)

        if "host" not in metadata_db_kwargs:
            metadata_db_kwargs["host"] = os.environ.get("VSARCHIVE_PGHOST", VSARCHIVE_PGHOST)

        self.pg_pool = psycopg2.pool.SimpleConnectionPool(1, 5, **metadata_db_kwargs)

        self.metadata_db_schema: str = metadata_db_schema

        if isinstance(archive, ResourcePath):
            self.archive_base = archive
        elif isinstance(archive, ResourcePathExpression):
            self.archive_base = ResourcePath(archive)
        else:
            self.archive_base = ResourcePath(ARCHIVE_URL)

    def query(
        self,
        query: str | sql.SQL | sql.Composed,
        data: dict,
        commit: bool = False,
        return_result: bool = True,
    ) -> tuple:
        """Run a simple query on the visit sequence database.

        Parameters
        ----------
        query : `str` or `sql.SQL`
            The query to send
        data : `dict`
            Data to include in the query
        commit : `bool`
            Commit the query (e.g. for an INSERT), defaults to False
        return_result : `bool`
            Return the result, defaults to True

        Returns
        -------
        result : `tuple`
            The result of the query
        """

        connection = None
        try:
            # Get a connection from the pool
            connection = self.pg_pool.getconn()
            with connection.cursor() as cursor:
                cursor.execute(query, data)
                result = cursor.fetchall() if return_result else (None,)
            if commit:
                connection.commit()
            else:
                # If a commit was not requested, make sure
                # there were no alterations to the database
                # in this transaction.
                connection.rollback()
        except Exception as e:
            # If something went wrong, restore the
            # connection to a stable state before
            # returning it to the pool.
            if connection:
                try:
                    connection.rollback()
                except Exception:
                    pass
            LOGGER.exception("Failed to execute query: %s", e)
            raise
        finally:
            # Make sure we always return the connection to
            # the pool.
            if connection:
                self.pg_pool.putconn(connection)

        return result

    def record_visitseq_metadata(
        self,
        visits: pd.DataFrame,
        label: str,
        telescope: str = "simonyi",
        table: str = "visitseq",
        url: str | None = None,
        first_day_obs: date | int | str | None = None,
        last_day_obs: date | int | str | None = None,
        creation_time: Time | None = None,
    ) -> UUID:
        """Record metadata for a new visit sequence.

        Parameters
        ----------
        visits : `pd.DataFrame`
            A DataFrame of visits, with column names following those
            in consdb.
        label : `str`
            A label for the sequence.
        telescope : `str`
            The telescope used, either "simonyi" or "auxtel".
            Defaults to "simonyi".
        table : `str`
            The table in the metadata archive database in which
            to insert the entry: one of ``simulations``,
            ``completed``, ``mixedvisitseq``, or ``visitseq``.
        url : `str`, optional
            A URL for the file from thich the table of visits can
            be download.
        first_day_obs : `date` or `int` or `str`, optional
            The first night of observations, defined according
            to SMTN-032 (UTC-12hrs).
        last_day_obs : `date` or `int` or `str`, optional
            The last day of observations, defined according
            to SMTN-032 (UTC-12hrs).
        creation_time : `Time`, optional
            The time the sequence was created, defaults no now.

        Returns
        -------
        visitseq_uuid : `UUID`
            The UUID of the new visit sequence.

        Notes
        -----
        This method is not normally used externally, but may be
        occasionally needed to add non-standard sequences of visits.
        Standard simulations and completed sets of visits should be
        registered with the ``record_simulation_metadata` and
        `record_completed_metadata` methods instead.
        """

        sha256 = compute_visits_sha256(visits)

        columns = [
            sql.Identifier("visitseq_sha256"),
            sql.Identifier("visitseq_label"),
            sql.Identifier("telescope"),
        ]
        data_placeholders = [
            sql.Placeholder("visitseq_sha256"),
            sql.Placeholder("visitseq_label"),
            sql.Placeholder("telescope"),
        ]
        data = {"visitseq_sha256": sha256, "visitseq_label": label, "telescope": telescope}

        if url is not None:
            columns.append(sql.Identifier("visitseq_url"))
            data_placeholders.append(sql.Placeholder("visitseq_url"))
            data["visitseq_url"] = url

        if first_day_obs is not None:
            columns.append(sql.Identifier("first_day_obs"))
            data_placeholders.append(sql.Placeholder("first_day_obs"))
            data["first_day_obs"] = _dayobs_to_date(first_day_obs)

        if last_day_obs is not None:
            columns.append(sql.Identifier("first_day_obs"))
            data_placeholders.append(sql.Placeholder("last_day_obs"))
            data["last_day_obs"] = _dayobs_to_date(last_day_obs)

        if creation_time is not None:
            assert not creation_time.masked
            assert creation_time.isscalar
            creation_datetime = creation_time.utc[0].datetime
            columns.append(sql.Identifier("creation_time"))
            data_placeholders.append(sql.Placeholder("creation_time"))
            data["creation_time"] = creation_datetime

        query = sql.SQL("INSERT INTO {}.{} ({}) VALUES ({}) RETURNING visitseq_uuid").format(
            sql.Identifier(self.metadata_db_schema),
            sql.Identifier(table),
            sql.SQL(", ").join(columns),
            sql.SQL(", ").join(data_placeholders),
        )

        result = self.query(query, data, commit=True)[0][0]
        return result

    def get_visitseq_metadata(
        self,
        visitseq_uuid: UUID,
        table: str = "visitseq",
    ) -> pd.Series:
        """Retrieve metadata for a visit sequence.

        Parameters
        ----------
        visitseq_uuid : `UUID`
            The UUID of the visit sequence to retrieve metadata for.
        table : `str`, optional
            The table in the metadata archive database in which to search.
            Must be one of "visitseq", "mixedvisitseq", "completed", or
            "simulations".
            Defaults to "visitseq".

        Returns
        -------
        visitseq : `pd.Series`
            A Pandas Series containing the metadata for the visit sequence.

        Raises
        ------
        ValueError
            If the table is not one of "visitseq", "mixedvisitseq",
            "completed", or "simulations".
        """
        if table not in {"visitseq", "mixedvisitseq", "completed", "simulations"}:
            raise ValueError()

        psycopg2_query = sql.SQL("SELECT * FROM {}.{} WHERE visitseq_uuid=%s").format(
            sql.Identifier(self.metadata_db_schema),
            sql.Identifier(table),
        )

        conn = None
        try:
            conn = self.pg_pool.getconn()
            text_query = psycopg2_query.as_string(conn)
        finally:
            if conn:
                self.pg_pool.putconn(conn)

        # pandas works better if the connection is made by sqlalchemy
        engine = sqlalchemy.create_engine("postgresql+psycopg2://", creator=self.pg_pool.getconn)
        with engine.connect() as sa_conn:
            visitseq = pd.read_sql(text_query, sa_conn, params=(visitseq_uuid,)).iloc[0, :]

        return visitseq

    def set_visitseq_url(self, table: str, visitseq_uuid: UUID, visitseq_url: str) -> None:
        """Update the URL for the visits file for a visit sequence.

        Parameters
        ----------
        table : `str`
            The table in the metadata archive database to update with the url.
            Must be one of "visitseq", "mixedvisitseq", "completed", or
            "simulations".
        visitseq_uuid : `UUID`
            The UUID of the visit sequence to update.
        visitseq_url : `str`
            The new URL for the visit sequence.

        Raises
        ------
        `ValueError`
            If the table is not one of "visitseq", "mixedvisitseq",
            "completed", or "simulations".
        """
        query = sql.SQL("UPDATE {}.{} SET visitseq_url={} WHERE visitseq_uuid={} RETURNING *;").format(
            sql.Identifier(self.metadata_db_schema),
            sql.Identifier(table),
            sql.Placeholder("visitseq_url"),
            sql.Placeholder("visitseq_uuid"),
        )
        data = {"visitseq_url": visitseq_url, "visitseq_uuid": visitseq_uuid}
        self.query(query, data, return_result=False, commit=True)

    def get_visitseq_url(self, visitseq_uuid: UUID) -> str:
        """Retrieve the URL for the file with the visits in a visit sequence.

        Parameters
        ----------
        visitseq_uuid : `UUID`
            The UUID of the visit sequence to search for.

        Returns
        -------
        url : `str`
            The URL for the visit sequence.

        Raises
        ------
        `ValueError`
            If no URL is found for the visit sequence.
        """
        query = sql.SQL("SELECT visitseq_url FROM {}.visitseq WHERE visitseq_uuid={}").format(
            sql.Identifier(self.metadata_db_schema),
            sql.Placeholder("visitseq_uuid"),
        )
        data = {"visitseq_uuid": visitseq_uuid}
        response = self.query(query, data)
        if len(response) < 1:
            raise ValueError(f"No URL for {visitseq_uuid} found")
        assert len(response) == 1, f"Datatabase has too many visit sequinces with UUID={visitseq_uuid}"
        return response[0][0]

    def record_simulation_metadata(
        self,
        visits: pd.DataFrame,
        label: str,
        telescope: str = "simonyi",
        url: str | None = None,
        first_day_obs: date | int | str | None = None,
        last_day_obs: date | int | str | None = None,
        creation_time: Time | None = None,
        scheduler_version: str | None = None,
        config_url: str | None = None,
        conda_env_sha256: bytes | None = None,
        parent_visitseq_uuid: UUID | None = None,
        sim_runner_kwargs: dict | None = None,
        parent_last_dayobs: str | date | int | None = None,
    ) -> UUID:
        """Record metadata for a new simulation sequence.

        Parameters
        ----------
        visits : pd.DataFrame
            A DataFrame of visits, with column names following those
            in consdb.
        label : str
            A label for the sequence.
        telescope : `str`
            The telescope used, either "simonyi" or "auxtel".
            Defaults to "simonyi".
        url : `str`, optional
            A URL for the file from thich the table of visits can
            be download.
        first_day_obs : `date` or `int` or `str`, optional
            The first night of observations, defined according
            to SMTN-032 (UTC-12hrs).
        last_day_obs : `date` or `int` or `str`, optional
            The last day of observations, defined according
            to SMTN-032 (UTC-12hrs).
        creation_time : `Time`, optional
            The time the sequence was created, defaults to now.
        scheduler_version : `str`, optional
            Version of ``rubin_scheduler`` used to run the simulation,
            by default None.
        config_url : `str`, optional
            URL for the config script, perhaps on github,
            by default `None`.
        conda_env_sha256 : `bytes`, optional
            SHA256 hash of output of ``conda list --json``, by default None.
        parent_visitseq_uuid : `UUID`, optional
            UUID of visitseq loaded into scheduler before running,
            by default `None`.
        sim_runner_kwargs : `dict`, optional
            Arguments to ``sim_runner`` as a `dict`, by default `None`.
        parent_last_dayobs : `str`, optional
            day_obs of last visit loaded into scheduler before running,
            by default `None`.

        Returns
        -------
        visitseq_uuid : UUID
            The UUID of the new visit sequence.
        """
        # I would have preferred to use kwargs here, but was not
        # able to get type checking happy with it.
        visitseq_uuid = self.record_visitseq_metadata(
            visits,
            label,
            telescope=telescope,
            table="simulations",
            url=url,
            first_day_obs=first_day_obs,
            last_day_obs=last_day_obs,
            creation_time=creation_time,
        )

        def make_set_clause(column: str) -> sql.Composed:
            set_clause = sql.SQL("{} = {}").format(sql.Identifier(column), sql.Placeholder(column))
            return set_clause

        set_clauses = []
        data = {}
        if scheduler_version is not None:
            set_clauses.append(make_set_clause("scheduler_version"))
            data["scheduler_version"] = scheduler_version

        if config_url is not None:
            set_clauses.append(make_set_clause("config_url"))
            data["config_url"] = config_url

        if conda_env_sha256 is not None:
            set_clauses.append(make_set_clause("conda_env_sha256"))
            data["conda_env_sha256"] = conda_env_sha256

        if parent_visitseq_uuid is not None:
            set_clauses.append(make_set_clause("parent_visitseq_uuid"))
            data["parent_visitseq_uuid"] = str(parent_visitseq_uuid)

        if sim_runner_kwargs is not None:
            sim_runner_munged_kwargs = {}
            for key in sim_runner_kwargs:
                try:
                    # Throws a TypeError if it cannot convert it to json
                    json_result = json.dumps(sim_runner_kwargs[key])
                    json_failed = len(json_result) > JSON_DUMP_LIMIT
                except TypeError:
                    # Cannot convert to json, just store a string
                    # representation instead.
                    json_failed = True

                # if we can save an argument as serealized json in reasonable
                # space, do, otherwise store a (possibly truncated) result of
                # repr.
                sim_runner_munged_kwargs[key] = (
                    ("Not json serializable", repr(sim_runner_kwargs[key])[:JSON_DUMP_LIMIT])
                    if json_failed
                    else sim_runner_kwargs[key]
                )

            set_clauses.append(make_set_clause("sim_runner_kwargs"))
            data["sim_runner_kwargs"] = psycopg2.extras.Json(sim_runner_munged_kwargs)

        if parent_last_dayobs is not None:
            set_clauses.append(make_set_clause("parent_last_dayobs"))
            data["parent_last_dayobs"] = parent_last_dayobs

        num_columns_to_update = len(set_clauses)

        if num_columns_to_update > 0:
            query = sql.SQL("UPDATE {}.simulations SET {} WHERE visitseq_uuid={} RETURNING *;").format(
                sql.Identifier(self.metadata_db_schema),
                sql.SQL(", ").join(set_clauses),
                sql.Placeholder("visitseq_uuid"),
            )
            data["visitseq_uuid"] = visitseq_uuid

            conn = None
            try:
                conn = self.pg_pool.getconn()
                cursor = conn.cursor()
                cursor.execute(query, data)
                result = cursor.fetchall()

                # Be extra cautious, and check that everything looks
                # reasonable before commiting the update.
                assert len(result) == 1
                cursor.execute("COMMIT;")
            finally:
                if conn:
                    self.pg_pool.putconn(conn)

        return visitseq_uuid

    def record_completed_metadata(
        self,
        visits: pd.DataFrame,
        label: str,
        telescope: str = "simonyi",
        url: str | None = None,
        first_day_obs: date | int | str | None = None,
        last_day_obs: date | int | str | None = None,
        creation_time: Time | None = None,
        query: str | None = None,
    ) -> UUID:
        """Record metadata for a sequence of visits queried from
        the consdb.

        Parameters
        ----------
        visits : `pd.DataFrame`
            A DataFrame of visits, with column names following those
            in consdb.
        label : `str`
            A label for the sequence.
        telescope : `str`
            The telescope used, either "simonyi" or "auxtel".
            Defaults to "simonyi".
        url : `str`, optional
            A URL for the file from thich the table of visits can
            be download.
        first_day_obs : `date` or `int` or `str`, optional
            The first night of observations, defined according
            to SMTN-032 (UTC-12hrs).
        last_day_obs : `date` or `int` or `str`, optional
            The last day of observations, defined according
            to SMTN-032 (UTC-12hrs).
        creation_time : `Time`, optional
            The time the sequence was created, defaults no now.
        query : `str`, optional
            The query to the consdb used to get the completed
            visits, by default None.

        Returns
        -------
        visitseq_uuid : `UUID`
            The UUID of the new visit sequence.
        """

        # I would have preferred to use kwargs here, but was not
        # able to get type checking happy with it.
        visitseq_uuid = self.record_visitseq_metadata(
            visits,
            label,
            telescope=telescope,
            table="completed",
            url=url,
            first_day_obs=first_day_obs,
            last_day_obs=last_day_obs,
            creation_time=creation_time,
        )

        if query is not None:
            composed_query = sql.SQL(
                "UPDATE {}.completed SET query={} WHERE visitseq_uuid={} RETURNING *"
            ).format(
                sql.Identifier(self.metadata_db_schema),
                sql.Placeholder("query"),
                sql.Placeholder("visitseq_uuid"),
            )
            data = {"query": query, "visitseq_uuid": visitseq_uuid}

            conn = None
            try:
                conn = self.pg_pool.getconn()
                cursor = conn.cursor()
                cursor.execute(composed_query, data)
                result = cursor.fetchall()

                # Be extra cautious, and check that everything looks
                # reasonable before commiting the update.
                assert len(result) == 1
                cursor.execute("COMMIT;")
            finally:
                if conn:
                    self.pg_pool.putconn(conn)

        return visitseq_uuid

    def record_mixed_metadata(
        self,
        visits: pd.DataFrame,
        label: str,
        last_early_day_obs: date | int | str,
        first_late_day_obs: date | int | str,
        early_parent_uuid: UUID,
        late_parent_uuid: UUID,
        telescope: str = "simonyi",
        url: str | None = None,
        first_day_obs: date | int | str | None = None,
        last_day_obs: date | int | str | None = None,
        creation_time: Time | None = None,
    ) -> UUID:
        """Record metadata for a new mixed visit sequence.

        Parameters
        ----------
        visits : `pd.DataFrame`
            A DataFrame of visits, with column names following those
            in consdb.
        label : `str`
            A label for the sequence.
        last_early_day_obs : `date` or `int` or `str`
            The last day obs of the early parent.
        first_late_day_obs : `date` or `int` or `str`
            The first day obs of the late parent.
        early_parent_uuid : `UUID`
            The UUID of the early parent.
        late_parent_uuid : `UUID`
            The UUID of the late parent.
        telescope : `str`, optional
            The telescope used, either "simonyi" or "auxtel".
            Defaults to "simonyi".
        first_day_obs : `date` or `int` or `str` or `None`, optional
            The first night of observations, defined according
            to SMTN-032 (UTC-12hrs).
        last_day_obs : `date` or `int` or `str` or `None`, optional
            The last day of observations, defined according
            to SMTN-032 (UTC-12hrs).
        creation_time : `Time` or `None`, optional
            The time the sequence was created, defaults to now.

        Returns
        -------
        visitseq_uuid : `UUID`
            The UUID of the new visit sequence.
        """
        visitseq_uuid = self.record_visitseq_metadata(
            visits,
            label,
            telescope=telescope,
            table="mixedvisitseq",
            url=url,
            first_day_obs=first_day_obs,
            last_day_obs=last_day_obs,
            creation_time=creation_time,
        )

        update_query = sql.SQL(
            """UPDATE {}.mixedvisitseq
                   SET last_early_day_obs={},
                       first_late_day_obs={},
                       early_parent_uuid={},
                       late_parent_uuid={}\
                   WHERE visitseq_uuid={} RETURNING *
                """
        ).format(
            sql.Identifier(self.metadata_db_schema),
            sql.Placeholder("last_early_day_obs"),
            sql.Placeholder("first_late_day_obs"),
            sql.Placeholder("early_parent_uuid"),
            sql.Placeholder("late_parent_uuid"),
            sql.Placeholder("visitseq_uuid"),
        )

        data = {
            "last_early_day_obs": _dayobs_to_date(last_early_day_obs),
            "first_late_day_obs": _dayobs_to_date(first_late_day_obs),
            "early_parent_uuid": early_parent_uuid,
            "late_parent_uuid": late_parent_uuid,
            "visitseq_uuid": visitseq_uuid,
        }

        conn = None
        try:
            conn = self.pg_pool.getconn()
            cursor = conn.cursor()
            cursor.execute(update_query, data)
            result = cursor.fetchall()

            # Be extra cautious, and check that everything looks
            # reasonable before commiting the update.
            assert len(result) == 1
            cursor.execute("COMMIT;")
        finally:
            if conn:
                self.pg_pool.putconn(conn)

        return visitseq_uuid

    def is_tagged(self, visitseq_uuid: UUID, tag: str) -> bool:
        """Return whether a visit sequence is tagged with a given tag.

        Parameters
        ----------
        visitseq_uuid : `UUID`
            The UUID of the visit sequence.
        tag : `str`
            The tag to check for.

        Returns
        -------
        is_tagged : `bool`
            True if the visit sequence is tagged with the given tag,
            False otherwise.
        """
        query = sql.SQL("SELECT COUNT(*)>=1 FROM {}.tags WHERE visitseq_uuid={} AND TAG={}").format(
            sql.Identifier(self.metadata_db_schema), sql.Placeholder("visitseq_uuid"), sql.Placeholder("tag")
        )
        data = {"visitseq_uuid": visitseq_uuid, "tag": tag}
        is_tagged = self.query(query, data)[0][0] > 0
        return is_tagged

    def tag(self, visitseq_uuid: UUID, *tags: str) -> None:
        """Tag a visit sequence with one or more tags.

        Parameters
        ----------
        visitseq_uuid : `UUID`
            The UUID of the visit sequence.
        *tags : `str`
            One or more tags to add to the visit sequence.
        """
        query = sql.SQL("INSERT INTO {}.tags (visitseq_uuid, tag) VALUES ({}, {})").format(
            sql.Identifier(self.metadata_db_schema), sql.Placeholder("visitseq_uuid"), sql.Placeholder("tag")
        )
        data = {"visitseq_uuid": visitseq_uuid}
        for tag in tags:
            if not self.is_tagged(visitseq_uuid, tag):
                data["tag"] = tag
                self.query(query, data, commit=True, return_result=False)

    def untag(self, visitseq_uuid: UUID, tag: str) -> None:
        """Untag a visit sequence with a given tag.

        Parameters
        ----------
        visitseq_uuid : `UUID`
            The UUID of the visit sequence.
        tag : `str`
            The tag to remove.

        Raises
        ------
        `ValueError`
            If the visit sequence is not tagged with the given tag.
        """
        if not self.is_tagged(visitseq_uuid, tag):
            raise ValueError(f"Visit sequence {visitseq_uuid} is not tagged with {tag}")

        query = sql.SQL("DELETE FROM {}.tags WHERE visitseq_uuid={} AND tag={}").format(
            sql.Identifier(self.metadata_db_schema),
            sql.Placeholder("visitseq_uuid"),
            sql.Placeholder("tag"),
        )
        data = {"visitseq_uuid": visitseq_uuid, "tag": tag}
        self.query(query, data, commit=True, return_result=False)

    def comment(self, visitseq_uuid: UUID, comment: str, author: str | None = None) -> None:
        """Attach a comment to a visit sequence in the
        metadata database..

        Parameters
        ----------
        visitseq_uuid : `UUID`
            The UUID of the visit sequence to comment on.
        comment : `str`
            The comment to attach.
        author : `str`, optional
            The author of the comment.
            Defaults to `None`.
        """
        comment_time = Time.now().utc.datetime
        query = ""
        data = {"visitseq_uuid": visitseq_uuid, "comment_time": comment_time, "comment": comment}
        if author is None:
            query = sql.SQL(
                "INSERT INTO {}.comments (visitseq_uuid, comment_time, comment) VALUES ({}, {}, {})"
            ).format(
                sql.Identifier(self.metadata_db_schema),
                sql.Placeholder("visitseq_uuid"),
                sql.Placeholder("comment_time"),
                sql.Placeholder("comment"),
            )
        else:
            query = sql.SQL(
                "INSERT INTO {}.comments (visitseq_uuid, comment_time, author, comment)"
                + " VALUES ({}, {}, {}, {})"
            ).format(
                sql.Identifier(self.metadata_db_schema),
                sql.Placeholder("visitseq_uuid"),
                sql.Placeholder("comment_time"),
                sql.Placeholder("author"),
                sql.Placeholder("comment"),
            )
            data["author"] = author

        self.query(query, data, commit=True, return_result=False)

    def get_comments(self, visitseq_uuid: UUID) -> pd.DataFrame:
        """Retrieve all comments attached to a specific visit sequence.

        Parameters
        ----------
        visitseq_uuid : `UUID`
            The UUID of the visit sequence whose comments should be fetched.

        Returns
        -------
        comments: `pd.DataFrame`
            A DataFrame containing the comments for the requested visit
            sequence. The DataFrame has the following columns:

            ``"visitseq_uuid"``
                The ID of the visit sequence.
            ``"comment_time"``
                Timestamp of the comment.
            ``"author"``
                Name of the user who added the comment; may be ``None``.
            ``"comment"``
                The comment text.

            If the visit sequence has no comments, an empty DataFrame with the
            above column names is returned.

        """
        psycopg2_query = sql.SQL("SELECT * FROM {}.comments WHERE visitseq_uuid = %s").format(
            sql.Identifier(self.metadata_db_schema)
        )

        conn = None
        try:
            conn = self.pg_pool.getconn()
            text_query = psycopg2_query.as_string(conn)
        finally:
            if conn:
                self.pg_pool.putconn(conn)

        # pandas works better if the connection is made by sqlalchemy
        engine = sqlalchemy.create_engine("postgresql+psycopg2://", creator=self.pg_pool.getconn)
        with engine.connect() as sa_conn:
            comments = pd.read_sql(text_query, sa_conn, params=(visitseq_uuid,))

        # sqlalchemy seems to mess up the dtypes. Fix them.
        comments["comment"] = comments["comment"].astype("string")

        return comments

    def register_file(
        self,
        visitseq_uuid: UUID,
        file_type: str,
        file_sha256: bytes,
        location: str | ResourcePath,
        update: bool = False,
    ) -> None:
        """Register a file in the visit sequence metadata database.

        Parameters
        ----------
        visitseq_uuid : `UUID`
            The UUID of the visit sequence with which the file should be
            associated.
        file_type : `str`
            The handle for the file type.
            The ``"visits"`` file type is not allowed
            because visit tables are recorded using `set_visitseq_url`.
        file_sha256 : `bytes`
            The hash of the file contents.
        location : `str` or `lsst.resources.ResourcePath`
            The location of the file to be registered.
        update : `bool`, optional
            If ``True`` and a record for the same ``visitseq_uuid`` and
            ``file_type`` already exists, the existing row will be updated with
            the new SHA‑256 hash and URL.  If ``False`` and a record already
            exists a ``ValueError`` is raised.

        Raises
        ------
        `ValueError`
            If ``file_type`` is ``"visits"`` (use `set_visitseq_url` instead),
            or if ``update`` is ``False`` and a record already exists for the
            given ``visitseq_uuid`` and ``file_type``.
        `TypeError`
            If ``location`` is neither a string nor a `ResourcePath`.
        """
        if file_type == "visits":
            raise ValueError("Use set_visitseq_url to register sets of visits themselves")

        file_url: str = ""
        match location:
            case ResourcePath():
                file_url = location.geturl()
            case str():
                file_url = location
            case _:
                raise ValueError("Unrecognised location type")

        # Check whether the file is already registered
        already_exists = False
        try:
            self.get_file_url(visitseq_uuid, file_type)
            already_exists = True
        except ValueError:
            already_exists = False

        if already_exists:
            if not update:
                raise ValueError("A file of type {file_type} is already registered for {visitseq_uuid}")

            query = sql.SQL(
                "UPDATE {}.files SET file_sha256={}, file_url={} WHERE visitseq_uuid={} AND file_type={}"
            ).format(
                sql.Identifier(self.metadata_db_schema),
                sql.Placeholder("file_sha256"),
                sql.Placeholder("file_url"),
                sql.Placeholder("visitseq_uuid"),
                sql.Placeholder("file_type"),
            )
        else:
            query = sql.SQL(
                "INSERT INTO {}.files (visitseq_uuid, file_type, file_sha256, file_url)"
                + " VALUES ({}, {}, {}, {})"
            ).format(
                sql.Identifier(self.metadata_db_schema),
                sql.Placeholder("visitseq_uuid"),
                sql.Placeholder("file_type"),
                sql.Placeholder("file_sha256"),
                sql.Placeholder("file_url"),
            )

        data = {
            "visitseq_uuid": visitseq_uuid,
            "file_type": file_type,
            "file_sha256": file_sha256,
            "file_url": file_url,
        }

        self.query(query, data, commit=True, return_result=False)

    def get_file_url(self, visitseq_uuid: UUID, file_type: str) -> str:
        """Return the URL for a registered file of a given type and
        visit sequence UUID.

        Parameters
        ----------
        visitseq_uuid : `UUID`
            The UUID of the visit sequence with which the desired
            file is associated.
        file_type : `str`
            Handle for the file type.

        Returns
        -------
        url : `str`
            The URL pointing to the requested file.
        """
        if file_type == "visits":
            # It's as easy to just do it as it is to raise
            # an exception
            return self.get_visitseq_url(visitseq_uuid)

        query = sql.SQL("SELECT file_url FROM {}.files WHERE visitseq_uuid={} AND file_type={}").format(
            sql.Identifier(self.metadata_db_schema),
            sql.Placeholder("visitseq_uuid"),
            sql.Placeholder("file_type"),
        )
        data = {"visitseq_uuid": visitseq_uuid, "file_type": file_type}
        result = self.query(query, data, return_result=True)
        if len(result) < 1:
            raise ValueError(f"No URLs found for {file_type} for visitseq {visitseq_uuid}")
        if len(result) > 1:
            raise ValueError(f"Too many URLs found for {file_type} for visitseq {visitseq_uuid}!")

        url = result[0][0]
        return url

    def get_file_sha256(self, visitseq_uuid: UUID, file_type: str) -> bytes:
        """Retrieve the SHA‑256 digest of a registered file.

        Parameters
        ----------
        visitseq_uuid : `UUID`
            The UUID of the visit sequence the file is associated with.
        file_type : `str`
            Handle for the file type.

        Returns
        -------
        file_sha256 : `bytes`
            The raw SHA‑256 digest of the file contents.  The value
            is returned as a 32‑byte object (the output of
            ``hashlib.sha256``).
        """

        query = sql.SQL("SELECT file_sha256 FROM {}.files WHERE visitseq_uuid={} AND file_type={}").format(
            sql.Identifier(self.metadata_db_schema),
            sql.Placeholder("visitseq_uuid"),
            sql.Placeholder("file_type"),
        )
        data = {"visitseq_uuid": visitseq_uuid, "file_type": file_type}
        result = self.query(query, data, return_result=True)
        if len(result) < 1:
            raise ValueError(f"No URLs found for {file_type} for visitseq {visitseq_uuid}")
        if len(result) > 1:
            raise ValueError(f"Too many URLs found for {file_type} for visitseq {visitseq_uuid}!")

        file_sha256 = result[0][0].tobytes()
        return file_sha256

    def record_nightly_stats(
        self,
        visitseq_uuid: UUID,
        visits: pd.DataFrame,
        columns: Tuple[str] = ("s_ra", "s_dec", "sky_rotation"),
    ) -> pd.DataFrame:
        """Compute and store statistics by night for a visit sequence.

        Parameters
        ----------
        visitseq_uuid : `UUID`
            The UUID of the visit sequence whose statistics are being
            recorded.
        visits : `pd.DataFrame`
            The full table of visits.  The DataFrame must contain at
            least the columns listed in ``columns`` and a ``day_obs``
            column used to group the statistics by night.
        columns : `tuple` of `str`, optional
            Names of the columns to aggregate.  By default the
            celestial coordinates (`s_ra`, `s_dec`) and the sky
            rotation angle (`sky_rotation`) are used.

        Returns
        -------
        stats_df : `pd.DataFrame`
            A DataFrame containing the nightly statistics that were
            written to the database.  The columns are:

            ``"day_obs"``
                The observation day.
            ``"value_name"``
                The column name from ``columns``
                that the statistic refers to.
            ``"p05"``
                5th percentile.
            ``"q1"``
                25th percentile (first quartile).
            ``"median"``
                50th percentile (median).
            ``"q3"``
                75th percentile (third quartile).
            ``"p95"``
                95th percentile.
            ``"visitseq_uuid"``
                The UUID of the sequence.
            ``"accumulated"``
                Always ``False`` for data written by
                this method.
        """
        stats_df = compute_nightly_stats(visits, columns)
        self.insert_nightly_stats(visitseq_uuid, stats_df)

        return stats_df

    def insert_nightly_stats(self, visitseq_uuid: UUID, nightly_stats: pd.DataFrame) -> None:
        """Instert by-night statistics into associoted with
        a visit sequence into the visit sequence metadata database.

        Parameters
        ----------
        visitseq_uuid : `UUID`
            The UUID of the visit sequence whose statistics are being
            recorded.
        stats_df : `pd.DataFrame`
            A DataFrame containing the nightly statistics that were
            written to the database.  The columns are:

            ``"day_obs"``
                The observation day.
            ``"value_name"``
                The value for which statistics were computed.
            ``"p05"``
                5th percentile.
            ``"q1"``
                25th percentile (first quartile).
            ``"median"``
                50th percentile (median).
            ``"q3"``
                75th percentile (third quartile).
            ``"p95"``
                95th percentile.
            ``"visitseq_uuid"``
                The UUID of the sequence.
            ``"accumulated"``
                Always ``False`` for data written by
                this method.
        """
        # We are adding a column, but do not want
        # to change the original df, so
        # create a copy.
        stats_df = nightly_stats.copy()
        stats_df["visitseq_uuid"] = visitseq_uuid

        # pandas to_sql fails with the native psycopg2, only
        # works with sqlalchemy
        engine = sqlalchemy.create_engine("postgresql+psycopg2://", creator=self.pg_pool.getconn)
        with engine.connect() as conn:
            num_rows_added = stats_df.to_sql(
                "nightly_stats", conn, schema=self.metadata_db_schema, if_exists="append", index=False
            )
            assert num_rows_added == len(stats_df)

    def compute_conda_env(self) -> tuple:
        """Find the current conda environment and its SHA‑256 hash.

        Returns
        -------
        tuple
            A two‑element tuple ``(conda_env_hash, conda_env_json)`` where
            ``conda_env_hash`` is a `bytes` instance containing the
            SHA‑256 digest, and ``conda_env_json`` is a `str` holding
            the JSON output of ``conda list --json``.
        """

        conda_list_result = subprocess.run(
            ["conda", "list", "--json"], capture_output=True, text=True, check=True
        )
        conda_env_json = conda_list_result.stdout
        conda_env_hash = bytes.fromhex(hashlib.sha256(conda_env_json.encode()).hexdigest())
        return conda_env_hash, conda_env_json

    def conda_env_is_saved(self, conda_env_hash: bytes) -> bool:
        """Check whether a conda environment with the given hash is already
        stored in the database.

        Parameters
        ----------
        conda_env_hash : `bytes`
            The SHA‑256 digest of the conda environment JSON representation.
            This value is expected to be the same output returned by
            `compute_conda_env`.

        Returns
        -------
        env_exists : `bool`
            ``True`` if a row with the supplied hash exists in the
            ``conda_env`` table; ``False`` otherwise.
        """
        query = sql.SQL("SELECT EXISTS(SELECT true from {}.conda_env WHERE conda_env_hash={})").format(
            sql.Identifier(self.metadata_db_schema), sql.Placeholder("conda_env_hash")
        )
        data = {"conda_env_hash": conda_env_hash}
        result = self.query(query, data, commit=False, return_result=True)
        env_exists = result[0][0]
        assert isinstance(env_exists, bool)
        return env_exists

    def record_conda_env(self) -> bytes:
        """Record the current Conda environment in the database.

        Returns
        -------
        conda_env_hash : `bytes`
            The SHA‑256 digest of the Conda environment JSON
            representation.

        Notes
        -----
        * The method does **not** return the JSON representation; if
          that is required the caller can obtain it via
          `compute_conda_env`.
        * No arguments are taken; the method operates on the active
          Conda environment at runtime.
        """
        conda_env_hash, conda_env_json = self.compute_conda_env()

        if self.conda_env_is_saved(conda_env_hash):
            warnings.warn("Conda env with hash already exists, not saving again.")
            return conda_env_hash

        query = sql.SQL("INSERT INTO {}.conda_env (conda_env_hash, conda_env) VALUES ({}, {})").format(
            sql.Identifier(self.metadata_db_schema),
            sql.Placeholder("conda_env_hash"),
            sql.Placeholder("conda_env"),
        )
        data = {"conda_env_hash": conda_env_hash, "conda_env": conda_env_json}
        self.query(query, data, commit=True, return_result=False)
        return conda_env_hash

    def construct_base_resource_path(
        self, telescope: str, creation_time: Time, visitseq_uuid: UUID
    ) -> ResourcePath:
        # Record in path according to datetime timezone (UTC-12)
        visitseq_base_rp = (
            self.archive_base.join(telescope)
            .join(Time(creation_time.mjd - 0.5, format="mjd").datetime.date().isoformat())
            .join(str(visitseq_uuid), forceDirectory=True)
        )
        return visitseq_base_rp

    def _write_data_to_archive(self, visitseq_uuid: UUID, content: bytes, file_name: str) -> ResourcePath:
        visitseq_metadata = self.get_visitseq_metadata(visitseq_uuid)
        visitseq_uuid = visitseq_metadata["visitseq_uuid"]
        telescope = visitseq_metadata["telescope"]
        creation_time = Time(visitseq_metadata["creation_time"])
        visitseq_base_rp = self.construct_base_resource_path(telescope, creation_time, visitseq_uuid)

        # Make sure the base for the visit sequence exists
        if not visitseq_base_rp.exists():
            visitseq_base_rp.mkdir()
            LOGGER.debug(f"Created {visitseq_base_rp}.")

        destination_rp = visitseq_base_rp.join(file_name)
        LOGGER.debug(f"Writing {destination_rp}.")
        destination_rp.write(content)
        LOGGER.debug(f"Wrote {destination_rp}.")
        return destination_rp

    def _write_file_to_archive(self, visitseq_uuid: UUID, origin: str | Path) -> Tuple[ResourcePath, bytes]:
        file_path = origin if isinstance(origin, Path) else Path(origin)
        with open(file_path, "rb") as origin_io:
            content = origin_io.read()

        content_sha256 = bytes.fromhex(hashlib.sha256(content).hexdigest())
        destination_rp = self._write_data_to_archive(visitseq_uuid, content, file_path.name)
        return destination_rp, content_sha256

    def _find_visitseq_table(self, visitseq_uuid: UUID) -> str:
        # Figure out which child table the visitseq_uuid is in by
        # querying each in turn.

        for table in ("mixedvisitseq", "completed", "simulations", "visitseq"):
            query = sql.SQL("SELECT COUNT(*) FROM {}.{} WHERE visitseq_uuid={};").format(
                sql.Identifier(self.metadata_db_schema),
                sql.Identifier(table),
                sql.Placeholder("visitseq_uuid"),
            )
            data = {"visitseq_uuid": visitseq_uuid}
            result = self.query(query, data)
            assert isinstance(result[0][0], int)
            num_rows: int = result[0][0]
            if num_rows > 0:
                return table

        raise ValueError("Visit sequence not found.")

    def archive_file(
        self, visitseq_uuid: UUID, origin: str | Path, file_type: str, update: bool = False
    ) -> ResourcePath:
        location, file_sha256 = self._write_file_to_archive(visitseq_uuid, origin)
        if file_type == "visits":
            visitseq_type = self._find_visitseq_table(visitseq_uuid)
            self.set_visitseq_url(visitseq_type, visitseq_uuid, location.geturl())
        else:
            self.register_file(visitseq_uuid, file_type, file_sha256, location, update=update)

        return location


def compute_nightly_stats(
    visits: pd.DataFrame, columns: Tuple[str] = ("s_ra", "s_dec", "sky_rotation")
) -> pd.DataFrame:
    stats_df = (
        visits.groupby("day_obs")[list(columns)]
        .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        .stack(level=0)
        .reset_index()
        .rename(
            columns={
                "level_1": "value_name",
                "5%": "p05",
                "25%": "q1",
                "50%": "median",
                "75%": "q3",
                "95%": "p95",
            }
        )
    )
    stats_df["accumulated"] = False
    return stats_df


@click.group()
def visitseq() -> None:
    """visitseq command line interface."""
    pass


@visitseq.command()
@click.argument("archive")
@click.argument("uuid", type=click.UUID)
@click.argument("comment")
def comment(uuid: UUID, comment: str, archive: str) -> None:
    test_database = {"database": "opsim_log", "host": "134.79.23.205", "schema": "ehntest"}
    vsarch = VisitSequenceArchive(test_database, archive)
    vsarch.comment(uuid, comment)


if __name__ == "__main__":
    visitseq()
