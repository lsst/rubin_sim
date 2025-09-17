import hashlib
import json
import logging
import os
import subprocess
import warnings
from datetime import date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
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
from lsst.resources import ResourcePath
from psycopg2 import sql
from rubin_scheduler.scheduler.utils import SchemaConverter

VSARCHIVE_PGDATABASE = "opsim_log"
VSARCHIVE_PGHOST = "134.79.23.205"
ARCHIVE_URL = "test_archive"
SQLITE_EXTINSIONS = {".db", ".sqlite", ".sqlite3", ".db3"}

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


class VisitSequenceArchiveMetadata:
    """Interface to metadata database that tracks sequences
    of visits.

    Parameters
    ----------
    metadata_db_kwargs: `Mapping`
        A dictionary or other mapping defining the connection
        parameters for connecting to the postgresql database
        that holds the sequence metadata. Keys are passed as keyword
        arguments to `psycopg2.pool.SimppleConnectionPool`.
    metadata_db_schema: `str`
        The schema in the database holding the metadata.

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
    This class is an interface to the postgresql metadata database.
    """

    def __init__(
        self,
        metadata_db_kwargs: Mapping | None = None,
        metadata_db_schema: str = "ehntest",
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

    def pd_read_sql(
        self, query_template: str, sql_params: list[sql.Composable], query_params: tuple
    ) -> pd.DataFrame:
        """Execute a SQL query using the internal PostgreSQL connection pool
        and return the results as a pandas DataFrame.

        Parameters
        ----------
        query_template : `str`
            A query template containing ``%s`` placeholders for positional
            parameters.  These placeholders are replaced by ``sql_params`` via
            `psycopg2.sql.SQL`.
        sql_params : `list[sql.Composable]`
            Elements that will be substituted into the template (e.g. table or
            schema identifiers).  They must be safe for use in a SQL statement
            and are not the same as the query parameters bound to the query.
            To make type checkers happy, you may need to explictly declare
            variables passed as ``list[sql.Composable]``.
        query_params : tuple
            Positional parameters that will be bound to the rendered SQL
            string.  These are the values that actually get passed to the
            database.

        Returns
        -------
        pandas.DataFrame
            The result set of the query, wrapped in a DataFrame.

        Raises
        ------
        psycopg2.Error
            Propagated if the underlying query fails.

        Notes
        -----
        This works around limitations in pandas when dealing with
        complex datatypes in raw postgresql connections.
        """
        # Pandas sometimes has trouble with postgresql when working with it
        # directly, so interact with it by way of sqlalchemy.
        # But, we need to use postgresql's tools for inserting identifiers
        # like the schema name, followed by the sqlalchemy templating
        # for actual parameters.
        query = sql.SQL(query_template).format(*sql_params)
        conn = None
        try:
            conn = self.pg_pool.getconn()
            text_query = query.as_string(conn)
        finally:
            if conn:
                self.pg_pool.putconn(conn)

        # Use the postgresql pool of connections.
        engine = sqlalchemy.create_engine("postgresql+psycopg2://", creator=self.pg_pool.getconn)
        with engine.connect() as sa_conn:
            df = pd.read_sql(text_query, sa_conn, params=query_params)

        return df

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
            try:
                creation_datetime = creation_time.utc[0].datetime
            except AttributeError:
                # astropy Times, even those for which .isscalar
                # is true, can sometimes need to be indexed
                # by 0 to get a true scalar, and sometimes
                # cannot be.
                creation_datetime = creation_time.utc.datetime
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

        query_template = "SELECT * FROM {}.{} WHERE visitseq_uuid=%s"
        sql_params: list[sql.Composable] = [sql.Identifier(self.metadata_db_schema), sql.Identifier(table)]
        query_params = (visitseq_uuid,)
        visitseq = self.pd_read_sql(query_template, sql_params, query_params).iloc[0, :]
        return visitseq

    def set_visitseq_url(self, visitseq_uuid: UUID, visitseq_url: str) -> None:
        """Update the URL for the visits file for a visit sequence.

        Parameters
        ----------
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
        table = self._find_visitseq_table(visitseq_uuid)
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

    def update_visitseq_metadata(self, visitseq_uuid: UUID, field: str, value: object) -> None:
        """Update a single metadata field for a visit sequence.
        Automatically finds the table in which to make the update.

        Parameters
        ----------
        visitseq_uuid : `UUID`
            The unique identifier of the visit sequence to update.
        field : `str`
            The name of the column in the metadata table to modify.
        value : `object`
            The new value to assign to ``field``.  The type must be
            compatible with the column's database type.
        """
        table = self._find_visitseq_table(visitseq_uuid)
        query = sql.SQL("UPDATE {}.{} SET {}={} WHERE visitseq_uuid={};").format(
            sql.Identifier(self.metadata_db_schema),
            sql.Identifier(table),
            sql.Identifier(field),
            sql.Placeholder("value"),
            sql.Placeholder("visitseq_uuid"),
        )
        data = {"value": value, "visitseq_uuid": visitseq_uuid}
        self.query(query, data, return_result=False, commit=True)

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
        query_template = "SELECT * FROM {}.comments WHERE visitseq_uuid = %s"
        sql_params: list[sql.Composable] = [sql.Identifier(self.metadata_db_schema)]
        query_params = (visitseq_uuid,)
        comments = self.pd_read_sql(query_template, sql_params, query_params)

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

    def query_nightly_stats(self, visitseq_uuid: UUID) -> pd.DataFrame:
        """Query the visit sequence metadata database for
         nightly statistics.

        Parameters
        ----------
        visitseq_uuid : `UUID`
            The UUID of the visit sequence whose statistics are being
            recorded.

        Returns
        -------
        stats_df : `pd.DataFrame`
            A table of nightly statistics.
        """
        query_template = "SELECT * FROM {}.nightly_stats WHERE visitseq_uuid=%s"
        sql_params: list[sql.Composable] = [sql.Identifier(self.metadata_db_schema)]
        query_params = (visitseq_uuid,)
        stats_df = self.pd_read_sql(query_template, sql_params, query_params)
        return stats_df

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

    def record_conda_env(self, conda_env_hash: bytes, conda_env_json: str) -> bytes:
        """Record the current Conda environment in the database.

        Parameters
        ----------
        conda_env_hash : `bytes`
            The SHA‑256 digest of the Conda environment JSON
            representation.
        conda_env_json : `str`
            The json representing the conda environment.
        """
        conda_env_hash, conda_env_json = compute_conda_env()

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

    def sims_on_nights(
        self,
        first_day_obs: str | None = None,
        last_day_obs: str | None = None,
        tags: tuple[str] = ("prenight", "ideal", "nominal"),
        telescope: str = "simonyi",
        max_simulation_age: int = 2,
    ) -> pd.DataFrame:
        if len(tags) > 0:
            tags_json = json.dumps(list(tags))
            query_template = """
            SELECT *
            FROM {}.simulations_extra
            WHERE %s BETWEEN first_day_obs AND last_day_obs
                AND %s BETWEEN first_day_obs AND last_day_obs
                AND telescope = %s
                AND tags @> %s:JSONB
                AND creation_time >= NOW() - INTERVAL '%s days'
            """
            query_params = (first_day_obs, last_day_obs, telescope, tags_json, max_simulation_age)
        else:
            query_template = """
            SELECT *
            FROM {}.simulations_extra
            WHERE %s BETWEEN first_day_obs AND last_day_obs
                AND %s BETWEEN first_day_obs AND last_day_obs
                AND telescope = %s
                AND creation_time >= NOW() - INTERVAL '%s days'
            """
            query_params = (first_day_obs, last_day_obs, telescope, max_simulation_age)

        vseq = self.pd_read_sql(query_template, [sql.Identifier(self.metadata_db_schema)], query_params)
        return vseq


#
# Computation
#


def compute_nightly_stats(
    visits: pd.DataFrame, columns: Tuple[str, ...] = ("s_ra", "s_dec", "sky_rotation")
) -> pd.DataFrame:
    if "day_obs" not in visits.columns:
        times = Time(np.floor(visits.obs_start_mjd - 0.5), format="mjd").to_datetime()
        assert isinstance(times, np.ndarray)
        visits = visits.assign(day_obs=pd.Series(times).dt.date)

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


def compute_conda_env() -> tuple:
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


def construct_base_resource_path(
    archive_base: ResourcePath, telescope: str, creation_time: Time, visitseq_uuid: UUID
) -> ResourcePath:
    """Build the base ``ResourcePath`` for a visit‑sequence
    archive entry.

    Parameters
    ----------
    archive_base : `ResourcePath`
        The base of the archive relative to which the visit sequence
        will be added.
    telescope : `str`
        Identifier for the telescope (e.g. ``"simonyi"`` or ``"auxtel"`).
    creation_time : `Time`
        The timestamp when the visit sequence was created.
    visitseq_uuid : `UUID`
        The unique identifier of the visit sequence.

    Returns
    -------
    ResourcePath
        A ``ResourcePath`` instance representing the base directory for
        this visit sequence.  The directory is structured as:

        ``<archive_base>/<telescope>/<date>/<visitseq_uuid>/``

        where ``<date>`` is the UTC‑12 date derived from ``creation_time``.
    """
    # Record in path according to datetime timezone (UTC-12)
    visitseq_base_rp = (
        archive_base.join(telescope)
        .join(Time(creation_time.mjd - 0.5, format="mjd").datetime.date().isoformat())
        .join(str(visitseq_uuid), forceDirectory=True)
    )
    return visitseq_base_rp


#
# Interaction with the file store
#


def _write_data_to_archive(content: bytes, visitseq_base_rp: ResourcePath, file_name: str) -> ResourcePath:
    # Make sure the base for the visit sequence exists
    if not visitseq_base_rp.exists():
        visitseq_base_rp.mkdir()
        LOGGER.debug(f"Created {visitseq_base_rp}.")

    destination_rp = visitseq_base_rp.join(file_name)
    LOGGER.debug(f"Writing {destination_rp}.")
    destination_rp.write(content)
    LOGGER.debug(f"Wrote {destination_rp}.")
    return destination_rp


def _write_file_to_archive(
    origin: str | Path,
    visitseq_base_rp: ResourcePath,
) -> Tuple[ResourcePath, bytes]:
    file_path = origin if isinstance(origin, Path) else Path(origin)
    with open(file_path, "rb") as origin_io:
        content = origin_io.read()

    content_sha256 = bytes.fromhex(hashlib.sha256(content).hexdigest())
    destination_rp = _write_data_to_archive(content, visitseq_base_rp, file_path.name)
    return destination_rp, content_sha256


def add_file(
    vsarch: VisitSequenceArchiveMetadata,
    uuid: UUID,
    origin: str | Path,
    file_type: str,
    archive_base: str | ResourcePath,
    update: bool = False,
) -> ResourcePath:
    """Archive a file associated with a visit sequence and
    register its location in the metadata database.

    Parameters
    ----------
    vsarch : `VisitSequenceArchiveMetadata`
        Instance of `VisitSequenceArchiveMetadata` used to
        query the sequence metadata and register the file.
    uuid : `UUID`
        The unique identifier of the visit sequence that the file
        should be linked to.
    origin : `str` or `pathlib.Path`
        Path to the local file to be archived.
    file_type : `str`
        Identifier for the type of file being archived.
    archive_base : `lsst.resources.ResourcePath`
        Base location of the archive.
    update : `bool`, optional
        If ``True`` and a record for the same ``visitseq_uuid`` and
        ``file_type`` already exists, the existing row will be
        updated with the new SHA‑256 hash and URL.  If ``False`` a
        `ValueError` will be raised on duplicate.

    Returns
    -------
    file_rp : `lsst.resources.ResourcePath`
        The `ResourcePath` pointing to the archived file
        on the file store.
    """
    if isinstance(archive_base, str):
        archive_base = ResourcePath(archive_base)
    assert isinstance(archive_base, ResourcePath)

    visitseq_metadata = vsarch.get_visitseq_metadata(uuid)
    uuid = visitseq_metadata["visitseq_uuid"]
    telescope = visitseq_metadata["telescope"]
    creation_time = Time(visitseq_metadata["creation_time"])
    visitseq_base_rp = construct_base_resource_path(archive_base, telescope, creation_time, uuid)

    # If we are adding a file that looks like opsim sqlite3 output but
    # specify the file type as visits, convert to hdf5 using
    # SchemaConverter and send the result.
    if file_type == "visits" and Path(origin).suffix.lower() in SQLITE_EXTINSIONS:
        schema_converter = SchemaConverter()
        visits = schema_converter.obs2opsim(schema_converter.opsim2obs(origin))
        with TemporaryDirectory() as temp_dir:
            new_origin = Path(temp_dir).joinpath("visits.h5")
            visits.to_hdf(new_origin, key="visits", complevel=2)
            location = add_file(vsarch, uuid, new_origin, "visits", archive_base, update)
        return location

    location, file_sha256 = _write_file_to_archive(origin, visitseq_base_rp)
    if file_type == "visits":
        vsarch.set_visitseq_url(uuid, location.geturl())
    else:
        vsarch.register_file(uuid, file_type, file_sha256, location, update=update)

    return location


#
# API
#


@click.group()
@click.option(
    "--database",
    default=os.getenv("VSARCHIVE_PGDATABASE", VSARCHIVE_PGDATABASE),
    help="PostgreSQL database name of the metadata database",
)
@click.option(
    "--host",
    default=os.getenv("VSARCHIVE_PGHOST", VSARCHIVE_PGHOST),
    help="PostgreSQL host address of the metadata database",
)
@click.option(
    "--user",
    default=None,
    help="PostgreSQL user name to use to connect to the metadata database",
)
@click.option(
    "--schema",
    default=os.getenv("VSARCHIVE_PGSCHEMA", "ehntest"),
    help="Schema of the metadata database containing the visit‑sequence tables",
)
@click.pass_context
def vseqarchive(
    click_context: click.Context,
    database: str,
    host: str,
    user: str | None,
    schema: str,
) -> None:
    """visitseq command line interface."""

    # Create an instance of the interface to
    # the metadata database that can be used
    # by all commands.
    metadata_db_kwargs = {"database": database, "host": host}
    if user:
        metadata_db_kwargs["user"] = user

    click_context.obj = VisitSequenceArchiveMetadata(metadata_db_kwargs, schema)


@vseqarchive.command()
@click.argument("table", type=str)
@click.argument("visits_file", type=click.Path(exists=True))
@click.argument("label")
@click.option(
    "--telescope",
    default="simonyi",
    show_default=True,
    help="Telescope identifier (e.g. simonyi or auxtel)",
)
@click.option("--url", default=None, help="Optional URL for the visits file")
@click.option("--first_day_obs", default=None, help="First night of observations (YYYY-MM-DD or int)")
@click.option("--last_day_obs", default=None, help="Last night of observations (YYYY-MM-DD or int)")
@click.option(
    "--creation_time",
    default=None,
    help="Creation time of the visit sequence in ISO format (e.g. 2025-01-01T00:00:00)",
)
@click.pass_obj
def record_visitseq_metadata(
    vsarch: VisitSequenceArchiveMetadata,
    table: str,
    visits_file: str,
    label: str,
    telescope: str,
    url: str | None,
    first_day_obs: str | int | None,
    last_day_obs: str | int | None,
    creation_time: str | None,
) -> None:
    """Record metadata for a new visit sequence from an HDF5 visits file.

    The first positional argument is the database table to insert into
    (e.g. ``visitseq``, ``simulations``, ``completed``, or
    ``mixedvisitseq``).  The second argument is the path to an
    HDF5 file that contains a dataset named ``visits`` which will be
    loaded into a :class:`pandas.DataFrame` and written to the
    specified table.
    """
    if Path(visits_file).suffix.lower() in SQLITE_EXTINSIONS:
        schema_converter = SchemaConverter()
        visits_df = schema_converter.obs2opsim(schema_converter.opsim2obs(visits_file))
    else:
        visits_df = pd.read_hdf(visits_file, key="visits")
    assert isinstance(visits_df, pd.DataFrame)

    # Convert the optional creation time string to an astropy Time object
    creation_ap_time = Time(creation_time) if creation_time is not None else Time.now()

    # Record the visit sequence metadata
    visitseq_uuid = vsarch.record_visitseq_metadata(
        visits_df,
        label,
        telescope=telescope,
        table=table,
        url=url,
        first_day_obs=first_day_obs,
        last_day_obs=last_day_obs,
        creation_time=creation_ap_time,
    )
    print(visitseq_uuid)


@vseqarchive.command()
@click.argument("uuid", type=click.UUID)
@click.option(
    "--table",
    default="visitseq",
    show_default=True,
    help="Table to query (e.g. visitseq, simulations, completed, mixedvisitseq)",
)
@click.pass_obj
def get_visitseq_metadata(vsarch: VisitSequenceArchiveMetadata, uuid: UUID, table: str) -> None:
    """Retrieve and display metadata for a visit sequence.

    The result is printed as a tab‑separated table for easy reading
    or machine‑parsing.
    """
    sequence_metadata = vsarch.get_visitseq_metadata(uuid, table=table)

    # Print the DataFrame as a tab‑separated table (no row index)
    print(sequence_metadata.to_frame().T.to_csv(sep="\t", index=False).rstrip("\n"))


@vseqarchive.command()
@click.argument("uuid", type=click.UUID)
@click.argument("url", type=click.STRING)
@click.pass_obj
def set_visitseq_url(vsarch: VisitSequenceArchiveMetadata, uuid: UUID, url: str) -> None:
    """Update the URL for a visit sequence file.

    The URL is stored in the specified table for the given visit sequence UUID.
    """
    vsarch.set_visitseq_url(uuid, url)


@vseqarchive.command()
@click.argument("uuid", type=click.UUID)
@click.pass_obj
def get_visitseq_url(vsarch: VisitSequenceArchiveMetadata, uuid: UUID) -> None:
    """Print the URL for the visits file of a visit sequence.

    The URL is stored in the appropriate child table of the metadata database.
    """
    url = vsarch.get_visitseq_url(uuid)
    click.echo(url)


@vseqarchive.command()
@click.argument("uuid", type=click.UUID)
@click.argument("field", type=str)
@click.argument("value", type=str)
@click.pass_obj
def update_visitseq_metadata(
    vsarch: VisitSequenceArchiveMetadata,
    uuid: UUID,
    field: str,
    value: str,
) -> None:
    """Update a single metadata field for a visit sequence.

    Parameters
    ----------
    vsarch : `VisiteSequenceArchiveMetadata`
        An instance of the interface to the metadata archive.
    uuid : `UUID`
        The UUID of the visit sequence to update.
    field : `str`
        The name of the column in the metadata table to modify.
    value : `str`
        The new value for the field.  Postgresql will cast the type
        to what is expected in the database schema.
    """
    # psycopg2 usually converts strings to the right
    # type base on column type in the database, but if
    # hex values are provided for bytes colums it
    # gets it wrong.
    # So, if we're asking to update a bytes column, convert
    # it from hex ourselves.
    update_value = value
    if field in {"conda_env_sha256"}:
        update_value = bytes.fromhex(value)

    vsarch.update_visitseq_metadata(uuid, field, update_value)


@vseqarchive.command()
@click.argument("uuid", type=click.UUID)
@click.argument("tag", type=str)
@click.pass_obj
def is_tagged(vsarch: VisitSequenceArchiveMetadata, uuid: UUID, tag: str) -> None:
    """Return whether a visit sequence is tagged with a given tag.

    The command prints ``true`` if the tag exists for the visit sequence
    and ``false`` otherwise.

    Parameters
    ----------
    uuid : `UUID`
        The UUID of the visit sequence to check.
    tag : `str`
        The tag name to query.
    """
    seq_is_tagged = vsarch.is_tagged(uuid, tag)
    if seq_is_tagged:
        click.echo("true")
    else:
        click.echo("false")


@vseqarchive.command()
@click.argument("uuid", type=click.UUID)
@click.argument("tags", nargs=-1, required=True)
@click.pass_obj
def tag(vsarch: VisitSequenceArchiveMetadata, uuid: UUID, tags: Tuple[str, ...]) -> None:
    """Add one or more tags to a visit sequence.

    Parameters
    ----------
    uuid : `UUID`
        The UUID of the visit sequence to tag.
    tags : `str`
        One or more tag names to add.
    """
    if len(tags) < 1:
        raise ValueError("At least one tag must be requested.")

    vsarch.tag(uuid, *tags)


@vseqarchive.command()
@click.argument("uuid", type=click.UUID)
@click.argument("tag", type=str)
@click.pass_obj
def untag(vsarch: VisitSequenceArchiveMetadata, uuid: UUID, tag: str) -> None:
    """Remove a tag from a visit sequence.

    Parameters
    ----------
    uuid : `UUID`
        The UUID of the visit sequence to modify.
    tag : `str`
        The name of the tag to remove.
    """
    vsarch.untag(uuid, tag)


@vseqarchive.command()
@click.argument("uuid", type=click.UUID)
@click.argument("comment")
@click.option(
    "--author",
    default=None,
    help="The author of the comment",
)
@click.pass_obj
def comment(
    vsarch: VisitSequenceArchiveMetadata, uuid: UUID, comment: str, author: str | None = None
) -> None:
    vsarch.comment(uuid, comment, author)


@vseqarchive.command()
@click.argument("uuid", type=click.UUID)
@click.pass_obj
def get_comments(vsarch: VisitSequenceArchiveMetadata, uuid: UUID) -> None:
    """Retrieve all comments attached to a specific visit sequence.

    The comments are printed as a tab‑separated table with columns:
    ``visitseq_uuid``, ``comment_time``, ``author``, ``comment``.

    Parameters
    ----------
    uuid : `UUID`
        The UUID of the visit sequence whose comments should be fetched.
    """
    comments_df = vsarch.get_comments(uuid)
    if not comments_df.empty:
        # Pandas will include the index by default; omit it.
        click.echo(comments_df.to_csv(sep="\t", index=False).rstrip("\n"))


@vseqarchive.command()
@click.argument("uuid", type=click.UUID)
@click.argument("origin", type=click.Path(exists=True))
@click.argument("file_type", type=click.STRING)
@click.option(
    "--archive-base",
    default=ARCHIVE_URL,
    show_default=True,
    help="Base directory for the archive (e.g. file://data/archive).",
)
@click.option(
    "--update",
    is_flag=True,
    help="If set, overwrite an existing record for the same visitseq_uuid and file_type.",
)
@click.pass_obj
def archive_file(
    vsarch: VisitSequenceArchiveMetadata,
    uuid: UUID,
    origin: str,
    file_type: str,
    archive_base: str,
    update: bool,
) -> None:
    """Archive a file and register its location in the metadata database."""
    # Convert the base path string into a ResourcePath
    archive_base_rp = ResourcePath(archive_base)

    archived_location = add_file(vsarch, uuid, origin, file_type, archive_base_rp, update=update)

    click.echo(archived_location.geturl())


@vseqarchive.command()
@click.argument("destination", type=click.Path(exists=False))
@click.argument("uuid", type=click.UUID)
@click.argument("file_type", type=click.STRING)
@click.pass_obj
def get_file(
    vsarch: VisitSequenceArchiveMetadata,
    destination: str,
    uuid: UUID,
    file_type: str,
) -> None:
    """Retrieve a registered file and copy it to a local destination.

    Parameters
    ----------
    vsarch : `VisitSequenceArchiveMetadata`
        The metadata interface used to look up the file URL.
    destination : `str`
        Path where the retrieved file should be written. If the
        destination does not exist it will be created.  For files
        registered with the ``visits`` type and a SQLite3 extension
        suffix, the file will be downloaded as an HDF5 ``visits.h5`` file,
        converted back to the original opsim SQLite format, and written to
        ``destination``.
    uuid : `UUID`
        The unique identifier of the visit sequence containing the
        requested file.
    file_type : `str`
        The handle for the type of file to retrieve (e.g. ``visits``).

    Notes
    -----
    The function obtains the URL from the metadata database,
    then copies the file from the URL to the local filesystem.  For a
    ``visits`` file whose destination ends with a SQLite3 extension
    (``.db``, ``.sqlite``), the file is first downloaded as a HDF5
    temporary file, converted to the legacy opsim SQLite format by
    `SchemaConverter`, and then written to ``destination``.
    Otherwise the file is copied directly.

    This command prints the local path of the copied file.
    """
    file_url = vsarch.get_file_url(uuid, file_type)
    origin_rp = ResourcePath(file_url)
    if file_type == "visits" and Path(destination).suffix.lower() in SQLITE_EXTINSIONS:
        with TemporaryDirectory() as temp_dir:
            h5_destination = ResourcePath(temp_dir).join("visits.h5")
            h5_destination.transfer_from(origin_rp, "copy")
            click.echo(f"Copied {origin_rp.geturl()} to {h5_destination.geturl()}")
            visits = pd.read_hdf(h5_destination.ospath, "visits")
            schema_converter = SchemaConverter()
            obs = schema_converter.opsimdf2obs(visits)
            schema_converter.obs2opsim(obs, destination)
            click.echo(f"Converted to opsim and wrote to {destination}")
    else:
        destination_rp = ResourcePath(destination)
        destination_rp.transfer_from(origin_rp, "copy")
        click.echo(f"Copied {origin_rp.geturl()} to {destination_rp.geturl()}")


@vseqarchive.command()
@click.argument("uuid", type=click.UUID)
@click.argument("visits_file", type=click.Path(exists=True))
@click.option(
    "--columns",
    "-c",
    multiple=True,
    help=(
        "Columns to compute nightly statistics on. "
        "If omitted, the default set defined in "
        "``compute_nightly_stats`` is used."
    ),
)
@click.pass_obj
def add_nightly_stats(
    vsarch: VisitSequenceArchiveMetadata,
    uuid: UUID,
    visits_file: str,
    columns: tuple[str, ...],
) -> None:
    """Compute nightly statistics for a visit sequence and add
    them to the visit sequence metadata database.

    Parameters
    ----------
    uuid : `UUID`
        The UUID of the visit sequence to attach the statistics to.
    visits_file : `str`
        Path to an HDF5 file containing a ``visits`` dataset.
    columns : ``tuple[str]``
        Optional list of columns to pass to `compute_nightly_stats`.
    """
    # Load the visits table from the HDF5 file
    visits_df = pd.read_hdf(visits_file, key="visits")
    assert isinstance(visits_df, pd.DataFrame), "Expected a DataFrame from key 'visits'"

    # Compute the nightly statistics, passing the requested columns if any
    if columns:
        stats_df = compute_nightly_stats(visits_df, columns=tuple(columns))
    else:
        stats_df = compute_nightly_stats(visits_df)

    # Insert into the database
    vsarch.insert_nightly_stats(uuid, stats_df)

    # Print the statistics as TSV
    tsv = stats_df.to_csv(sep="\t", index=False).rstrip("\n")
    click.echo(tsv)


@vseqarchive.command()
@click.argument("uuid", type=click.UUID)
@click.pass_obj
def query_nightly_stats(vsarch: VisitSequenceArchiveMetadata, uuid: UUID) -> None:
    """Retrieve nightly statistics for a visit sequence and
    print them as a tab‑separated table.

    Parameters
    ----------
    uuid : `UUID`
        The unique identifier of the visit sequence.
    """
    stats_df = vsarch.query_nightly_stats(uuid)
    output = stats_df.to_csv(sep="\t", index=False).rstrip("\n") if not stats_df.empty else ""
    click.echo(output)


@vseqarchive.command()
@click.pass_obj
def record_conda_env(vsarch: VisitSequenceArchiveMetadata) -> None:
    """Record the current Conda environment in the metadata database,
    and print the hash used as a key for the env in the database.
    """
    env_hash, env_json = compute_conda_env()
    vsarch.record_conda_env(env_hash, env_json)
    click.echo(env_hash.hex())


if __name__ == "__main__":
    vseqarchive()
