__all__ = ["compute_visits_sha256", "VisitSequenceArchiveMetadata"]

import hashlib
import json
import logging
import os
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Mapping, Tuple
from uuid import UUID

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import psycopg2.pool
import sqlalchemy
import yaml
from astropy.time import Time
from lsst.resources import ResourcePath
from psycopg2 import sql

from .util import compute_conda_env, dayobs_to_date, opsimdb_to_hdf5

VSARCHIVE_PGDATABASE = "opsim_log"
VSARCHIVE_PGHOST = "usdf-maf-visit-seq-archive-tx-ro.sdf.slac.stanford.edu"
VSARCHIVE_PGUSER = "reader"
VSARCHIVE_PGSCHEMA = "vsmd"

JSON_DUMP_LIMIT = 4096

LOGGER = logging.getLogger(__name__)

psycopg2.extras.register_uuid()


def compute_visits_sha256(visits: pd.DataFrame) -> bytes:
    recs = visits.to_records()
    visitseq_hash = hashlib.sha256(str(recs.dtype).encode())
    visitseq_hash.update(np.ascontiguousarray(recs).data.tobytes())
    visitseq_sha256 = bytes.fromhex(visitseq_hash.hexdigest())
    return visitseq_sha256


def metadata_yaml_from_prototype(
    sim_date: str | date,
    sim_index: str | int,
    proto_sim_archive_uri: str = "s3://rubin:rubin-scheduler-prenight/opsim/",
) -> str:
    """Return a sim metadata YAML derived from the prototype sim archive.

    Parameters
    ----------
    sim_date : `str` or `datetime.date`
        The simulation date in ISO format or a ``date`` object.
    sim_index : `str` or `int`
        The simulation index identifying the specific simulation
        within the date directory.
    proto_sim_archive_uri : `str`, optional
        Base URI of the simulation archive.
        Defaults to the prototype location in the S3 bucket.

    Returns
    -------
    sim_metadata: `str`
        `yaml` representation of the simulation metadata.

    """
    if isinstance(sim_date, date):
        sim_date = sim_date.isoformat()
    assert isinstance(sim_date, str)

    sim_archive_rp = (
        ResourcePath(proto_sim_archive_uri)
        .join(sim_date, forceDirectory=True)
        .join(str(sim_index), forceDirectory=True)
    )
    sims_found = sim_archive_rp.exists() and sim_archive_rp.join("sim_metadata.yaml").exists()
    if not sim_archive_rp.exists():
        raise ValueError(f"No simulations found at {sim_archive_rp.geturl()}.")

    if not sim_archive_rp.join("sim_metadata.yaml").exists():
        raise ValueError(
            f"No simulation metadata found at {sim_archive_rp.join('sim_metadata.yaml').geturl()}"
        )

    if sims_found:
        metadata = yaml.safe_load(sim_archive_rp.join("sim_metadata.yaml").read().decode())
    else:
        raise ValueError("No simulation found")

    if "label" not in metadata:
        raise ValueError("Metadata yaml must include a label")

    label = metadata["label"]

    # If there is no creation time given, try extracting it from the label
    if "creation_time" not in metadata:
        # The prenight labels usually end in the creation time.
        if " ".join(label.split()[-3:-1]) == "run at":
            maybe_date_str = label.split()[-1]
            is_date_str = False
            try:
                datetime.fromisoformat(maybe_date_str)
                is_date_str = True
            except ValueError:
                is_date_str = False

            if is_date_str:
                metadata["creation_time"] = maybe_date_str

    # If we still do not have a creation time, invent one from the date
    # and index.
    if "creation_time" not in metadata:
        creation_time = datetime.fromisoformat(sim_date + "T12:00:00Z") + timedelta(seconds=int(sim_index))
        metadata["creation_time"] = creation_time.isoformat()

    if "files" in metadata:
        for file_type in metadata["files"]:
            if "url" not in metadata["files"][file_type]:
                fname = metadata["files"][file_type]["name"]
                metadata["files"][file_type]["url"] = sim_archive_rp.join(fname).geturl()

    metadata_yaml = yaml.dump(metadata)

    return metadata_yaml


class VisitSequenceArchiveMetadata:
    """Interface to metadata database that tracks sequences
    of visits.

    Parameters
    ----------
    metadata_db_kwargs: `Mapping` or `None`
        A dictionary or other mapping defining the connection
        parameters for connecting to the postgresql database
        that holds the sequence metadata. Keys are passed as keyword
        arguments to ``psycopg2.pool.SimpleConnectionPool``.
        If ``None``, the dictionary is built from the environment
        variables ``VSARCHIVE_PGDATABASE``, ``VSARCHIVE_PGHOST``,
        ``VSARCHIVE_PGUSER``, and ``VSARCHIVE_PGPORT`` (if they
        exist) or corresponding module variables in
        ``rubin_sim.sim_archive.vseqarchiv`` (if they do not).
    metadata_db_schema: `str` or `None`
        The schema in the database holding the metadata. If
        ``None``, defaults to
        ``rubin_sim.sim_archive.vseqarchive.VSARCHIVE_PGSCHEMA``.
        Default is ``None``.

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
        metadata_db_schema: str | None = None,
    ):
        if metadata_db_schema is None:
            metadata_db_schema = VSARCHIVE_PGSCHEMA
        assert isinstance(metadata_db_schema, str)

        if not isinstance(metadata_db_kwargs, dict):
            metadata_db_kwargs = {} if metadata_db_kwargs is None else dict(metadata_db_kwargs)
        assert isinstance(metadata_db_kwargs, dict)

        if "database" not in metadata_db_kwargs:
            if "VSARCHIVE_PGDATABASE" in os.environ:
                metadata_db_kwargs["database"] = os.environ["VSARCHIVE_PGDATABASE"]
            else:
                metadata_db_kwargs["database"] = VSARCHIVE_PGDATABASE

        if "host" not in metadata_db_kwargs:
            if "VSARCHIVE_PGHOST" in os.environ:
                metadata_db_kwargs["host"] = os.environ["VSARCHIVE_PGHOST"]
            else:
                metadata_db_kwargs["host"] = VSARCHIVE_PGHOST

        if "user" not in metadata_db_kwargs:
            if "VSARCHIVE_PGUSER" in os.environ:
                metadata_db_kwargs["user"] = os.environ["VSARCHIVE_PGUSER"]
            else:
                metadata_db_kwargs["user"] = VSARCHIVE_PGUSER

        if "port" not in metadata_db_kwargs and "VSARCHIVE_PGPORT" in os.environ:
            metadata_db_kwargs["port"] = os.environ["VSARCHIVE_PGPORT"]

        self.pg_pool = psycopg2.pool.SimpleConnectionPool(1, 5, **metadata_db_kwargs)
        # On some operations, pandas does not
        # work well directly with the psycopg2
        # connection, but can work with sqlalchemy.
        # Make and alchemy engine that acts as
        # a middle-man between pandas and psycopg2,
        # using the same connection pool.
        # There is a potential issue in that SA
        # does not return connections to the
        # psycopg pool when it is done. But, it
        # has its own pool, and so in expected use
        # it will just get one and keep reusing it.
        self.sa_engine = sqlalchemy.create_engine(
            "postgresql+psycopg2://", creator=self.pg_pool.getconn, pool_pre_ping=True
        )

        self.metadata_db_schema: str = metadata_db_schema

    def query(
        self,
        query: str | sql.SQL | sql.Composed,
        data: dict = {},
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
                search_path_query = sql.SQL("SET search_path TO {};").format(
                    sql.Identifier(self.metadata_db_schema)
                )
                cursor.execute(search_path_query)

                if len(data) > 0:
                    cursor.execute(query, data)
                else:
                    cursor.execute(query)
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

    def create_schema_in_database(self) -> None:
        """Create the visit sequence metadata schema in a database."""

        # Creation of the production schema should be
        # a one-off execution.
        assert "test" in self.metadata_db_schema

        # Make sure the schema does not already exist
        schema_test_query = sql.SQL(
            "SELECT EXISTS(SELECT * FROM information_schema.schemata WHERE schema_name = {});"
        ).format(sql.Placeholder("schema"))
        data = {"schema": self.metadata_db_schema}
        schema_exists_return = self.query(schema_test_query, data, return_result=True)
        schema_exists = schema_exists_return[0][0]
        if schema_exists:
            raise ValueError("Schema already exists.")

        creation_sql_file = Path(__file__).parent.parent.parent / "sql" / "make_vseqmeta.sql"
        with open(creation_sql_file, "r") as f:
            query_template = f.read()

        # Sanity check what we read.
        assert "CREATE SCHEMA" in query_template

        query = sql.SQL(query_template).format(
            sql.Identifier(self.metadata_db_schema), sql.Identifier(self.metadata_db_schema)
        )
        self.query(query, {}, commit=True, return_result=False)
        print("Created test database and schema ", self.metadata_db_schema)

    def pd_read_sql(
        self,
        query_template: str,
        sql_params: list[sql.Composable] | None = None,
        query_params: tuple | None = None,
    ) -> pd.DataFrame:
        """Execute a SQL query using the internal PostgreSQL connection pool
        and return the results as a pandas DataFrame.

        Parameters
        ----------
        query_template : `str`
            A query template.
        sql_params : `list` [`sql.Composable`]
            Elements that will be substituted into the template (e.g. table or
            schema identifiers) using ``psycopg2.sql.SQL.format``.
            To make type checkers happy, you may need to explictly declare
            variables passed as ``list[sql.Composable]``.
        query_params : `tuple`
            Positional parameters that will be bound by sqlalchemy.

        Returns
        -------
        df : `pd.DataFrame`
            The result set of the query, wrapped in a DataFrame.

        Notes
        -----
        This works around limitations in pandas when dealing with
        complex datatypes in raw postgresql connections.
        """

        if sql_params is None:
            sql_params = []
        assert isinstance(sql_params, list)

        if query_params is None:
            query_params = tuple([])
        assert isinstance(query_params, tuple)

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

        with self.sa_engine.connect() as sa_conn:
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
            to SITCOMTN-032 (UTC-12hrs).
        last_day_obs : `date` or `int` or `str`, optional
            The last day of observations, defined according
            to SITCOMTN-032 (UTC-12hrs).
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
        registered with the ``record_simulation_metadata`` and
        ``record_completed_metadata`` methods instead.
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
        data: Dict[str, Any] = {"visitseq_sha256": sha256, "visitseq_label": label, "telescope": telescope}

        if url is not None:
            columns.append(sql.Identifier("visitseq_url"))
            data_placeholders.append(sql.Placeholder("visitseq_url"))
            data["visitseq_url"] = url

        if first_day_obs is not None:
            columns.append(sql.Identifier("first_day_obs"))
            data_placeholders.append(sql.Placeholder("first_day_obs"))
            data["first_day_obs"] = dayobs_to_date(first_day_obs)

        if last_day_obs is not None:
            columns.append(sql.Identifier("last_day_obs"))
            data_placeholders.append(sql.Placeholder("last_day_obs"))
            data["last_day_obs"] = dayobs_to_date(last_day_obs)

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
                creation_utc = creation_time.utc
                assert isinstance(creation_utc, Time)
                creation_datetime = creation_utc.datetime
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
        if table not in {"visitseq", "mixedvisitseq", "completed", "simulations", "simulations_extra"}:
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
        parent_last_day_obs: str | date | int | None = None,
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
            to SITCOMTN-032 (UTC-12hrs).
        last_day_obs : `date` or `int` or `str`, optional
            The last day of observations, defined according
            to SITCOMTN-032 (UTC-12hrs).
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
        parent_last_day_obs : `str`, optional
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
        data: Dict[str, Any] = {}
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

        if parent_last_day_obs is not None:
            set_clauses.append(make_set_clause("parent_last_day_obs"))
            data["parent_last_day_obs"] = parent_last_day_obs

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
            to SITCOMTN-032 (UTC-12hrs).
        last_day_obs : `date` or `int` or `str`, optional
            The last day of observations, defined according
            to SITCOMTN-032 (UTC-12hrs).
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
            to SITCOMTN-032 (UTC-12hrs).
        last_day_obs : `date` or `int` or `str` or `None`, optional
            The last day of observations, defined according
            to SITCOMTN-032 (UTC-12hrs).
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
            "last_early_day_obs": dayobs_to_date(last_early_day_obs),
            "first_late_day_obs": dayobs_to_date(first_late_day_obs),
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
        data: Dict[str, Any] = {"visitseq_uuid": visitseq_uuid}
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
        visitseq_uuid: `uuid.UUID`
            The UUID of the visit sequence to comment on.
        comment : `str`
            The comment to attach.
        author : `str`, optional
            The author of the comment.
            Defaults to `None`.
        """
        comment_utc = Time.now().utc
        # mypy cannot figure out that comment_utc cannot be masked
        # so assert that it is not as a hint to it.
        assert isinstance(comment_utc, Time)
        comment_time = comment_utc.datetime
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

    def record_conda_env(
        self, conda_env_hash: bytes | None = None, conda_env_json: str | None = None
    ) -> bytes:
        """Record the current Conda environment in the database.

        Parameters
        ----------
        conda_env_hash : `bytes` or `None`
            The SHA‑256 digest of the Conda environment JSON
            representation. ``None`` (the default)
            uses the current conda environment.
        conda_env_json : `str` or `None`
            The json representing the conda environment.
            ``None`` (the default) uses the current
            conda environment.
        """
        if conda_env_hash is None != conda_env_json is None:
            raise ValueError("conda_env_hash and conda_env_json must both be set, or neither.")

        if conda_env_hash is None:
            conda_env_hash, conda_env_json = compute_conda_env()

        assert isinstance(conda_env_hash, bytes)
        assert isinstance(conda_env_json, str)

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
        first_day_obs: str | date | int | None = None,
        last_day_obs: str | date | int | None = None,
        tags: tuple[str, ...] = ("prenight", "ideal", "nominal"),
        telescope: str = "simonyi",
        max_simulation_age: int = 2,
    ) -> pd.DataFrame:
        """Return a table of simulations that cover a given night range.

        Parameters
        ----------
        first_day_obs : `str` | `date` | `int` | `None`, optional
            The first observation day of the window for which simulations are
            requested.  It can be specified as a `datetime.date` object, an
            ISO‑formatted string (e.g. ``'2025-12-01'``), or an integer in
            SMTN‑032 format (e.g. ``20251201``).  If ``None`` (the default),
            the method uses the current dayobs.
        last_day_obs : `str` | `date` | `int` | `None`, optional
            The last observation day of the window.  It accepts the same
            types as ``first_day_obs``.  If ``None`` the same value as
            ``first_day_obs`` is used, giving a single‑night query.
        tags : `tuple[str]`, optional
            A sequence of tags that must be present on the simulation record.
            The default tags are ``("prenight", "ideal", "nominal")``.  If
            an empty tuple is supplied, the tag test is omitted.
        telescope : `str`, optional
            The telescope simulated.  The default is ``simonyi``.
        max_simulation_age : `int`, optional
            The maximum age of a simulation in days.
            The default is 2 days.

        Returns
        -------
        vseqs : `pd.DataFrame`
            A table of metadata for matching simulations.
        """

        if first_day_obs is None:
            first_day_obs = datetime.now(timezone(timedelta(hours=-12))).date()
        if last_day_obs is None:
            last_day_obs = first_day_obs

        first_day_obs = dayobs_to_date(first_day_obs)
        last_day_obs = dayobs_to_date(last_day_obs)

        assert isinstance(first_day_obs, date)
        assert isinstance(last_day_obs, date)

        if len(tags) > 0:
            tags_json = json.dumps(list(tags))
            query_template = """
            SELECT *
            FROM {}.simulations_extra AS s
            WHERE %s BETWEEN first_day_obs AND last_day_obs
                AND %s BETWEEN first_day_obs AND last_day_obs
                AND telescope = %s
                AND tags @> %s::JSONB
                AND creation_time >= NOW() - INTERVAL '%s days'
            """
            query_params: Tuple = (first_day_obs, last_day_obs, telescope, tags_json, max_simulation_age)
        else:
            query_template = """
            SELECT *
            FROM {}.simulations_extra AS s
            WHERE %s BETWEEN first_day_obs AND last_day_obs
                AND %s BETWEEN first_day_obs AND last_day_obs
                AND telescope = %
                AND creation_time >= NOW() - INTERVAL '%s days'
            """
            query_params = (first_day_obs, last_day_obs, telescope, max_simulation_age)

        vseqs = self.pd_read_sql(
            query_template,
            [sql.Identifier(self.metadata_db_schema)],
            query_params,
        )
        return vseqs

    def sims_on_night_with_stats(
        self,
        day_obs: str | date | int | None = None,
        tags: tuple[str, ...] = ("prenight", "ideal", "nominal"),
        telescope: str = "simonyi",
        max_simulation_age: int = 2,
    ) -> pd.DataFrame:
        """Return a table of simulations that cover a given night range.

        Parameters
        ----------
        day_obs : `str` | `date` | `int` | `None`, optional
            The day_obs for which to get simulations.
        tags : `tuple[str]`, optional
            A sequence of tags that must be present on the simulation record.
            The default tags are ``("prenight", "ideal", "nominal")``.  If
            an empty tuple is supplied, the tag test is omitted.
        telescope : `str`, optional
            The telescope simulated.  The default is ``simonyi``.
        max_simulation_age : `int`, optional
            The maximum age of a simulation in days.
            The default is 2 days.

        Returns
        -------
        vseqs : `pd.DataFrame`
            A table of metadata for matching simulations.
        """

        if day_obs is None:
            day_obs = datetime.now(timezone(timedelta(hours=-12))).date()

        day_obs = dayobs_to_date(day_obs)
        assert isinstance(day_obs, date)

        if len(tags) > 0:
            tags_json = json.dumps(list(tags))
            query_template = """
            WITH aggstats AS (
                SELECT
                    visitseq_uuid,
                    JSONB_OBJECT_AGG(
                        value_name,
                        TO_JSONB(ns) - 'value_name' - 'visitseq_uuid' - 'day_obs'
                    ) AS stats
                FROM {}.nightly_stats AS ns
                GROUP BY ns.visitseq_uuid
            )
            SELECT s.*, ns.stats
            FROM {}.simulations_extra AS s
            LEFT JOIN aggstats AS ns ON s.visitseq_uuid=ns.visitseq_uuid
            WHERE %s BETWEEN first_day_obs AND last_day_obs
                AND telescope = %s
                AND tags @> %s::JSONB
                AND creation_time >= NOW() - INTERVAL '%s days'
            """
            query_params: Tuple = (day_obs, telescope, tags_json, max_simulation_age)
        else:
            query_template = """
            WITH aggstats AS (
                SELECT
                    visitseq_uuid,
                    JSONB_OBJECT_AGG(
                        value_name,
                        TO_JSONB(ns) - 'value_name' - 'visitseq_uuid' - 'day_obs'
                    ) AS stats
                FROM {}.nightly_stats AS ns
                GROUP BY ns.visitseq_uuid
            )
            SELECT s.*, ns.stats
            FROM {}.simulations_extra AS s
            LEFT JOIN aggstats AS ns ON s.visitseq_uuid=ns.visitseq_uuid
            WHERE %s BETWEEN first_day_obs AND last_day_obs
                AND telescope = %
                AND creation_time >= NOW() - INTERVAL '%s days'
            """
            query_params = (day_obs, telescope, max_simulation_age)

        vseqs = self.pd_read_sql(
            query_template,
            [sql.Identifier(self.metadata_db_schema), sql.Identifier(self.metadata_db_schema)],
            query_params,
        )
        return vseqs

    def import_sim_from_yaml(self, metadata_yaml: str, archive_base: str | None = None) -> UUID:
        """Import a simulation from a YAML metadata description.

        Parameters
        ----------
        metadata_yaml : `str`
            YAML string with simulation metadata.
        archive_base : `str` or `None`
            Base location of the archive to which to copy the visit
            sequence. If ``None``, the visits will not be copied.
            Defaults to ``None``.

        Returns
        -------
        visitseq_uuid : `UUID`
            The UUID of the newly registered visit sequence.
        """

        sim_archive_metadata = yaml.safe_load(metadata_yaml)

        if "label" not in sim_archive_metadata:
            raise ValueError("Metadata yaml must include a label")

        if "files" not in sim_archive_metadata or "observations" not in sim_archive_metadata["files"]:
            raise ValueError("Metadata yaml must include the observations file")

        label = sim_archive_metadata["label"]
        record_sim_kwargs = {"label": label}

        # The prenight labels usually end in the creation time.
        if " ".join(label.split()[-3:-1]) == "run at":
            maybe_date_str = label.split()[-1]
            try:
                record_sim_kwargs["creation_time"] = Time(datetime.fromisoformat(maybe_date_str))
            except ValueError:
                pass

        for key in ["scheduler_version", "telescope"]:
            if key in sim_archive_metadata:
                record_sim_kwargs[key] = sim_archive_metadata[key]

        for side in ["first", "last"]:
            try:
                record_sim_kwargs[f"{side}_day_obs"] = sim_archive_metadata["simulated_dates"][side]
            except KeyError:
                pass

        # Get the visits hdf5 file
        with TemporaryDirectory() as temp_dir:
            visits_h5_fname = Path(temp_dir) / "visits.h5"
            obs_rp = ResourcePath(sim_archive_metadata["files"]["observations"]["url"])
            with obs_rp.as_local() as obs_local_rp:
                opsimdb_to_hdf5(obs_local_rp.ospath, visits_h5_fname)

            record_sim_kwargs["visits"] = pd.read_hdf(visits_h5_fname, "observations")

            sim_uuid = self.record_simulation_metadata(**record_sim_kwargs)

            if archive_base is not None:
                # Import here to avoid circular imports
                from .vseqarchive import add_file

                add_file(self, sim_uuid, visits_h5_fname, "visits", archive_base)

        if "files" in sim_archive_metadata:
            for file_type in sim_archive_metadata["files"]:
                file_rp = ResourcePath(sim_archive_metadata["files"][file_type]["url"])
                file_sha256 = bytes.fromhex(hashlib.sha256(file_rp.read()).hexdigest())
                self.register_file(sim_uuid, file_type, file_sha256, file_rp)

        if "tags" in sim_archive_metadata:
            valid_tags = [t for t in sim_archive_metadata["tags"] if isinstance(t, str)]
            self.tag(sim_uuid, *valid_tags)

        return sim_uuid

    def import_sim_from_prototype_sim_archive(
        self, archive_base: str, sim_date: str | date, sim_index: str | int, proto_sim_archive_uri: str
    ) -> UUID:
        """Import a simulation metadata from the prototype achive.

        Parameters
        ----------
        archive_base : `str`
            Base location where the new archive entry will be written
            (e.g. an S3 bucket or a local directory).  The method
            will write and hdf5 with visits to this location and
            register the resulting URL in the database.
        sim_date : `str` or `datetime.date`
            The ISO‑formatted date (e.g. ``'2025-03-12'``) or a
            ``date`` object that identifies (with the sim_index)
            a simulation in the prototype archive.
        sim_index : `str` or `int`
            The index of the simulation which completes the
            identification of the simulation in the prototype.
        proto_sim_archive_uri : `str`
            Base URI of the prototype simulation archive.  By default
            this is ``'s3://rubin:rubin-scheduler-prenight/opsim/'``.

        Returns
        -------
        visitseq_uuid : `UUID`
            The UUID of the visit sequence that was created and
            registered.
        """

        metadata_yaml = metadata_yaml_from_prototype(sim_date, sim_index, proto_sim_archive_uri)
        sim_uuid = self.import_sim_from_yaml(metadata_yaml, archive_base)
        self.tag(sim_uuid, "from_prototype_sim_archive")

        sim_date_str = sim_date if isinstance(sim_date, str) else sim_date.isoformat()
        self.comment(sim_uuid, f"Imported from prototype sim_archive {sim_date_str}, {sim_index}")
        return sim_uuid

    def sim_metadata_yaml(self, visitseq_uuid: UUID) -> str:
        """Return a YAML representation of a simulation's metadata.

        Parameters
        ----------
        visitseq_uuid : `UUID`
            The unique identifier of the visit sequence for which
            metadata should be exported.

        Returns
        -------
        md_yaml: `str`
            yaml representation of (some of) the simulation metadata
            in the metadata database.
        """

        metadata_seq = self.get_visitseq_metadata(visitseq_uuid, "simulations_extra")

        metadata: dict = {"uuid": str(metadata_seq.visitseq_uuid), "label": str(metadata_seq.visitseq_label)}
        for key in ["telescope", "tags", "scheduler_version"]:
            if metadata_seq[key] is not None:
                metadata[key] = metadata_seq[key]

        if metadata_seq["creation_time"] is not None:
            metadata["creation_time"] = metadata_seq["creation_time"].isoformat()

        if metadata_seq.first_day_obs is not None or metadata_seq.last_day_obs is not None:
            metadata["simulation_dates"] = {}
            if metadata_seq.first_day_obs is not None:
                metadata["simulation_dates"]["first"] = metadata_seq.first_day_obs
            if metadata_seq.last_day_obs is not None:
                metadata["simulation_dates"]["last"] = metadata_seq.last_day_obs

        if metadata_seq.files is not None:
            metadata["files"] = {}
            for file_type in metadata_seq.files:
                metadata["files"][file_type] = {
                    "url": metadata_seq.files[file_type],
                    "name": ResourcePath(metadata_seq.files[file_type]).basename(),
                }

        metadata_yaml = yaml.dump(metadata)
        return metadata_yaml
