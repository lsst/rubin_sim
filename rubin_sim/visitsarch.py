import hashlib
import json
from datetime import date, datetime
from types import MappingProxyType
from typing import Mapping
from uuid import UUID

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import psycopg2.pool
from astropy.time import Time
from lsst.resources import ResourcePath, ResourcePathExpression
from psycopg2 import sql

USDF_METADATA_DATABASE = MappingProxyType(
    {"database": "opsim_log", "host": "134.79.23.205", "schema": "vsarchive"}
)

TEST_METADATA_DATABASE = MappingProxyType(
    {"database": "opsim_log", "host": "134.79.23.205", "schema": "vsarchtest"}
)

JSON_DUMP_LIMIT = 4096

psycopg2.extras.register_uuid()


def _dayobs_to_date(dayobs: str | date | int) -> date:
    match dayobs:
        case int():
            year = dayobs // 10000
            month = (dayobs // 100) % 100
            day = dayobs & 100
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
    def __init__(
        self,
        metadata_db: Mapping = TEST_METADATA_DATABASE,
        archive_url: ResourcePathExpression = "test_archive",
    ):
        self.metadata_db: Mapping = metadata_db
        self.archive_base: ResourcePath = ResourcePath(archive_url)
        self.metadata_db_schema: str = metadata_db["schema"]
        metadata_connection_kwargs = {k: metadata_db[k] for k in metadata_db if k != "schema"}
        self.pg_pool = psycopg2.pool.SimpleConnectionPool(1, 5, **metadata_connection_kwargs)

    def direct_metadata_query(
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

        conn = None
        try:
            conn = self.pg_pool.getconn()
            cursor = conn.cursor()
            cursor.execute(query, data)
            result = cursor.fetchall() if return_result else (None,)
            if commit:
                cursor.execute("COMMIT;")
        finally:
            if conn:
                self.pg_pool.putconn(conn)

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

        result = self.direct_metadata_query(query, data, commit=True)[0][0]
        return result

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

    def is_tagged(self, visitseq_uuid: UUID, tag: str) -> bool:
        query = sql.SQL("SELECT COUNT(*)>=1 FROM {}.tags WHERE visitseq_uuid={} AND TAG={}").format(
            sql.Identifier(self.metadata_db_schema), sql.Placeholder("visitseq_uuid"), sql.Placeholder("tag")
        )
        data = {"visitseq_uuid": visitseq_uuid, "tag": tag}
        is_tagged = self.direct_metadata_query(query, data)[0][0] > 0
        return is_tagged

    def tag(self, visitseq_uuid: UUID, *tags: str) -> None:
        query = sql.SQL("INSERT INTO {}.tags (visitseq_uuid, tag) VALUES ({}, {})").format(
            sql.Identifier(self.metadata_db_schema), sql.Placeholder("visitseq_uuid"), sql.Placeholder("tag")
        )
        data = {"visitseq_uuid": visitseq_uuid}
        for tag in tags:
            if not self.is_tagged(visitseq_uuid, tag):
                data["tag"] = tag
                self.direct_metadata_query(query, data, commit=True, return_result=False)

    def untag(self, visitseq_uuid: UUID, tag: str) -> None:
        if self.is_tagged(visitseq_uuid, tag):
            query = sql.SQL("DELETE FROM {}.tags WHERE visitseq_uuid={} AND tag={}").format(
                sql.Identifier(self.metadata_db_schema),
                sql.Placeholder("visitseq_uuid"),
                sql.Placeholder("tag"),
            )
            data = {"visitseq_uuid": visitseq_uuid, "tag": tag}
            self.direct_metadata_query(query, data, commit=True, return_result=False)

    def comment(self, visitseq_uuid: UUID, comment: str, author: str | None = None) -> None:
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

        self.direct_metadata_query(query, data, commit=True, return_result=False)

    def get_comments(self, visitseq_uuid: UUID) -> pd.DataFrame:
        psycopg2_query = sql.SQL("SELECT * FROM {}.comments WHERE visitseq_uuid = %s").format(
            sql.Identifier(self.metadata_db_schema)
        )

        conn = None
        try:
            conn = self.pg_pool.getconn()
            text_query = psycopg2_query.as_string(conn)
            comments = pd.read_sql(text_query, conn, params=[visitseq_uuid])
        finally:
            if conn:
                self.pg_pool.putconn(conn)

        # sqlalchemy seems to mess up the dtypes. Fix them.
        comments["comment"] = comments["comment"].astype("string")

        return comments
