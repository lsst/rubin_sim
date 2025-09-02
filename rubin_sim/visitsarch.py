import hashlib
from datetime import date
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.pool
from astropy.time import Time
from lsst.resources import ResourcePath, ResourcePathExpression

USDF_METADATA_DATABASE = MappingProxyType(
    {"database": "opsim_log", "host": "134.79.23.205", "schema": "vsarchive"}
)

TEST_METADATA_DATABASE = MappingProxyType(
    {"database": "opsim_log", "host": "134.79.23.205", "schema": "vsarchtest"}
)


def _dayobs_to_str(dayobs: str | date | int) -> str:
    if isinstance(dayobs, int):
        year = dayobs // 10000
        month = (dayobs // 100) % 100
        day = dayobs & 100
        dayobs = date(year, month, day)

    result = str(dayobs)
    return result


def compute_visits_sha256(visits: pd.DataFrame) -> str:
    recs = visits.to_records()
    visitseq_hash = hashlib.sha256(str(recs.dtype).encode())
    visitseq_hash.update(np.ascontiguousarray(recs).data.tobytes())
    visitseq_sha256 = visitseq_hash.hexdigest()
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

    def direct_metadata_query(self, query: str, commit: bool = False) -> tuple:
        """Run a simple query on the visit sequence database.

        Parameters
        ----------
        query : `str`
            The query to RuntimeError
        commit : `bool`
            Commit the query (e.g. for an INSERT)

        Returns
        -------
        result : `tuple`
            The result of the query
        """

        conn = None
        try:
            conn = self.pg_pool.getconn()
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
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
    ) -> str:
        sha256 = compute_visits_sha256(visits)

        column_names = ["visitseq_sha256", "visitseq_label", "telescope"]
        values = [f"decode('{sha256}', 'hex')", "'" + label + "'", "'" + telescope + "'"]

        if url is not None:
            column_names.append("visitseq_url")
            values.append("'" + url + "'")

        if first_day_obs is not None:
            column_names.append("first_day_obs")
            values.append(_dayobs_to_str(first_day_obs))

        if last_day_obs is not None:
            column_names.append("first_day_obs")
            values.append(_dayobs_to_str(last_day_obs))

        if creation_time is not None:
            assert not creation_time.masked
            assert creation_time.isscalar
            column_names.append("creation_time")
            values.append("'" + creation_time.utc[0].strftime("%Y-%m-%dT%H:%M:%SZ") + "'")

        query = f"""
            INSERT INTO {self.metadata_db_schema}.{table} ({", ".join(column_names)})
            VALUES ({", ".join(values)})
            RETURNING visitseq_uuid;
        """
        result = self.direct_metadata_query(query, commit=True)[0][0]

        return result
