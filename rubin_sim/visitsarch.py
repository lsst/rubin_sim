import hashlib
from datetime import date
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.pool
import pytz
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

        db_table = self.metadata_db_schema + "." + table
        query_dest_part = "INSERT INTO " + db_table + "(visitseq_sha256, visitseq_label, telescope"
        query_values_part = f"VALUES (decode('{sha256}', 'hex'), '{label}', '{telescope}'"

        if url is not None:
            query_dest_part += ", visitseq_url"
            query_values_part += f", '{url}'"

        if first_day_obs is not None:
            query_dest_part += ", first_day_obs"
            query_values_part += ", " + _dayobs_to_str(first_day_obs)

        if first_day_obs is not None:
            query_dest_part += ", first_day_obs"
            query_values_part += f", '{_dayobs_to_str(first_day_obs)}'"

        if last_day_obs is not None:
            query_dest_part += ", last_day_obs"
            query_values_part += f", '{_dayobs_to_str(last_day_obs)}'"

        if creation_time is not None:
            query_dest_part += ", creation_time"
            query_values_part += ", '" + creation_time.to_datetime(timezone=pytz.UTC).isoformat() + "'"

        query_dest_part += ") "
        query_values_part += ") "
        query = query_dest_part + query_values_part + "RETURNING visitseq_uuid"

        conn = self.pg_pool.getconn()
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
        finally:
            self.pg_pool.putconn(conn)
        return result[0][0]
