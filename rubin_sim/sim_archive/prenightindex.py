__all__ = ["get_prenight_index", "select_latest_prenight_sim", "get_sim_uuid", "get_sim_index_info"]

import logging
import os
from datetime import date
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd
from lsst.resources import ResourcePath

if "PRENIGHT_INDEX_URL" in os.environ:
    PRENIGHT_INDEX_URL = os.environ["PRENIGHT_INDEX_URL"]
else:
    from rubin_sim.sim_archive.vseqarchive import PRENIGHT_INDEX_URL

from rubin_sim.sim_archive.vseqmetadata import VisitSequenceArchiveMetadata

from .util import dayobs_to_date

MAX_AGE = 365
LOGGER = logging.getLogger(__name__)
TELESCOPES = ("simonyi", "auxtel")


def get_prenight_index_from_database(
    day_obs: str | int | date,
    telescope: str = "simonyi",
    schema: str | None = None,
    host: str | None = None,
    user: str | None = None,
    database: str | None = None,
) -> pd.DataFrame:
    """
    Retrieve the pre‑night observation index for a given night from the
    LSST SIM Archive.

    Parameters
    ----------
    day_obs : `str` or `int` or `date`
        The night for which to fetch the index.  Accepts a date string,
        YYYYMMDD encoded into an integer, or a :class:`datetime.date`
    telescope : `str`, optional
        Telescope name (default ``"simonyi"``).  Passed directly to
        :class:`~rubin_sim.sim_archive.vseqmetadata.VisitSequenceArchiveMetadata`.
    schema : `str` or `None`, optional
        Optional database schema name.
    host : `str` or `None`, optional
        Database host address.
    user : `str` or `None`, optional
        Database user name.
    database : `str` or `None`, optional
        Database name.

    Returns
    -------
    prenights: `pandas.DataFrame`
        A DataFrame indexed by ``visitseq_uuid`` containing all pre‑night
        entries for the requested night.  Columns mirror those returned
        by the underlying `sims_on_nights` call.

    Notes
    -----
    * The function uses a hard‑coded maximum simulation age of
      :data:`MAX_AGE` (365 days) to limit the search.
    * If any of ``host``, ``user`` or ``database`` are supplied, they
      override the defaults in the ``metadata_dsn`` dictionary passed to
      :class:`~rubin_sim.sim_archive.vseqmetadata.VisitSequenceArchiveMetadata`.
    """

    day_obs_date = dayobs_to_date(day_obs)
    assert isinstance(day_obs_date, date)

    metadata_dsn = {}
    if host is not None:
        metadata_dsn["host"] = host
    if user is not None:
        metadata_dsn["user"] = user
    if database is not None:
        metadata_dsn["database"] = database

    visit_seq_archive_metadata = VisitSequenceArchiveMetadata(metadata_dsn, schema)
    prenights = visit_seq_archive_metadata.sims_on_nights(
        day_obs_date, day_obs_date, tags=["prenight"], telescope=telescope, max_simulation_age=MAX_AGE
    ).set_index("visitseq_uuid")
    LOGGER.info(f"Got metadata on {len(prenights)} simulations from database")
    return prenights


def get_prenight_index_from_bucket(
    day_obs: str | int | date,
    telescope: str = "simonyi",
    prenight_index_path: str | ResourcePath = PRENIGHT_INDEX_URL,
) -> pd.DataFrame:
    """
    Load the pre‑night observation index for a given night from a remote
    bucket.

    Parameters
    ----------
    day_obs : `str` or `int` or `date`
        The night for which to fetch the index.  Accepts a date string,
        YYYYMMDD encoded into an integer, a ``datetime.date``
    telescope : `str`, optional
        Telescope name (default ``"simonyi"``).
    prenight_index_path : `str` or `ResourcePath`, optional
        The root path where the pre‑night index
        JSON files are stored.  It can be a string URL or a
        ``ResourcePath`` instance.

    Returns
    -------
    prenights : `pandas.DataFrame`
        A DataFrame indexed by the JSON key (``visitseq_uuid``) containing
        all pre‑night entries for the requested night.  The columns match
        those stored in the JSON file.
    """
    day_obs_date = dayobs_to_date(day_obs)
    assert isinstance(day_obs_date, date)

    prenight_index_path = ResourcePath(prenight_index_path, forceDirectory=True)
    assert isinstance(prenight_index_path, ResourcePath)

    year = day_obs_date.year
    month = day_obs_date.month
    isodate = day_obs_date.isoformat()

    prenight_index_resource_path = (
        prenight_index_path.join(telescope)
        .join(str(year))
        .join(str(month))
        .join(f"{telescope}_prenights_for_{isodate}.json")
    )
    with prenight_index_resource_path.as_local() as local_resource_path:
        prenights = pd.read_json(local_resource_path.ospath, orient="index")

    prenights.index.name = "visitseq_uuid"

    return prenights


def get_prenight_index(
    day_obs: str | int | date,
    telescope: str = "simonyi",
    schema: str | None = None,
    host: str | None = None,
    user: str | None = None,
    database: str | None = None,
    prenight_index_path: str | ResourcePath | None = None,
) -> pd.DataFrame:
    """Retrieve the pre‑night observation index for a given night.

    Parameters
    ----------
    day_obs : `str` or `int` or `date`
        Night for which to fetch the index.
    telescope : `str`
        Telescope name (default ``simonyi``).
    schema : `str` or `None`, optional
        Optional database schema name.
    host : `str` or `None`, optional
        Database host address.
    user : `str` or `None`, optional
        Database user name.
    database : `str` or `None`, optional
        Database name.
    prenight_index_path :  `str` or `ResourcePath`, optional
        Root path to the bucket files (used only if the database call fails).

    Returns
    -------
    prenights: `pd.DataFrame`
        Pre‑night index indexed by ``visitseq_uuid``.
    """
    try:
        # Try the database first
        prenights = get_prenight_index_from_database(day_obs, telescope, schema, host, user, database)
    except Exception:
        # Fall back to the bucket
        LOGGER.info("Database not accessible, falling back on the index in the S3 bucket.")
        if prenight_index_path is None:
            prenight_index_path = PRENIGHT_INDEX_URL
        try:
            prenights = get_prenight_index_from_bucket(day_obs, telescope, prenight_index_path)
        except FileNotFoundError:
            prenights = pd.DataFrame()

    return prenights


def get_sim_uuid(day_obs: int | date | str, sim_date: date, daily_id: int | str, **kwargs: Any) -> UUID:
    """Get the UUID of a simulation given its observation night, creation date,
    and daily ID.

    Parameters
    ----------
    day_obs : `int` or `str` or `date`
        The observation night for which to search for the simulation. Can be a
        date string, YYYYMMDD encoded into an integer, or a
        `datetime.date` object. The date rollover follows SITCOMTN-032
        (-12hr timezone).
    sim_date : `date`
        The creation date of the simulation to find.
    daily_id : `int` or `str`
        The daily ID of the simulation to find.
    **kwargs
        Additional keyword arguments passed to :func:`get_prenight_index`.

    Returns
    -------
    uuid : `uuid.UUID`
        The UUID of the matching simulation.

    Raises
    ------
    ValueError
        If no simulation is found matching the specified criteria.
    """
    day_obs = int(
        f"{day_obs.year:04d}{day_obs.month:02d}{day_obs.day:02d}" if isinstance(day_obs, date) else day_obs
    )
    assert isinstance(day_obs, int)

    maybe_uuid: UUID | None = None

    for telescope in TELESCOPES:
        prenight_index = get_prenight_index(day_obs, telescope, **kwargs)
        matching_sims = (prenight_index.sim_creation_day_obs == sim_date) & (
            prenight_index.daily_id == int(daily_id)
        )
        if np.any(matching_sims):
            matching_uuids = prenight_index.index[matching_sims].values
            assert matching_uuids.shape == (1,)
            maybe_uuid = UUID(str(matching_uuids.item()))

    if maybe_uuid is None:
        raise ValueError(f"No simulation found for {sim_date}, {daily_id}")

    assert isinstance(maybe_uuid, UUID)
    return maybe_uuid


def get_sim_index_info(day_obs: int | date | str, visitseq_uuid: UUID | str, **kwargs: Any) -> pd.Series:
    """Get metadata for a simulation given its observation night and UUID.

    Parameters
    ----------
    day_obs : `int` or `str` or `date`
        The observation night for which to search for the simulation. Can be a
        date string, YYYYMMDD encoded into an integer, or a
        :class:`datetime.date` object. The date rollover follows SITCOMTN-032
        (-12hr timezone).
    visitseq_uuid : `uuid.UUID` or `str`
        The UUID of the simulation to retrieve information for.
    **kwargs
        Additional keyword arguments passed to :func:`get_prenight_index`.

    Returns
    -------
    info : `pandas.Series`
        A Series containing the index information for the simulation.

    Raises
    ------
    ValueError
        If no simulation with the given UUID is found, or if multiple
        simulations are found with the same UUID (indicating a problem in the
        metadata database).
    """
    day_obs = int(
        f"{day_obs.year:04d}{day_obs.month:02d}{day_obs.day:02d}" if isinstance(day_obs, date) else day_obs
    )

    visitseq_uuid_str: str | None = None
    if isinstance(visitseq_uuid, str):
        assert isinstance(visitseq_uuid, str)
        visitseq_uuid_str = visitseq_uuid
        visitseq_uuid = UUID(visitseq_uuid)
    else:
        visitseq_uuid_str = str(visitseq_uuid)

    assert isinstance(visitseq_uuid_str, str)
    assert isinstance(visitseq_uuid, UUID)

    maybe_sim_index_info: None | pd.Series | pd.DataFrame = None
    for telescope in TELESCOPES:
        prenight_index = get_prenight_index(day_obs, telescope, **kwargs)
        if not isinstance(prenight_index.index[0], str):
            prenight_index.index = prenight_index.index.map(str)
        if visitseq_uuid_str in prenight_index.index:
            maybe_sim_index_info = prenight_index.loc[visitseq_uuid_str, :]
            maybe_sim_index_info[prenight_index.index.name] = visitseq_uuid

    if isinstance(maybe_sim_index_info, pd.DataFrame):
        raise ValueError(
            f"Multiple sims for UUID {visitseq_uuid}: there is a problem in the sim metadata database!"
        )

    if maybe_sim_index_info is None:
        raise ValueError(f"No sim with UUID {visitseq_uuid} found")

    assert isinstance(maybe_sim_index_info, pd.Series)
    return maybe_sim_index_info


def select_latest_prenight_sim(
    prenights: pd.DataFrame,
    tags: tuple[str, ...] = ("ideal", "nominal"),
    max_simulation_age: int = 2,
) -> dict | None:
    """Select the best prenight simulation from a DataFrame.

    Parameters
    ----------
    prenights: `pd.DataFrame`
        The table of prenights from which to select.
    tags : `tuple` [`str`]
        A tuple of tags to filter simulations by.
        Defaults to ``('ideal', 'nominal')``.
    max_simulation_age : `int`
        The maximum age of simulations to consider, in days.
        Simulations older than ``max_simulation_age`` will not be considered.
        Defaults to 2.

    Returns
    -------
    best_sim : `dict` or `None`
        A dictionary with metadata for the simulation.
    """
    best_sim = None
    for _, sim in prenights.reset_index().iterrows():
        if not set(tags).issubset(sim["tags"]):
            continue
        if best_sim is not None:
            if sim["creation_time"] < best_sim["creation_time"]:
                continue

        age = (pd.Timestamp.now(tz="UTC") - sim["creation_time"]).days
        if age > max_simulation_age:
            continue

        best_sim = sim

    if best_sim is not None:
        LOGGER.info(f"Most recent simulation meeting requested criteria is {best_sim['visitseq_url']}.")
    else:
        LOGGER.debug("No simulations met the requested criteria.")

    return best_sim
