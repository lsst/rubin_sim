# This code is a short-term modification to let older versions of rubin_sim
# read pre-night simulations made using the vsarch code on the tickets/SP-2167
# branch, before that branch is actually merged.

# It consists of a minimial set of functions from that branch needed
# to support fetch_obsloctap_visits, collected into one file.

__all__ = ["_fetch_obsloctap_visits"]

import logging
from datetime import date, datetime, timedelta, timezone
from tempfile import TemporaryDirectory
from typing import Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from astropy.time import Time
from lsst.resources import ResourcePath
from rubin_scheduler.site_models.almanac import Almanac

from rubin_sim import maf
from rubin_sim.maf.stackers import BaseStacker
from rubin_sim.maf.utils.opsim_utils import get_sim_data

LOGGER = logging.getLogger(__name__)
PRENIGHT_INDEX_URL = "s3://rubin:rubin-scheduler-prenight/opsim/prenight_index/"


def _get_visits(
    visits_url: str | ResourcePath,
    query: str = "",
    stackers: Sequence[BaseStacker] = (),
) -> pd.DataFrame:
    """Retrieve visits for a visit sequence.

    Parameters
    ----------
    visits_url : `str`
        The origin of the visits.
    query : `str`
        A query in `pandas.DataFrame.query` syntax to select visits
        from the returned sequence. Applied before stackers are run.
    stackers : `Sequence` [`BaseStacker`]
        A sequence of maf stackers to apply.

    Returns
    -------
    visits: `pd.DataFrame`
        The visits in the sequence.
    """
    origin_rp = ResourcePath(visits_url)

    with TemporaryDirectory() as temp_dir:
        h5_destination = ResourcePath(temp_dir).join("visits.h5")
        h5_destination.transfer_from(origin_rp, "copy")
        visits = pd.read_hdf(h5_destination.ospath, key="observations")

    assert isinstance(visits, pd.DataFrame)

    if len(query) > 0:
        visits.query(query, inplace=True)

    if len(stackers) > 0:
        visit_records = visits.to_records()
        for stacker in stackers:
            visit_records = stacker.run(visit_records)
        visits = pd.DataFrame(visit_records)

    return visits


def _dayobs_to_date(dayobs: str | date | int | Time) -> date:
    """Convert dayobs in as a str, date, int, or astropyTime
    to a python datetime.date.

    Parameters
    ----------
    dayobs: `str` or `datetime.date` or `int` or `astropy.time.Time`
        The date of observation an flexible format.

    Return
    ------
    dayobs_date : `datetime.date`
        The date of observation as a ``datetime.date``.
    """
    match dayobs:
        case int():
            year = dayobs // 10000
            month = (dayobs // 100) % 100
            day = dayobs % 100
            dayobs = date(year, month, day)
        case str():
            dayobs = datetime.fromisoformat(dayobs).date()
        case Time():
            # Pacify type checkers
            assert isinstance(dayobs, Time)
            dayobs_dt = dayobs.to_datetime(timezone=timezone(timedelta(hours=-12)))
            if isinstance(dayobs_dt, datetime):
                dayobs = dayobs_dt.date()
            else:
                assert isinstance(dayobs_dt, np.ndarray)
                assert len(dayobs_dt) == 1
                dayobs = dayobs_dt.item().date()
                assert isinstance(dayobs, date)
        case datetime():
            dayobs = dayobs.date()
        case _:
            assert isinstance(dayobs, date)

    assert isinstance(dayobs, date)
    return dayobs


def _get_prenight_index_from_bucket(
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
    day_obs_date = _dayobs_to_date(day_obs)
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

    return prenights


def _get_prenight_index(
    day_obs: str | int | date,
    telescope: str = "simonyi",
    prenight_index_path: str | ResourcePath | None = None,
) -> pd.DataFrame:
    """Retrieve the pre‑night observation index for a given night.

    Parameters
    ----------
    day_obs : `str` or `int` or `date`
        Night for which to fetch the index.
    telescope : `str`
        Telescope name (default ``simonyi``).
    prenight_index_path :  `str` or `ResourcePath`, optional
        Root path to the bucket files (used only if the database call fails).

    Returns
    -------
    prenights: `pd.DataFrame`
        Pre‑night index indexed by ``visitseq_uuid``.
    """
    if prenight_index_path is None:
        prenight_index_path = PRENIGHT_INDEX_URL
    try:
        prenights = _get_prenight_index_from_bucket(day_obs, telescope, prenight_index_path)
    except FileNotFoundError:
        prenights = pd.DataFrame()

    return prenights


def _select_latest_prenight_sim(
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


def _find_latest_prenight_sim_for_nights(
    first_day_obs: str | None = None,
    last_day_obs: str | None = None,
    tags: tuple[str, ...] = ("ideal", "nominal"),
    telescope: str = "simonyi",
    max_simulation_age: int = 2,
    get_prenight_index_kwargs: dict | None = None,
) -> dict:
    """Find the most recent prenight simulation that covers a night.

    Parameters
    ----------
    first_day_obs : `str` or  `None`
        The date of the evening for the first night for which to get
        a simulation. If `None`, then the current date will be used.
    last_day_obs : `str` or  `None`
        The date of the evening for the last night for which to get
        a simulation. If `None`, then the current date will be used.
    tags : `tuple[str]`
        A tuple of tags to filter simulations by.
        Defaults to ``('ideal', 'nominal')``.
    telescope : `str`
        The telescope to search for (simonyi or auxtel).
        Defaults to simonyi.
    max_simulation_age : `int`
        The maximum age of simulations to consider, in days.
        Simulations older than ``max_simulation_age`` will not be considered.
        Defaults to 2.

    Returns
    -------
    sim_metadata : `dict`
        A dictionary with metadata for the simulation.
    """
    if first_day_obs is None:
        first_day_obs = datetime.now(ZoneInfo("Etc/GMT+12")).date().isoformat()
    if last_day_obs is None:
        last_day_obs = first_day_obs

    if get_prenight_index_kwargs is None:
        get_prenight_index_kwargs = {}
    assert isinstance(get_prenight_index_kwargs, dict)

    sims_for_first_night = _get_prenight_index(first_day_obs, telescope, **get_prenight_index_kwargs)
    if first_day_obs != last_day_obs:
        sims_for_last_night = _get_prenight_index(last_day_obs, telescope, **get_prenight_index_kwargs)
        full_range_sims = set(sims_for_first_night.index) & set(sims_for_last_night.index)
    else:
        full_range_sims = set(sims_for_first_night.index)

    result: dict = {}
    if full_range_sims:
        candidate_sims = sims_for_first_night.loc[tuple(full_range_sims), :]
        maybe_result = _select_latest_prenight_sim(candidate_sims, tags, max_simulation_age)
        if maybe_result is not None:
            assert isinstance(maybe_result, pd.Series)
            result = maybe_result.to_dict()

    return result


def _fetch_sim_for_nights(
    first_day_obs: str | None = None,
    last_day_obs: str | None = None,
    which_sim: ResourcePath | str | dict | None = None,
    stackers: Sequence[BaseStacker] = (),
    get_sim_data_kwargs: dict | None = None,
) -> pd.DataFrame | None:
    """Fetches visit metadata from an opsim database for specified nights.

    Parameters
    ----------
    first_day_obs : `str` or  `None`
        The date of the evening for the first night for which to get
        a simulation. If `None`, then the current date will be used.
    last_day_obs : `str` or  `None`
        The date of the evening for the last night for which to get
        a simulation. If `None`, then the current date will be used.
    which_sim : `ResourcePath` or `str` or `dict` on `None`
        The ``resourcePath`` or URL of the opsim file from which to
        load visits, or the arguments to
        ``find_latest_prenight_sim_for_nights``
        to use to determine which simulation to load. ``None`` uses
        default arguments to ``find_latest_prenight_sim_for_nights``.
        Defaults to ``None``.
    stackers : `Sequence[BaseStacker]`
        A sequence of maf stackers to apply.
    get_sim_data_kwargs : `dict`
        Additional arguments to ``get_sim_data`` to use to load
        the visits.

    Returns
    -------
    visits : `pd.DataFrame`
        A pandas DataFrame containing visit parameters.
    """
    if first_day_obs is None:
        first_day_obs = datetime.now(ZoneInfo("Etc/GMT+12")).date().isoformat()
    if last_day_obs is None:
        last_day_obs = first_day_obs

    opsim_rp: ResourcePath | None = None
    match which_sim:
        case ResourcePath():
            opsim_rp = which_sim
        case str():
            opsim_rp = ResourcePath(which_sim)
        case dict():
            this_sim = _find_latest_prenight_sim_for_nights(first_day_obs, last_day_obs, **which_sim)
            if len(this_sim) == 0:
                raise ValueError("No matching simulations found")
            visitseq_url = this_sim["visitseq_url"]
            if visitseq_url is not None:
                opsim_rp = ResourcePath(visitseq_url)
            elif "opsim" in this_sim["files"]:
                opsim_rp = ResourcePath(this_sim["files"]["opsim"])
            else:
                raise ValueError("No visits found")
        case None:
            this_sim = _find_latest_prenight_sim_for_nights(first_day_obs, last_day_obs)
            if len(this_sim) == 0:
                raise ValueError("No matching simulations found")
            visitseq_url = this_sim["visitseq_url"]
            if visitseq_url is not None:
                opsim_rp = ResourcePath(visitseq_url)
            elif "opsim" in this_sim["files"]:
                opsim_rp = ResourcePath(this_sim["files"]["opsim"])
            else:
                raise ValueError("Ne visits found")
        case _:
            raise NotImplementedError()

    assert isinstance(opsim_rp, ResourcePath)

    if len(stackers) == 0:
        if isinstance(get_sim_data_kwargs, dict) and "stackers" in get_sim_data_kwargs:
            stackers = get_sim_data_kwargs["stackers"]
        else:
            stackers = []
    assert isinstance(stackers, list)

    dayobsiso_requested = maf.DayObsISOStacker in [s.__class__ for s in stackers]
    if not dayobsiso_requested:
        # We want it to filter out dates that were not requested,
        # so add it to the stacker even if it was not requested.
        stackers.append(maf.DayObsISOStacker())

    if opsim_rp.getExtension() in (".db", ".sqlite3"):
        if get_sim_data_kwargs is None:
            get_sim_data_kwargs = {}
        assert isinstance(get_sim_data_kwargs, dict)
        get_sim_data_kwargs["stackers"] = stackers
        visits_recarray = get_sim_data(opsim_rp, **get_sim_data_kwargs)
        visits = pd.DataFrame(visits_recarray)
    else:
        visits = _get_visits(opsim_rp, stackers=stackers)

    LOGGER.debug(f"Loaded {len(visits)} from {opsim_rp}")
    on_requested_dates = (first_day_obs <= visits["day_obs_iso8601"]) & (
        visits["day_obs_iso8601"] <= last_day_obs
    )
    visits = visits.loc[on_requested_dates, :]
    # If it dayobsiso was not requested, do not return it.
    if not dayobsiso_requested:
        visits.drop(columns="day_obs_iso8601", inplace=True)

    return visits


def _fetch_obsloctap_visits(
    day_obs: str | None = None,
    nights: int = 2,
    telescope: str = "simonyi",
    columns: Sequence[str] = (
        "observationStartMJD",
        "fieldRA",
        "fieldDec",
        "rotSkyPos",
        "band",
        "visitExposureTime",
        "night",
        "target_name",
    ),
) -> pd.DataFrame | None:
    """Return visits from latest nominal prenight briefing simulation.

    Parameters
    ----------
    day_obs : `str`
        The day_obs of the night, in YYYY-MM-DD format (e.g. 2025-03-26).
        Default None will use the date of the next sunset.
    nights : `int`
        The number of nights of observations to return.
        Defaults to 2.
    telescope : `str`
        The telescope to get visits for: "simonyi" or "auxtel".
        Defaults to "simonyi".
    colums : `Sequence`
        A sequence of columns from the simulation to include.

    Returns
    -------
    visits : `pd.DataFrame`
        The visits from the prenight simulation.
    """
    # Start with the first night that starts after the reference time,
    # which is the current time by default.
    # So, if the reference time is during a night, it starts with the
    # following night.
    night_bounds = pd.DataFrame(Almanac().sunsets)
    current_mjd = Time.now().mjd
    assert isinstance(current_mjd, float)
    reference_mjd = (Time.now() if day_obs is None else Time(day_obs, format="iso", scale="utc")).mjd
    assert isinstance(reference_mjd, float)
    first_night = night_bounds.query(f"sunset > {reference_mjd}").night.min()
    last_night = first_night + nights - 1

    night_bounds.set_index("night", inplace=True)
    start_mjd = night_bounds.loc[first_night, "sunset"]
    assert isinstance(start_mjd, float)
    end_mjd = night_bounds.loc[last_night, "sunrise"]
    assert isinstance(end_mjd, float)

    first_day_obs = Time(start_mjd - 0.5, format="mjd").iso[:10]
    assert isinstance(first_day_obs, str)
    last_day_obs = Time(end_mjd - 0.5, format="mjd").iso[:10]
    assert isinstance(last_day_obs, str)

    which_sim = {
        "tags": ("ideal", "nominal"),
        "telescope": telescope,
        "max_simulation_age": int(np.ceil(current_mjd - reference_mjd)) + 1,
    }
    visits = _fetch_sim_for_nights(first_day_obs, last_day_obs, which_sim=which_sim)
    if visits is not None:
        assert isinstance(visits, pd.DataFrame)
        visits = visits.loc[:, list(columns)]

    return visits
