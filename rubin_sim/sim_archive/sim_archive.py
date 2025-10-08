"""Tools for maintaining the archive of opsim output and metadata."""

__all__ = [
    "make_sim_data_dir",
    "drive_sim",
    "find_latest_prenight_sim_for_nights",
    "fetch_sim_for_nights",
    "fetch_obsloctap_visits",
    "fetch_sim_stats_for_night",
]

import hashlib
import logging
import lzma
import os
import pickle
import shutil
import socket
from numbers import Integral, Number
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
import rubin_scheduler
import yaml
from astropy.time import Time
from lsst.resources import ResourcePath
from rubin_scheduler.scheduler import sim_runner
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.schedulers import CoreScheduler
from rubin_scheduler.scheduler.utils import SchemaConverter

from .future_vsarch import _fetch_obsloctap_visits as fetch_obsloctap_visits

LOGGER = logging.getLogger(__name__)


def make_sim_data_dir(
    observations: np.recarray,
    reward_df: pd.DataFrame | None = None,
    obs_rewards: pd.DataFrame | None = None,
    in_files: dict = {},
    sim_runner_kwargs: dict = {},
    tags: list = [],
    label: str | None = None,
    data_path: str | Path | None = None,
    opsim_metadata: dict | None = None,
) -> Path | str:
    """Create or fill a local simulation archive directory.

    Parameters
    ----------
    observations : `numpy.recarray`
        The observations data, in the "obs" format as accepted and
        created by `rubin_scheduler.scheduler.utils.SchemaConverter`.
    reward_df : `pandas.DataFrame`, optional
        The reward data, by default None.
    obs_rewards : `pandas.DataFrame`, optional
        The observation rewards data, by default None.
    in_files : `dict`, optional
        Additional input files to be included in the archive,
        by default {}.
    sim_runner_kwargs : `dict`, optional
        Additional simulation runner keyword arguments, by default {}.
    tags : `list` [`str`], optional
        A list of tags/keywords to be included in the metadata, by
        default [].
    label : `str`, optional
        A label to be included in the metadata, by default None.
    data_path : `str` or `pathlib.Path`, optional
        The path to the simulation archive directory, by default None.
    opsim_metadata : `dict`
        Metadata to be included.

    Returns
    -------
    data_dir : `pathlib.Path` or `tempfile.TemporaryDirectory`
        The temporary directory containing the simulation archive.
    """
    if data_path is None:
        data_dir = TemporaryDirectory()
        data_path = Path(data_dir.name)
    else:
        data_dir = None

        if not isinstance(data_path, Path):
            data_path = Path(data_path)

    files = {}

    # Save the observations
    files["observations"] = {"name": "opsim.db"}
    opsim_output_fname = data_path.joinpath(files["observations"]["name"])
    SchemaConverter().obs2opsim(observations, filename=opsim_output_fname)

    # Save the rewards
    if reward_df is not None and obs_rewards is not None:
        files["rewards"] = {"name": "rewards.h5"}
        rewards_fname = data_path.joinpath(files["rewards"]["name"])
        if reward_df is not None:
            reward_df.to_hdf(rewards_fname, key="reward_df")
        if obs_rewards is not None:
            obs_rewards.to_hdf(rewards_fname, key="obs_rewards")

    # Save basic statistics
    files["statistics"] = {"name": "obs_stats.txt"}
    stats_fname = data_path.joinpath(files["statistics"]["name"])
    with open(stats_fname, "w") as stats_io:
        print(SchemaConverter().obs2opsim(observations).describe().T.to_csv(sep="\t"), file=stats_io)

    # Add supplied files
    for file_type, fname in in_files.items():
        files[file_type] = {"name": Path(fname).name}
        try:
            shutil.copyfile(fname, data_path.joinpath(files[file_type]["name"]))
        except shutil.SameFileError:
            pass

    # Add file hashes
    for file_type in files:
        fname = data_path.joinpath(files[file_type]["name"])
        with open(fname, "rb") as file_io:
            content = file_io.read()

        files[file_type]["md5"] = hashlib.md5(content).hexdigest()

    def convert_mjd_to_dayobs(mjd: float) -> str:
        # Use dayObs defn. from SITCOMTN-32: https://sitcomtn-032.lsst.io/
        evening_local_mjd = np.floor(mjd - 0.5).astype(int)
        evening_local_iso = Time(evening_local_mjd, format="mjd").iso[:10]
        assert isinstance(evening_local_iso, str)
        return evening_local_iso

    if opsim_metadata is None:
        opsim_metadata = {}

    opsim_metadata["scheduler_version"] = rubin_scheduler.__version__
    opsim_metadata["host"] = socket.getfqdn()
    opsim_metadata["username"] = os.environ["USER"]

    simulation_dates = {}
    if "sim_start_mjd" in sim_runner_kwargs:
        simulation_dates["first"] = convert_mjd_to_dayobs(sim_runner_kwargs["sim_start_mjd"])

        if "sim_duration" in sim_runner_kwargs:
            simulation_dates["last"] = convert_mjd_to_dayobs(
                sim_runner_kwargs["sim_start_mjd"] + sim_runner_kwargs["sim_duration"] - 1
            )
    else:
        simulation_dates["first"] = convert_mjd_to_dayobs(observations["mjd"].min())
        simulation_dates["last"] = convert_mjd_to_dayobs(observations["mjd"].max())

    if len(sim_runner_kwargs) > 0:
        opsim_metadata["sim_runner_kwargs"] = {}
        for key, value in sim_runner_kwargs.items():
            # Cast numpy number types to ints, floats, and reals to avoid
            # confusing the yaml module.
            match value:
                case bool():
                    opsim_metadata["sim_runner_kwargs"][key] = value
                case Integral():
                    opsim_metadata["sim_runner_kwargs"][key] = int(value)
                case Number():
                    opsim_metadata["sim_runner_kwargs"][key] = float(value)
                case _:
                    opsim_metadata["sim_runner_kwargs"][key] = str(value)

    opsim_metadata["simulated_dates"] = simulation_dates
    opsim_metadata["files"] = files

    if len(tags) > 0:
        for tag in tags:
            assert isinstance(tag, str), "Tags must be strings."
        opsim_metadata["tags"] = tags

    if label is not None:
        assert isinstance(label, str), "The sim label must be a string."
        opsim_metadata["label"] = label

    sim_metadata_fname = data_path.joinpath("sim_metadata.yaml")
    with open(sim_metadata_fname, "w") as sim_metadata_io:
        print(yaml.dump(opsim_metadata, indent=4), file=sim_metadata_io)

    files["metadata"] = {"name": sim_metadata_fname}

    if data_dir is not None:
        # If we created a temporary directory, if we do not return it, it
        # will get automatically cleaned up, losing our work.
        # So, return it.
        return data_dir

    return data_path


def drive_sim(
    observatory: ModelObservatory,
    scheduler: CoreScheduler,
    label: str | None = None,
    tags: list = [],
    script: str | None = None,
    notebook: str | None = None,
    opsim_metadata: dict | None = None,
    **kwargs: Any,
) -> tuple:
    """Run a simulation and archive the results.

    Parameters
    ----------
    observatory : `ModelObservatory`
        The model for the observatory.
    scheduler : `CoreScheduler`
        The scheduler to use.
    archive_uri : `str`, optional
        The root URI of the archive resource into which the results should
        be stored. Defaults to None.
    label : `str`, optional
        The label for the simulation in the archive. Defaults to None.
    tags : `list` of `str`, optional
        The tags for the simulation in the archive. Defaults to an
        empty list.
    script : `str`
        The filename of the script producing this simulation.
        Defaults to None.
    notebook : `str`, optional
        The filename of the notebook producing the simulation.
        Defaults to None.
    opsim_metadata : `dict`, optional
        Extra metadata to store in the archive.

    Returns
    -------
    observatory : `ModelObservatory`
        The model for the observatory.
    scheduler : `CoreScheduler`
        The scheduler used.
    observations : `numpy.recarray`
        The observations produced.
    reward_df : `pandas.DataFrame`, optional
        The table of rewards. Present if `record_rewards`
        or `scheduler.keep_rewards` is True.
    obs_rewards : `pandas.Series`, optional
        The mapping of entries in reward_df to observations. Present if
        `record_rewards` or `scheduler.keep_rewards` is True.
    resource_path : `ResourcePath`, optional
        The resource path to the archive of the simulation. Present if
        `archive_uri` was set.

    Notes
    -----
    Additional parameters not described above will be passed into
    `sim_runner`.

    If the `archive_uri` parameter is not supplied, `sim_runner` is run
    directly, so that `drive_sim` can act as a drop-in replacement of
    `sim-runner`.

    In a jupyter notebook, the notebook can be saved for the notebook
    paramater using `%notebook $notebook_fname` (where `notebook_fname`
    is variable holding the filename for the notebook) in the cell prior
    to calling `drive_sim`.
    """
    if "record_rewards" in kwargs:
        if kwargs["record_rewards"] and not scheduler.keep_rewards:
            raise ValueError("To keep rewards, scheduler.keep_rewards must be True")
    else:
        kwargs["record_rewards"] = scheduler.keep_rewards

    in_files = {}
    if script is not None:
        in_files["script"] = script

    if notebook is not None:
        in_files["notebook"] = notebook

    with TemporaryDirectory() as local_data_dir:
        LOGGER.debug(f"Using temporary directory {local_data_dir}.")
        # We want to store the state of the scheduler at the start of
        # the sim, so we need to save it now before we run the simulation.
        scheduler_path = Path(local_data_dir).joinpath("scheduler.pickle.xz")
        with lzma.open(scheduler_path, "wb", format=lzma.FORMAT_XZ) as pio:
            pickle.dump(scheduler, pio)
            in_files["scheduler"] = scheduler_path.as_posix()

        LOGGER.debug("About to call sim_runner.")
        sim_results = sim_runner(observatory, scheduler, **kwargs)
        LOGGER.debug("sim_runner complete.")

        observations = sim_results[2]
        reward_df = sim_results[3] if scheduler.keep_rewards else None
        obs_rewards = sim_results[4] if scheduler.keep_rewards else None

        data_dir = make_sim_data_dir(
            observations,
            reward_df=reward_df,
            obs_rewards=obs_rewards,
            in_files=in_files,
            sim_runner_kwargs=kwargs,
            tags=tags,
            label=label,
            opsim_metadata=opsim_metadata,
        )

        resource_path = ResourcePath(data_dir.name, forceDirctory=True)  # type: ignore

    results = sim_results + (resource_path,)
    return results


def find_latest_prenight_sim_for_nights(
    first_day_obs: str | None = None,
    last_day_obs: str | None = None,
    tags: tuple[str] = ("ideal", "nominal"),
    telescope: str = "simonyi",
    max_simulation_age: int = 2,
    archive_uri: str = "s3://rubin:rubin-scheduler-prenight/opsim/",
    compilation_uri: str = "s3://rubin:rubin-scheduler-prenight/opsim/compiled_metadata_cache.h5",
) -> pd.DataFrame:
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
    archive_uri : `str`
        The URI of the archive from which to fetch the simulation.
        Defaults to ``s3://rubin:rubin-scheduler-prenight/opsim/``.
    compilation_uri : `str`
        The URI of the compiled metadata HDF5 file for efficient querying.
        Defaults to
        ``s3://rubin:rubin-scheduler-prenight/opsim/compiled_metadata_cache.h5``.

    Returns
    -------
    sim_metadata : `dict`
        A dictionary with metadata for the simulation.
    """
    raise NotImplementedError()


def fetch_sim_for_nights(
    first_day_obs: str | None = None,
    last_day_obs: str | None = None,
    which_sim: ResourcePath | str | dict | None = None,
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
    git_sim_data_kwargs : `dict`
        Additional arguments to ``get_sim_data`` to use to load
        the visits.

    Returns
    -------
    visits : `pd.DataFrame`
        A pandas DataFrame containing visit parameters.
    """
    # should call vseqarchive.get_visits
    raise NotImplementedError()


def old_fetch_obsloctap_visits(
    day_obs: str | None = None, nights: int = 2, telescope: str = "simonyi"
) -> pd.DataFrame:
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

    Returns
    -------
    visits : `pd.DataFrame`
        The visits from the prenight simulation.
    """
    raise NotImplementedError


def fetch_sim_stats_for_night(day_obs: str | int | None = None) -> dict:
    """Count the visits on a night in the latest nominal sim for a night.

    Parameters
    ----------
    day_obs : `str` or 'int' or `None`
        Date (in UTC-12hrs timezone) for which to get the count of visits,
        in ISO8601 (YYYY-MM-DD as a string) or int dayobs (int(YYYYMMDD))
        or `None` (day_obs including the evening of yesterday in local time).

    Returns
    -------
    sim_stats : `dict`
        A dict with statistics for the night.
        Presently, it has one key: `nominal_visits`, the number of visits
        in the latest nominal simulation.
    """
    raise NotImplementedError()
