"""Tools for maintaining the prototype archive of opsim output and metadata."""

__all__ = [
    "make_sim_archive_dir",
    "transfer_archive_dir",
    "check_opsim_archive_resource",
    "read_archived_sim_metadata",
    "make_sim_archive_cli",
    "compile_sim_metadata",
    "read_sim_metadata_from_hdf",
    "verify_compiled_sim_metadata",
    "compile_sim_archive_metadata_cli",
    "find_latest_prenight_sim_for_nights",
    "fetch_sim_for_nights",
    "fetch_obsloctap_visits",
    "fetch_sim_stats_for_night",
    "export_sim_to_prototype_sim_archive",
]

import argparse
import datetime
import hashlib
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

import numpy as np
import pandas as pd
import yaml
from astropy.time import Time
from lsst.resources import ResourcePath
from rubin_scheduler.scheduler.utils import SchemaConverter
from rubin_scheduler.site_models.almanac import Almanac

from rubin_sim import maf
from rubin_sim.maf.utils.opsim_utils import get_sim_data
from rubin_sim.sim_archive import vseqarchive

from .sim_archive import make_sim_data_dir as make_sim_archive_dir

LOGGER = logging.getLogger(__name__)

try:
    from lsst.resources import ResourcePath
except ModuleNotFoundError:
    LOGGER.error("Module lsst.resources required to use rubin_sim.sim_archive.")


def _next_sim_date_and_index(archive_base_uri: str) -> tuple:
    insert_date = datetime.datetime.now(datetime.UTC).date().isoformat()
    insert_date_rpath = ResourcePath(archive_base_uri).join(insert_date, forceDirectory=True)
    if not insert_date_rpath.exists():
        insert_date_rpath.mkdir()
        LOGGER.debug(f"Created {insert_date_rpath}.")

    # Number the sims in the insert date dir by
    # looing for all the interger directories, and choosing the next one.
    found_ids = []
    for base_dir, found_dirs, found_files in insert_date_rpath.walk():
        if base_dir == insert_date_rpath:
            for found_dir in found_dirs:
                try:
                    found_dir_index = found_dir[:-1] if found_dir.endswith("/") else found_dir
                    found_ids.append(int(found_dir_index))
                except ValueError:
                    pass

    new_id = max(found_ids) + 1 if len(found_ids) > 0 else 1
    resource_rpath = insert_date_rpath.join(f"{new_id}", forceDirectory=True)
    resource_rpath.mkdir()
    LOGGER.debug(f"Created {resource_rpath}.")
    return insert_date, new_id, resource_rpath


def transfer_archive_dir(
    archive_dir: str, archive_base_uri: str = "s3://rubin:rubin-scheduler-prenight/opsim/"
) -> ResourcePath:
    """Transfer the contents of an archive directory to an resource.

    Parameters
    ----------
    archive_dir : `str`
        The path to the archive directory containing the files to be
        transferred.
    archive_base_uri : `str`, optional
        The base URI where the archive files will be transferred to.
        Default is "s3://rubin:rubin-scheduler-prenight/opsim/".

    Returns
    -------
    resource_rpath : `ResourcePath`
        The destination resource.
    """

    LOGGER.debug(f"Beginning copy of {archive_dir} to {archive_base_uri}.")
    metadata_fname = Path(archive_dir).joinpath("sim_metadata.yaml")
    with open(metadata_fname, "r") as metadata_io:
        sim_metadata = yaml.safe_load(metadata_io)
        LOGGER.debug(f"Completed read of {archive_dir}.")

    insert_date, new_id, resource_rpath = _next_sim_date_and_index(archive_base_uri)

    # Include the metadata file itself.
    sim_metadata["files"]["metadata"] = {"name": "sim_metadata.yaml"}

    for file_info in sim_metadata["files"].values():
        source_fname = Path(archive_dir).joinpath(file_info["name"])
        with open(source_fname, "rb") as source_io:
            content = source_io.read()
            LOGGER.debug(f"Read {source_fname}.")

        destination_rpath = resource_rpath.join(file_info["name"])
        destination_rpath.write(content)

        LOGGER.info(f"Copied {source_fname} to {destination_rpath}")

    return resource_rpath


def check_opsim_archive_resource(archive_uri: str) -> dict:
    """Check the contents of an opsim archive resource.

    Parameters
    ----------
    archive_uri : `str`
        The URI of the archive resource to be checked.

    Returns
    -------
    validity: `dict`
        A dictionary of files checked, and their validity.
    """
    LOGGER.debug(f"Starting to check file hashes in opsim sim archive {archive_uri}.")
    metadata_path = ResourcePath(archive_uri).join("sim_metadata.yaml")
    with metadata_path.open(mode="r") as metadata_io:
        sim_metadata = yaml.safe_load(metadata_io)
        LOGGER.debug(f"Read sim metadata from {metadata_path}.)")

    results = {}

    for file_info in sim_metadata["files"].values():
        resource_path = ResourcePath(archive_uri).join(file_info["name"])
        LOGGER.info(f"Reading {resource_path}.")
        content = resource_path.read()

        results[file_info["name"]] = file_info["md5"] == hashlib.md5(content).hexdigest()
        if results[file_info["name"]]:
            LOGGER.debug(f"{resource_path} checked and found to match recorded md5.")
        else:
            LOGGER.debug(f"{resource_path} has an md5 that differs from the recorded md5!")

    return results


def _build_archived_sim_label(base_uri: str, metadata_resource: ResourcePath, metadata: dict) -> str:
    label_base = metadata_resource.dirname().geturl().removeprefix(base_uri).rstrip("/").lstrip("/")

    # If a label is supplied by the metadata, use it
    if "label" in metadata:
        label = f"{label_base} {metadata['label']}"
        return label

    try:
        sim_dates = metadata["simulated_dates"]
        first_date = sim_dates["first"]
        last_date = sim_dates["last"]
        label = f"{label_base} of {first_date}"
        if last_date != first_date:
            label = f"{label} through {last_date}"
    except KeyError:
        label = label_base

    if "scheduler_version" in metadata:
        label = f"{label} with {metadata['scheduler_version']}"

    return label


def read_archived_sim_metadata(
    base_uri: str,
    latest: str = None,
    num_nights: int = 5,
    compilation_resource: ResourcePath | str | None = None,
    verify_compilation: bool = False,
) -> dict:
    """Read metadata for a time range of archived opsim output.

    Parameters
    ----------
    base_uri : `str`
        The base URI of the archive resource to be checked.
    latest : `str`, optional
        The date of the latest simulation whose metadata should be loaded.
        This is the date on which the simulations was added to the archive,
        not necessarily the date on which the simulation was run, or any
        of the dates simulated.
        Default is today.
    num_nights : `int`
        The number of nights of the date window to load.
    compilation_resource : `ResourcePath` or `str` or  `None`
        The ResourcePath to an hdf5 compilation of metadata.
    verify_compilation : `bool`
        Verify that metadata in compilation corresponds to an existing
        resource, and the all existing resources have metadata. Defaults
        to False, which will work provided that the compilation is complete
        and correct up to the date of the last simulation it includes.

    Returns
    -------
    sim_metadata: `dict`
        A dictionary of metadata for simulations in the date range.
    """
    latest_mjd = int(Time.now().mjd if latest is None else Time(latest).mjd)
    earliest_mjd = int(latest_mjd - (num_nights - 1))
    LOGGER.debug(
        f"Looking for simulation metadata with MJD between {earliest_mjd} and {latest_mjd} in {base_uri}."
    )

    compilation = {}
    compiled_uris_by_date = {}
    max_compiled_date = "1900-01-01"
    if compilation_resource is not None:
        LOGGER.debug(f"Reading metadata cache {compilation_resource}.")
        try:
            compilation.update(read_sim_metadata_from_hdf(compilation_resource))
            for uri in compilation:
                iso_date = Path(urlparse(uri).path).parts[-2]
                if iso_date not in compiled_uris_by_date:
                    compiled_uris_by_date[iso_date] = []
                compiled_uris_by_date[iso_date].append(uri)
                max_compiled_date = max(max_compiled_date, iso_date)
            LOGGER.debug(f"Maximum simulation execution date in metadata cache: {max_compiled_date}")
        except FileNotFoundError:
            LOGGER.warning(f"No metadata cache {compilation_resource}, not using cache.")
            pass

    all_metadata = {}
    for mjd in range(earliest_mjd, latest_mjd + 1):
        iso_date = Time(mjd, format="mjd").iso[:10]

        # Make the comparison >=, rather than >, so
        # it won't miss sims in which the compilation does not complete
        # the final date.
        if verify_compilation or (iso_date >= max_compiled_date):
            date_resource = ResourcePath(base_uri).join(iso_date, forceDirectory=True)
            if date_resource.exists():
                for base_dir, found_dirs, found_files in date_resource.walk(
                    file_filter=r".*sim_metadata.yaml"
                ):
                    for found_file in found_files:
                        found_resource = ResourcePath(base_dir).join(found_file)
                        LOGGER.debug(f"Found {found_resource}")
                        sim_uri = str(found_resource.dirname())
                        if sim_uri in compilation:
                            LOGGER.debug(f"Not reading {found_resource}, already in the read compliation.")
                            these_metadata = compilation[sim_uri]
                        else:
                            LOGGER.debug(f"Reading {found_resource} (absent from compilation).")
                            these_metadata = yaml.safe_load(found_resource.read().decode("utf-8"))
                            these_metadata["label"] = _build_archived_sim_label(
                                base_uri, found_resource, these_metadata
                            )
                            LOGGER.debug(f"Read successfully: {found_resource}")
                            if iso_date < max_compiled_date:
                                LOGGER.error(
                                    f"Simulation at {sim_uri} expected but not found in compilation."
                                )
                        all_metadata[sim_uri] = these_metadata
            else:
                LOGGER.debug(f"No simulations found with generation date of {iso_date}")
        else:
            if iso_date in compiled_uris_by_date:
                for sim_uri in compiled_uris_by_date[iso_date]:
                    all_metadata[sim_uri] = compilation[sim_uri]

        if verify_compilation:
            if iso_date in compiled_uris_by_date:
                for sim_uri in compiled_uris_by_date[iso_date]:
                    if sim_uri not in all_metadata:
                        message = f"Simulation at {sim_uri} in compiled metadata but not archive."
                        print(message)
                        LOGGER.error(message)
            else:
                LOGGER.debug(
                    f"Date {iso_date} not expected to be in the metadata compilation, not checking for it."
                )

    if len(all_metadata) == 0:
        earliest_iso = Time(earliest_mjd, format="mjd").iso[:10]
        latest_iso = Time(latest_mjd, format="mjd").iso[:10]
        LOGGER.info(f"No simulations run between {earliest_iso} through {latest_iso} found in {base_uri}")

    return all_metadata


def make_sim_archive_cli(*args: Any) -> str:
    parser = argparse.ArgumentParser(description="Add files to sim archive")
    parser.add_argument(
        "label",
        type=str,
        help="A label for the simulation.",
    )
    parser.add_argument(
        "opsim",
        type=str,
        help="File name of opsim database.",
    )
    parser.add_argument("--rewards", type=str, default=None, help="A rewards HDF5 file.")
    parser.add_argument(
        "--scheduler_version",
        type=str,
        default=None,
        help="The version of the scheduler that producte the opsim database.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        help="A snapshot of the scheduler used to produce the database, at the start of the simulation.",
    )
    parser.add_argument(
        "--script", type=str, default=None, help="The file name of the script run to create the simulation."
    )

    notebook_help = "The file name of the notebook run to create the simulation."
    notebook_help = notebook_help + " This can be produced using the %%notebook magic."
    parser.add_argument(
        "--notebook",
        type=str,
        default=None,
        help=notebook_help,
    )
    parser.add_argument(
        "--archive_base_uri",
        type=str,
        default="s3://rubin:rubin-scheduler-prenight/opsim/",
        help="Base URI for the archive",
    )
    parser.add_argument("--tags", type=str, default=[], nargs="*", help="The tags on the simulation.")
    parser.add_argument("--telescope", type=str, default="simonyi", help="The telescope simulated.")
    arg_values = parser.parse_args() if len(args) == 0 else parser.parse_args(args)

    observations = SchemaConverter().opsim2obs(arg_values.opsim)

    if arg_values.rewards is not None:
        try:
            reward_df = pd.read_hdf(arg_values.rewards, "reward_df")
        except KeyError:
            reward_df = None

        try:
            obs_rewards = pd.read_hdf(arg_values.rewards, "obs_rewards")
        except KeyError:
            obs_rewards = None
    else:
        reward_df = None
        obs_rewards = None

    filename_args = ["scheduler", "script", "notebook"]
    in_files = {}
    for filename_arg in filename_args:
        try:
            filename = getattr(arg_values, filename_arg)
            if filename is not None:
                in_files[filename] = filename
        except AttributeError:
            pass

    data_path = make_sim_archive_dir(
        observations,
        reward_df,
        obs_rewards,
        in_files,
        tags=arg_values.tags,
        label=arg_values.label,
        opsim_metadata={"telescope": arg_values.telescope},
    )
    LOGGER.info(f"Created simulation archived directory: {data_path.name}")

    sim_archive_uri = transfer_archive_dir(data_path.name, arg_values.archive_base_uri)
    LOGGER.info(f"Transferred {data_path} to {sim_archive_uri}")

    return sim_archive_uri


def compile_sim_metadata(
    archive_uri: str, compilation_resource: str | ResourcePath, num_nights: int = 10000, append: bool = False
) -> ResourcePath:
    """Read sim archive metadata and export it to tables in an hdf5 files.

    Parameters
    ----------
    archive_uri : `str`
        URI of the sim archive from which to read metadata.
    compilation_resource : `str` or `ResourcePath`
        Resource for hdf5 file to be written to
    num_nights : `int`, optional
        Number of nights to include, by default 10000
    append : `bool`, optional
        Do not rebuild the whole compilation, but instead reread what is
        there already, and add new metadata after the last date already
        include. Defaults to False.

    Returns
    -------
    compilation_fname : `ResourcePath`
        The resource to which the hdf5 file was written.
    """
    LOGGER.debug("Starting compile_sim_metadata.")

    if append:
        sim_metadata = read_archived_sim_metadata(
            archive_uri, num_nights=num_nights, compilation_resource=compilation_resource
        )
    else:
        sim_metadata = read_archived_sim_metadata(archive_uri, num_nights=num_nights)

    sim_rows = []
    file_rows = []
    sim_runner_kwargs = []
    tags = []
    for uri, metadata in list(sim_metadata.items()):
        sim_row = {"sim_uri": uri}
        for key, value in metadata.items():
            match key:
                case "files":
                    for file_type, file_metadata in value.items():
                        this_file = {"sim_uri": uri, "file_type": file_type} | file_metadata
                        file_rows.append(this_file)
                case "sim_runner_kwargs":
                    these_args = {"sim_uri": uri} | value
                    sim_runner_kwargs.append(these_args)
                case "tags":
                    for tag in value:
                        tags.append({"sim_uri": uri, "tag": tag})
                case "simulated_dates":
                    sim_row["first_simulated_date"] = value["start"] if "start" in value else value["first"]
                    sim_row["last_simulated_date"] = value["end"] if "end" in value else value["last"]
                case _:
                    sim_row[key] = value
        sim_rows.append(sim_row)

    sim_df = pd.DataFrame(sim_rows).set_index("sim_uri")
    file_df = pd.DataFrame(file_rows).set_index("sim_uri")
    sim_runner_kwargs_df = pd.DataFrame(sim_runner_kwargs).set_index("sim_uri")
    tags_df = pd.DataFrame(tags).set_index("sim_uri")

    with TemporaryDirectory() as local_data_dir:
        local_hdf_fname = Path(local_data_dir).joinpath("scheduler.pickle.xz")
        sim_df.to_hdf(local_hdf_fname, key="simulations", format="table")
        file_df.to_hdf(local_hdf_fname, key="files", format="table")
        sim_runner_kwargs_df.to_hdf(local_hdf_fname, key="kwargs", format="table")
        tags_df.to_hdf(local_hdf_fname, key="tags", format="table")

        with open(local_hdf_fname, "rb") as local_hdf_io:
            hdf_bytes = local_hdf_io.read()

        if isinstance(compilation_resource, str):
            compilation_resource = ResourcePath(compilation_resource)

        assert isinstance(compilation_resource, ResourcePath)
        LOGGER.info(f"Writing metadata compilation to {compilation_resource}")
        compilation_resource.write(hdf_bytes)

    return compilation_resource


def read_sim_metadata_from_hdf(compilation_resource: str | ResourcePath) -> dict:
    """Read sim archive metadata from an hdf5 file.
    Return a dict as if it were generated by read_archived_sim_metadata.

    Parameters
    ----------
    compilation_fname : `str` or `ResourcePath`
        The source of the hdf5 data to read.

    Returns
    -------
    sim_archive_metadata: `dict`
        A nested dictionary with the simulation metadata.
    """

    if isinstance(compilation_resource, str):
        compilation_resource = ResourcePath(compilation_resource)
    assert isinstance(compilation_resource, ResourcePath)

    with compilation_resource.as_local() as local_compilation_resource:
        compilation_fname: str = local_compilation_resource.ospath
        LOGGER.debug(f"{compilation_resource} copied to {compilation_fname}.")
        sim_df = pd.read_hdf(compilation_fname, "simulations")
        file_df = pd.read_hdf(compilation_fname, "files")
        sim_runner_kwargs_df = pd.read_hdf(compilation_fname, "kwargs")
        tags_df = pd.read_hdf(compilation_fname, "tags")

    def make_file_dict(g: pd.Series | pd.DataFrame) -> dict:
        if isinstance(g, pd.Series):
            g = g.to_frame().T

        return g.set_index("file_type").T.to_dict()

    sim_df["files"] = file_df.groupby("sim_uri").apply(make_file_dict)

    def make_kwargs_dict(g):
        sim_kwargs = g.to_dict(orient="records")[0]
        # Keyword args that are not set get recorded as nans.
        # Do not include them in the dictionary.
        nan_keys = []
        for key in sim_kwargs:
            try:
                if np.isnan(sim_kwargs[key]):
                    nan_keys.append(key)
            except TypeError:
                pass

        for key in nan_keys:
            del sim_kwargs[key]

        return sim_kwargs

    sim_df["sim_runner_kwargs"] = sim_runner_kwargs_df.groupby("sim_uri").apply(make_kwargs_dict)

    sim_df["tags"] = tags_df.groupby("sim_uri").agg({"tag": list})

    sim_metadata = sim_df.to_dict(orient="index")

    for sim_uri in sim_metadata:
        # Return begin and end date to their nested dict format.
        sim_metadata[sim_uri]["simulated_dates"] = {
            "first": sim_metadata[sim_uri]["first_simulated_date"],
            "last": sim_metadata[sim_uri]["last_simulated_date"],
        }
        del sim_metadata[sim_uri]["first_simulated_date"]
        del sim_metadata[sim_uri]["last_simulated_date"]

        # Unset keys show up as nans.
        # Do not put them in the resultant dict.
        nan_keys = []
        for key in sim_metadata[sim_uri]:
            try:
                if np.isnan(sim_metadata[sim_uri][key]):
                    nan_keys.append(key)
            except TypeError:
                pass

        for key in nan_keys:
            del sim_metadata[sim_uri][key]

    return sim_metadata


def verify_compiled_sim_metadata(
    archive_uri: str, compilation_resource: str | ResourcePath, num_nights: int = 10000
) -> list[dict]:
    """Verify that a compilation of sim archive metadata matches directaly
    read metadata.

    Parameters
    ----------
    archive_uri : `str`
        Archive from which to directly read metadata.
    compilation_resource : `str` or `ResourcePath`
        Resource for the metadata compilation
    num_nights : `int`, optional
        number of nights to check, by default 10000

    Returns
    -------
    differences : `list[dict]`
        A list of dicts describing differences. If they match, it will return
        and empty list.
    """

    direct_sim_metadata = read_archived_sim_metadata(archive_uri, num_nights=num_nights)

    try:
        # One old sim uses a couple of non-standard keywords, so update them.
        simulated_dates = direct_sim_metadata["s3://rubin:rubin-scheduler-prenight/opsim/2023-12-15/1/"][
            "simulated_dates"
        ]
        simulated_dates["first"] = simulated_dates["start"]
        del simulated_dates["start"]
        simulated_dates["last"] = simulated_dates["end"]
        del simulated_dates["end"]
    except KeyError:
        # If the archive doesn't have this old sim, don't worry about it.
        pass

    compiled_sim_metadata = read_sim_metadata_from_hdf(compilation_resource)

    # Test that everything in direct_sim_metadata has a corresponding matching
    # entry in the compilation.
    differences = []
    for sim_uri in direct_sim_metadata:
        for key in direct_sim_metadata[sim_uri]:
            try:
                if direct_sim_metadata[sim_uri][key] != compiled_sim_metadata[sim_uri][key]:
                    differences.append(
                        {
                            "sim_uri": sim_uri,
                            "key": key,
                            "direct_value": direct_sim_metadata[sim_uri][key],
                            "compiled_value": compiled_sim_metadata[sim_uri][key],
                        }
                    )
            except KeyError:
                differences.append(
                    {
                        "sim_uri": sim_uri,
                        "key": key,
                        "direct_value": direct_sim_metadata[sim_uri][key],
                        "compiled_value": "MISSING",
                    }
                )

    # Test that everything in the compilation has a corresponding matching
    # entry in direct_sim_metadata.
    for sim_uri in compiled_sim_metadata:
        for key in compiled_sim_metadata[sim_uri]:
            if sim_uri not in direct_sim_metadata:
                differences.append(
                    {
                        "sim_uri": sim_uri,
                        "direct_value": "MISSING",
                    }
                )
            elif key not in direct_sim_metadata[sim_uri]:
                differences.append(
                    {
                        "sim_uri": sim_uri,
                        "key": key,
                        "direct_value": "MISSING",
                        "compiled_value": compiled_sim_metadata[sim_uri][key],
                    }
                )

    return differences


def compile_sim_archive_metadata_cli(*args: Any) -> None:
    parser = argparse.ArgumentParser(description="Create a metadata compilation HDF5 file at a URI")
    parser.add_argument(
        "--compilation_uri",
        type=str,
        default=None,
        help="The URI of the metadata archive compilation to write. "
        + "Defaults to compilation_metadate.h5 in the archive.",
    )
    parser.add_argument(
        "--archive_base_uri",
        type=str,
        default="s3://rubin:rubin-scheduler-prenight/opsim/",
        help="Base URI for the archive",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Do not rebuild the whole compilation, "
        + "but add new simulations with dates after the last current entry.",
    )

    log_file = os.environ.get("SIM_ARCHIVE_LOG_FILE", None)
    if log_file is not None:
        logging.basicConfig(
            filename=log_file, format="%(asctime)s: %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z"
        )
    else:
        logging.basicConfig(level=logging.INFO)

    arg_values = parser.parse_args() if len(args) == 0 else parser.parse_args(args)
    archive_uri = arg_values.archive_base_uri
    compilation_uri = arg_values.compilation_uri
    append = arg_values.append
    if compilation_uri is None:
        compilation_resource = ResourcePath(archive_uri).join("compiled_metadata_cache.h5")
    else:
        compilation_resource = ResourcePath(compilation_uri)

    compilation_resource = compile_sim_metadata(archive_uri, compilation_resource, append=append)


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

    if first_day_obs is None:
        first_day_obs = Time(Time.now().mjd - 0.5, format="mjd").iso[:10]
    if last_day_obs is None:
        last_day_obs = first_day_obs

    sim_metadata = read_archived_sim_metadata(
        archive_uri, num_nights=max_simulation_age, compilation_resource=compilation_uri
    )
    LOGGER.debug(f"Total simulations it the last {max_simulation_age} days: {len(sim_metadata)}.")

    best_sim = None
    for uri, sim in sim_metadata.items():
        sim["uri"] = uri
        sim["exec_date"] = uri.split("/")[-3]
        sim["date_index"] = int(uri.split("/")[-2])

        if sim["simulated_dates"]["first"] > first_day_obs:
            continue
        if sim["simulated_dates"]["last"] < last_day_obs:
            continue
        if "telescope" in sim and sim["telescope"].lower() != telescope.lower():
            continue
        if not set(tags).issubset(sim["tags"]):
            continue
        if best_sim is not None:
            if sim["exec_date"] < best_sim["exec_date"]:
                continue
            if sim["date_index"] < best_sim["date_index"]:
                continue
        best_sim = sim

    if best_sim is not None:
        best_sim["opsim_rp"] = (
            ResourcePath(archive_uri)
            .join(best_sim["exec_date"], forceDirectory=True)
            .join(f"{best_sim['date_index']}", forceDirectory=True)
            .join(best_sim["files"]["observations"]["name"])
        )
        LOGGER.info(f"Most recent simulation meeting requested criteria is {best_sim['uri']}.")
    else:
        LOGGER.debug("No simulations met the requested criteria.")

    return best_sim


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

    if first_day_obs is None:
        first_day_obs = Time(Time.now().mjd - 0.5, format="mjd").iso[:10]
    if last_day_obs is None:
        last_day_obs = first_day_obs

    match which_sim:
        case ResourcePath():
            opsim_rp = which_sim
        case str():
            opsim_rp = ResourcePath(which_sim)
        case dict():
            opsim_rp = find_latest_prenight_sim_for_nights(first_day_obs, last_day_obs, **which_sim)[
                "opsim_rp"
            ]
        case None:
            opsim_rp = find_latest_prenight_sim_for_nights(first_day_obs, last_day_obs)["opsim_rp"]
        case _:
            raise NotImplementedError()

    assert isinstance(opsim_rp, ResourcePath)

    if get_sim_data_kwargs is None:
        get_sim_data_kwargs = {}
    assert isinstance(get_sim_data_kwargs, dict)

    # Limit visits returned to the nights we requested
    if "stackers" not in get_sim_data_kwargs:
        get_sim_data_kwargs["stackers"] = []
    dayobsiso_requested = maf.DayObsISOStacker in [s.__class__ for s in get_sim_data_kwargs["stackers"]]
    if not dayobsiso_requested:
        # We want it to filter out dates that were not requested,
        # so add it to the stacker even if it was not requested.
        get_sim_data_kwargs["stackers"].append(maf.DayObsISOStacker())

    visits = get_sim_data(opsim_rp, **get_sim_data_kwargs)

    LOGGER.debug(f"Loaded {len(visits)} from {opsim_rp}")
    visits_df = pd.DataFrame(visits)
    visits_df = visits_df.loc[
        (first_day_obs <= visits_df["day_obs_iso8601"]) & (visits_df["day_obs_iso8601"] <= last_day_obs),
        :,
    ]
    # If it dayobsiso was not requested, do not return it.
    if not dayobsiso_requested:
        visits_df.drop(columns="day_obs_iso8601", inplace=True)

    return visits_df


def fetch_obsloctap_visits(
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
    dbcols = [
        "observationStartMJD",
        "fieldRA",
        "fieldDec",
        "rotSkyPos",
        "band",
        "visitExposureTime",
        "night",
        "target_name",
    ]

    # Start with the first night that starts after the reference time,
    # which is the current time by default.
    # So, if the reference time is during a night, it starts with the
    # following night.
    night_bounds = pd.DataFrame(Almanac().sunsets)
    reference_time = Time.now() if day_obs is None else Time(day_obs, format="iso", scale="utc")
    first_night = night_bounds.query(f"sunset > {reference_time.mjd}").night.min()
    last_night = first_night + nights - 1

    night_bounds.set_index("night", inplace=True)
    start_mjd = night_bounds.loc[first_night, "sunset"]
    end_mjd = night_bounds.loc[last_night, "sunrise"]

    first_day_obs = Time(start_mjd - 0.5, format="mjd").iso[:10]
    last_day_obs = Time(end_mjd - 0.5, format="mjd").iso[:10]

    which_sim = {
        "tags": ("ideal", "nominal"),
        "telescope": telescope,
        "max_simulation_age": int(np.ceil(Time.now().mjd - reference_time.mjd)) + 1,
    }
    get_sim_data_kwargs = {
        "sqlconstraint": (f"observationStartMJD BETWEEN {start_mjd} AND {end_mjd}",),
        "dbcols": dbcols,
    }
    visits = fetch_sim_for_nights(
        first_day_obs, last_day_obs, which_sim=which_sim, get_sim_data_kwargs=get_sim_data_kwargs
    )

    return visits


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

    # Maybe we should move schedview.DayObs into rubin_sim so we can use it
    # here without introducing a schedview dependency.
    match day_obs:
        case str():
            pass
        case int():
            day_obs = datetime.datetime.strptime(str(day_obs), "%Y%m%d").date().isoformat()
        case None:
            day_obs = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
        case _:
            raise NotImplementedError(f"Cannot get day_obs from a {type(day_obs)}.")

    assert isinstance(day_obs, str)

    # Initialize the dictionary with the night stats
    night_sim_stats = {}

    # Count the visits in the latest simulated nominal night

    # Go far enough back to get the automatic pre-nights for the night
    max_simulation_age = int(np.ceil(Time.now().mjd - Time(day_obs).mjd)) + 3

    visits = fetch_sim_for_nights(
        first_day_obs=day_obs,
        last_day_obs=day_obs,
        which_sim={"max_simulation_age": max_simulation_age},
    )
    night_sim_stats["nominal_visits"] = len(visits) if isinstance(visits, pd.DataFrame) else 0

    # We can add whatever additional stats we like here. Possibilities include
    # statistics from multiple simulations, counts for different bands in the
    # nominal simulation, numbers of DDF/WFD/ToO visits, etc.

    return night_sim_stats


def export_sim_to_prototype_sim_archive(
    archive_metadata: vseqarchive.VisitSequenceArchiveMetadata, sim_uuid: UUID, proto_sim_archive_url: str
) -> ResourcePath:
    """Export a simulation to the prototype simulation archive.

    Parameters
    ----------
    archive_metadata : `vseqarchive.VisitSequenceArchiveMetadata`
        Interface to the visit‑sequence metadata database from
        which the simulation is to be imported.
    sim_uuid : `uuid.UUID`
        UUID of the simulation to be exported.
    proto_sim_archive_url : ``str``
        Base URL of the prototype simulation archive to which
        the simulations is to be exported.

    Returns
    -------
    sim_rp: `lsst.resources.ResourcePath`
        ResourcePath pointing to the root of the newly created
        prototype archive entry.
    """

    insert_date, new_id, proto_sim_rpath = _next_sim_date_and_index(proto_sim_archive_url)
    metadata = yaml.safe_load(archive_metadata.sim_metadata_yaml(sim_uuid))

    have_opsim = False
    for file_type in metadata["files"]:
        if file_type == "observations":
            have_opsim = True

        origin = ResourcePath(metadata["files"][file_type]["url"])
        destination = proto_sim_rpath.join(origin.basename())
        destination.transfer_from(origin, "copy")
        metadata["files"][file_type]["url"] = destination.geturl()

    if not have_opsim:
        visits_url = archive_metadata.get_visitseq_url(sim_uuid)
        visits_rp = ResourcePath(visits_url)
        with visits_rp.as_local() as visits_local_rp:
            with TemporaryDirectory() as temp_dir:
                opsimdb_fname = str(Path(temp_dir) / "opsim.db")
                vseqarchive.hdf5_to_opsimdb(visits_local_rp.ospath, opsimdb_fname)
                opsimdb_rp = proto_sim_rpath.join("opsim.db")
                opsimdb_rp.transfer_from(ResourcePath(opsimdb_fname), "copy")
        metadata["files"]["observations"] = {"name": "opsim.db", "url": opsimdb_rp.geturl()}

    metadata["files"]["metadata"] = {"name": "sim_metadata.yaml"}
    metadata["files"]["metadata"]["url"] = proto_sim_rpath.join(
        metadata["files"]["metadata"]["name"]
    ).geturl()

    metadata_yaml = yaml.dump(metadata)
    ResourcePath(metadata["files"]["metadata"]["url"]).write(metadata_yaml.encode("utf-8"))

    return proto_sim_rpath
