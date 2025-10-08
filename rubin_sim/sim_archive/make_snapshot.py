__all__ = [
    "get_scheduler",
    "save_scheduler",
    "add_make_scheduler_snapshot_args",
    "make_scheduler_snapshot_cli",
    "get_scheduler_from_config",
    "get_scheduler_instance_from_repo",
]

import argparse
import bz2
import gzip
import importlib.util
import logging
import lzma
import pickle
import sys
import types
import typing
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from git import Repo
from rubin_scheduler.scheduler.example import example_scheduler
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.schedulers.core_scheduler import CoreScheduler
from rubin_scheduler.scheduler.utils import SchemaConverter


def get_scheduler_from_config(config_script_path: str | Path) -> typing.Tuple[int, CoreScheduler]:
    """Generate a CoreScheduler according to a configuration in a file.

    Parameters
    ----------
    config_script_path : `str`
        The configuration script path

    Returns
    -------
    nside : `int`
        The nside.
    scheduler : `CoreScheduler`
        An instance of the Rubin Observatory FBS.

    Raises
    ------
    ValueError
        If the config file is invalid, or has invalid content.
    """

    # Follow example from official python docs:
    # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    # The T&S supplied files have code in a __name__ == 'config' conditional
    # that we need to be executed, so we *must* name the module "config"
    config_module_name: str = "config"
    config_module_spec = importlib.util.spec_from_file_location(config_module_name, config_script_path)
    if config_module_spec is None or config_module_spec.loader is None:
        # Make type checking happy
        raise ValueError(f"Cannot load config file {config_script_path}")

    config_module: types.ModuleType = importlib.util.module_from_spec(config_module_spec)
    sys.modules[config_module_name] = config_module
    config_module_spec.loader.exec_module(config_module)

    try:
        scheduler: CoreScheduler = config_module.scheduler
        nside: int = scheduler.nside
    except NameError:
        nside, scheduler = config_module.get_scheduler()
        assert isinstance(nside, int)
        assert isinstance(scheduler, CoreScheduler)

    return nside, scheduler


def get_scheduler_instance_from_repo(
    config_repo: str,
    config_script: str,
    config_ref: str = "main",
) -> CoreScheduler:
    """Generate a CoreScheduler according to a configuration in git.

    Parameters
    ----------
    config_repo : `str`
        The git repository with the configuration.
    config_script : `str`
        The configuration script path (relative to the repository root).
    config_ref : `str`, optional
        The branch of the repository to use, by default "main"

    Returns
    -------
    scheduler : `CoreScheduler`
        An instance of the Rubin Observatory FBS.

    Raises
    ------
    ValueError
        If the config file is invalid, or has invalid content.
    """

    with TemporaryDirectory() as local_config_repo_parent:
        repo: Repo = Repo.clone_from(config_repo, local_config_repo_parent)
        repo.git.checkout(config_ref)
        full_config_script_path: Path = Path(repo.working_dir).joinpath(config_script)
        scheduler = get_scheduler_from_config(full_config_script_path)[1]

    return scheduler


def get_scheduler(
    config_repo: str | None = None,
    config_script: str | None = None,
    config_ref: str = "main",
    visits_db: str | None = None,
) -> CoreScheduler:
    """Generate a CoreScheduler according to a configuration.

    Parameters
    ----------
    config_repo : `str`
        The git repository with the configuration.
    config_script : `str`
        The configuration script path (relative to the repository root).
    config_ref : `str`, optional
        The branch of the repository to use, by default "main"
    visits_db : `str` or `None`
        Database from which to load pre-existing visits

    Returns
    -------
    scheduler : `CoreScheduler`
        An instance of the Rubin Observatory FBS.

    Raises
    ------
    ValueError
        If the config file is invalid, or has invalid content.
    """
    if config_repo is not None and len(config_repo) > 0:
        if config_script is None:
            raise ValueError("If the config repo is set, the script must be as well.")
        logging.info(
            f"Instantiating scheduler form {config_script} on the {config_ref} branch of {config_repo}"
        )
        scheduler = get_scheduler_instance_from_repo(
            config_repo=config_repo, config_script=config_script, config_ref=config_ref
        )
    elif config_script is not None:
        logging.info(f"Reading scheduler from {config_script}")
        scheduler = get_scheduler_from_config(config_script)[1]
    else:
        logging.info("Creating example scheduler")
        example_scheduler_result = example_scheduler()
        if isinstance(example_scheduler_result, CoreScheduler):
            scheduler = example_scheduler_result
        else:
            # It might return a observatory, scheduler, observations tuple
            # instead.
            scheduler = example_scheduler_result[1]

    if isinstance(visits_db, str) and len(visits_db) > 0:
        logging.info(f"Adding visits from {visits_db}.")
        obs: np.recarray = SchemaConverter().opsim2obs(visits_db)
        if len(obs) > 0:
            scheduler.add_observations_array(obs)
        logging.info("Finished adding visits")

    return scheduler


def save_scheduler(scheduler: CoreScheduler, file_name: str) -> None:
    """Save an instances of the scheduler in a pickle file,
    compressed according to its extension.

    Parameters
    ----------
    scheduler : `CoreScheduler`
        The scheduler to save.
    file_name : `str`
        The file in which to save the schedulers.
    """
    opener: typing.Callable = open

    if file_name.endswith(".bz2"):
        opener = bz2.open
    elif file_name.endswith(".xz"):
        opener = lzma.open
    elif file_name.endswith(".gz"):
        opener = gzip.open

    with opener(file_name, "wb") as pio:
        pickle.dump(scheduler, pio)


def add_make_scheduler_snapshot_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments needed for saving a scheduler to an argument parser."""
    parser.add_argument("--scheduler_fname", type=str, help="The file in which to save the scheduler.")
    parser.add_argument(
        "--repo", type=str, default=None, help="The repository from which to load the configuration."
    )
    parser.add_argument(
        "--script", type=str, default=None, help="The path to the config script (relative to the repo root)."
    )
    parser.add_argument(
        "--ref",
        type=str,
        default="main",
        help="The git reference (tag/commit/branch) of the repo from which to get the script",
    )
    parser.add_argument(
        "--visits", type=str, default="", help="Opsim database from which to load previous visits."
    )


def make_scheduler_snapshot_cli(cli_args: list = []) -> None:
    parser = argparse.ArgumentParser(description="Create a scheduler pickle")
    add_make_scheduler_snapshot_args(parser)
    args: argparse.Namespace = parser.parse_args() if len(cli_args) == 0 else parser.parse_args(cli_args)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s:%(name)s:%(levelname)s:%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    scheduler: CoreScheduler = get_scheduler(args.repo, args.script, args.ref, args.visits)

    save_scheduler(scheduler, args.scheduler_fname)


def make_ideal_model_observatory_cli(cli_args: list = []) -> None:
    parser = argparse.ArgumentParser(description="Create an ideal model observatory pickle")
    parser.add_argument("scheduler", type=str, help="Scheduler pickle file to read.")
    parser.add_argument("model_observatory", type=str, help="Model observatory pickle file to write.")
    args: argparse.Namespace = parser.parse_args() if len(cli_args) == 0 else parser.parse_args(cli_args)

    with open(args.scheduler, "rb") as scheduler_io:
        scheduler = pickle.load(scheduler_io)

    observatory = ModelObservatory(
        nside=scheduler.nside,
        cloud_data="ideal",
        seeing_data="ideal",
        downtimes="ideal",
    )

    with open(args.model_observatory, "wb") as observatory_io:
        pickle.dump(observatory, observatory_io)


if __name__ == "__main__":
    make_scheduler_snapshot_cli()
