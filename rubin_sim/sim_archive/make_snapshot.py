__all__ = [
    "get_scheduler",
    "save_scheduler",
    "add_make_scheduler_snapshot_args",
    "make_scheduler_snapshot_cli",
    "get_scheduler_instance_from_repo",
]

import argparse
import bz2
import gzip
import importlib.util
import lzma
import pickle
import sys
import types
import typing
from pathlib import Path
from tempfile import TemporaryDirectory

from git import Repo
from rubin_scheduler.scheduler.example import example_scheduler
from rubin_scheduler.scheduler.schedulers.core_scheduler import CoreScheduler


def get_scheduler_instance_from_path(config_script_path: str | Path) -> CoreScheduler:
    """Generate a CoreScheduler according to a configuration in a file.

    Parameters
    ----------
    config_script_path : `str`
        The configuration script path (relative to the repository root).

    Returns
    -------
    scheduler : `CoreScheduler`
        An instance of the Rubin Observatory FBS.

    Raises
    ------
    ValueError
        If the config file is invalid, or has invalid content.
    """

    config_module_name: str = "scheduler_config"
    config_module_spec = importlib.util.spec_from_file_location(config_module_name, config_script_path)
    if config_module_spec is None or config_module_spec.loader is None:
        # Make type checking happy
        raise ValueError(f"Cannot load config file {config_script_path}")

    config_module: types.ModuleType = importlib.util.module_from_spec(config_module_spec)
    sys.modules[config_module_name] = config_module
    config_module_spec.loader.exec_module(config_module)

    scheduler: CoreScheduler = config_module.get_scheduler()[1]
    return scheduler


def get_scheduler_instance_from_repo(
    config_repo: str,
    config_script: str,
    config_branch: str = "main",
) -> CoreScheduler:
    """Generate a CoreScheduler according to a configuration in git.

    Parameters
    ----------
    config_repo : `str`
        The git repository with the configuration.
    config_script : `str`
        The configuration script path (relative to the repository root).
    config_branch : `str`, optional
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
        repo: Repo = Repo.clone_from(config_repo, local_config_repo_parent, branch=config_branch)
        full_config_script_path: Path = Path(repo.working_dir).joinpath(config_script)
        scheduler = get_scheduler_instance_from_path(full_config_script_path)

    return scheduler


def get_scheduler(
    config_repo: str | None,
    config_script: str | None,
    config_branch: str = "main",
) -> CoreScheduler:
    """Generate a CoreScheduler according to a configuration in git.

    Parameters
    ----------
    config_repo : `str`
        The git repository with the configuration.
    config_script : `str`
        The configuration script path (relative to the repository root).
    config_branch : `str`, optional
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
    if config_repo is not None and len(config_repo) > 0:
        if config_script is None:
            raise ValueError("If the config repo is set, the script must be as well.")
        scheduler = get_scheduler_instance_from_repo(
            config_repo=config_repo, config_script=config_script, config_branch=config_branch
        )
    elif config_script is not None:
        scheduler = get_scheduler_instance_from_path(config_script)
    else:
        example_scheduler_result = example_scheduler()
        if isinstance(example_scheduler_result, CoreScheduler):
            scheduler = example_scheduler_result
        else:
            # It might return a observatory, scheduler, observations tuple
            # instead.
            scheduler = example_scheduler_result[1]

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
        "--branch", type=str, default="main", help="The branch of the repo from which to get the script"
    )


def make_scheduler_snapshot_cli(cli_args: list = []) -> None:
    parser = argparse.ArgumentParser(description="Create a scheduler pickle")
    add_make_scheduler_snapshot_args(parser)
    args: argparse.Namespace = parser.parse_args() if len(cli_args) == 0 else parser.parse_args(cli_args)

    scheduler: CoreScheduler = get_scheduler(args.repo, args.config, args.branch)
    save_scheduler(scheduler, args.scheduler_fname)


if __name__ == "__main__":
    make_scheduler_snapshot_cli()
