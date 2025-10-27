"""Tools for running the set of simulations used for a pre-night briefing."""

__all__ = [
    "AnomalousOverheadFunc",
    "run_prenights",
    "prenight_sim_cli",
]

import argparse
import bz2
import gzip
import io
import logging
import lzma
import os
import pickle
import typing
import warnings
from datetime import datetime
from functools import partial
from pathlib import Path
from tempfile import TemporaryFile
from typing import Callable, Optional, Sequence
from warnings import warn

import numpy as np
import numpy.typing as npt
from astropy.time import Time
from matplotlib.pylab import Generator
from rubin_scheduler.scheduler import sim_runner
from rubin_scheduler.scheduler.example import example_scheduler
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.schedulers.core_scheduler import CoreScheduler
from rubin_scheduler.site_models import Almanac

from rubin_sim.sim_archive import make_sim_data_dir
from rubin_sim.sim_archive.sim_archive import drive_sim

from .make_snapshot import add_make_scheduler_snapshot_args, get_scheduler, save_scheduler
from .util import dayobs_to_date

try:
    from rubin_sim.data import get_baseline  # type: ignore
except ModuleNotFoundError:
    get_baseline = partial(warn, "Cannot find default baseline because rubin_sim is not installed.")

DEFAULT_ARCHIVE_URI = "s3://rubin:rubin-scheduler-prenight/opsim/"
LOGGER = logging.getLogger(__name__)


def _run_sim(
    sim_start_mjd: float,
    scheduler_io: io.BufferedRandom,
    label: str,
    tags: Sequence[str] = tuple(),
    sim_duration: float = 2,
    anomalous_overhead_func: Optional[Callable] = None,
    opsim_metadata: dict | None = None,
) -> None:
    assert False
    logging.info(f"Running {label}.")

    scheduler_io.seek(0)
    scheduler = pickle.load(scheduler_io)

    observatory = ModelObservatory(
        nside=scheduler.nside,
        cloud_data="ideal",
        seeing_data="ideal",
        downtimes="ideal",
    )

    drive_sim(
        observatory=observatory,
        scheduler=scheduler,
        label=label,
        tags=tags,
        script=__file__,
        sim_start_mjd=sim_start_mjd,
        sim_duration=sim_duration,
        record_rewards=True,
        anomalous_overhead_func=anomalous_overhead_func,
        opsim_metadata=opsim_metadata,
    )


def _mjd_now() -> float:
    assert False
    # Used instead of just Time.now().mjd to make type checker happy.
    mjd = Time.now().mjd
    assert isinstance(mjd, float)
    return mjd


def _iso8601_now() -> str:
    assert False
    # Used instead of just Time.now().mjd to make type checker happy.
    now_iso = Time.now().iso
    assert isinstance(now_iso, str)
    return now_iso


def _create_scheduler_io(
    day_obs_mjd: float,
    scheduler_fname: Optional[str] = None,
    scheduler_instance: Optional[CoreScheduler] = None,
) -> io.BufferedRandom:
    assert False

    if scheduler_instance is not None:
        scheduler = scheduler_instance
    elif scheduler_fname is None:
        sample_scheduler = example_scheduler()
        if not isinstance(sample_scheduler, CoreScheduler):
            raise TypeError()

        scheduler = sample_scheduler

    else:
        opener: typing.Callable = open

        if scheduler_fname.endswith(".bz2"):
            opener = bz2.open
        elif scheduler_fname.endswith(".xz"):
            opener = lzma.open
        elif scheduler_fname.endswith(".gz"):
            opener = gzip.open

        with opener(scheduler_fname, "rb") as sched_io:
            scheduler = pickle.load(sched_io)

    scheduler.keep_rewards = True

    scheduler_io = TemporaryFile()
    pickle.dump(scheduler, scheduler_io)
    scheduler_io.seek(0)
    return scheduler_io


class AnomalousOverheadFunc:
    """Callable to return random overhead.

    Parameters
    ----------
    seed : `int`
        Random number seed.
    slew_scale : `float`
        The scale for the scatter in the slew offest (seconds).
    visit_scale : `float`, optional
        The scale for the scatter in the visit overhead offset (seconds).
        Defaults to 0.0.
    slew_loc : `float`, optional
        The location of the scatter in the slew offest (seconds).
        Defaults to 0.0.
    visit_loc : `float`, optional
        The location of the scatter in the visit offset (seconds).
        Defaults to 0.0.
    """

    def __init__(
        self,
        seed: int,
        slew_scale: float,
        visit_scale: float = 0.0,
        slew_loc: float = 0.0,
        visit_loc: float = 0.0,
    ) -> None:
        self.rng: Generator = np.random.default_rng(seed)
        self.visit_loc: float = visit_loc
        self.visit_scale: float = visit_scale
        self.slew_loc: float = slew_loc
        self.slew_scale: float = slew_scale

    def __call__(self, visittime: float, slewtime: float) -> float:
        """Return a randomized offset for the visit overhead.

        Parameters
        ----------
        visittime : `float`
            The visit time (seconds).
        slewtime : `float`
            The slew time (seconds).

        Returns
        -------
        offset: `float`
            Random offset (seconds).
        """

        slew_overhead: float = slewtime * self.rng.normal(self.slew_loc, self.slew_scale)

        # Slew might be faster that expected, but it will never take negative
        # time.
        if (slewtime + slew_overhead) < 0:
            slew_overhead = 0.0

        visit_overhead: float = visittime * self.rng.normal(self.slew_loc, self.slew_scale)
        # There might be anomalous overhead that makes visits take longer,
        # but faster is unlikely.
        if visit_overhead < 0:
            visit_overhead = 0.0

        return slew_overhead + visit_overhead


def run_prenights(
    day_obs_mjd: float,
    scheduler_file: Optional[str] = None,
    minutes_delays: tuple[float, ...] = (0, 1, 10, 60, 240),
    anomalous_overhead_seeds: tuple[int, ...] = (101, 102, 103, 104, 105),
    sim_nights: int = 3,
    opsim_metadata: dict | None = None,
) -> None:
    assert False
    """Run the set of scheduler simulations needed to prepare for a night.

    Parameters
    ----------
    day_obs_mjd : `float`
        The starting MJD.
    scheduler_file : `str`, optional
        File from which to load the scheduler. None defaults to the example
        scheduler in rubin_sim, if it is installed.
        The default is None.
    minutes_delays : `tuple` of `float`
        Delayed starts to be simulated.
    anomalous_overhead_seeds: `tuple` of `int`
        Random number seeds to use for anomalous overhead runs.
    sim_nights: `int`
        Number of nights to simulate. Defaults to 3.
    opsim_metadata: `dict`
        Extra metadata for the archive
    """
    warnings.warn(
        "Use sims_sv_survey tools instead of rubin_sim to run prenight simulations!",
        DeprecationWarning,
        stacklevel=2,
    )
    exec_time: str = _iso8601_now()
    scheduler_io: io.BufferedRandom = _create_scheduler_io(day_obs_mjd, scheduler_fname=scheduler_file)

    # Assign args common to all sims for this execution.
    run_sim = partial(_run_sim, scheduler_io=scheduler_io, opsim_metadata=opsim_metadata)

    # Find the start of observing for the specified day_obs.
    # Almanac.get_sunset_info does not use day_obs, so just index
    # Almanac.sunsets for what we want directly.
    all_sun_n12_setting: npt.NDArray[np.float_] = Almanac().sunsets["sun_n12_setting"]
    before_first_day_obs: npt.NDArray[np.bool_] = all_sun_n12_setting < day_obs_mjd + 0.5
    after_first_day_obs: npt.NDArray[np.bool_] = all_sun_n12_setting > day_obs_mjd + 1.5
    on_first_day_obs: npt.NDArray[np.bool_] = ~(before_first_day_obs | after_first_day_obs)
    sim_start_mjd: float = all_sun_n12_setting[on_first_day_obs].item()

    all_sun_n12_rising: npt.NDArray[np.float_] = Almanac().sunsets["sun_n12_rising"]
    before_last_day_obs: npt.NDArray[np.bool_] = all_sun_n12_setting < day_obs_mjd + sim_nights + 0.5
    after_last_day_obs: npt.NDArray[np.bool_] = all_sun_n12_setting > day_obs_mjd + sim_nights + 1.5
    on_last_day_obs: npt.NDArray[np.bool_] = ~(before_last_day_obs | after_last_day_obs)
    sim_end_mjd: float = all_sun_n12_rising[on_last_day_obs].item()
    sim_duration: float = sim_end_mjd - sim_start_mjd

    # Begin with an ideal pure model sim.
    completed_run_without_delay: bool = False
    if len(minutes_delays) == 0 or (np.array(minutes_delays) == 0).any():
        run_sim(
            sim_start_mjd,
            label=f"Nominal start and overhead, ideal conditions, run at {exec_time}",
            tags=["ideal", "nominal"],
            sim_duration=sim_duration,
        )
        completed_run_without_delay = True

    # Delayed start
    for minutes_delay in minutes_delays:
        if completed_run_without_delay and minutes_delay == 0:
            # Did this already.
            continue

        delayed_sim_start_mjd = sim_start_mjd + minutes_delay / (24.0 * 60)
        sim_duration = sim_end_mjd - delayed_sim_start_mjd

        run_sim(
            delayed_sim_start_mjd,
            label=f"Start time delayed by {minutes_delay} minutes,"
            + f" Nominal slew and visit overhead, ideal conditions, run at {exec_time}",
            tags=["ideal", f"delay_{minutes_delay}"],
            sim_duration=sim_duration,
        )

    # Run a few different scatters of visit time
    anomalous_overhead_scale = 0.1
    for anomalous_overhead_seed in anomalous_overhead_seeds:
        anomalous_overhead_func = AnomalousOverheadFunc(anomalous_overhead_seed, anomalous_overhead_scale)
        run_sim(
            sim_start_mjd,
            label=f"Anomalous overhead {anomalous_overhead_seed, anomalous_overhead_scale},"
            + f" Nominal start, ideal conditions, run at {exec_time}",
            tags=["ideal", "anomalous_overhead"],
            anomalous_overhead_func=anomalous_overhead_func,
            sim_duration=sim_duration,
        )


def _parse_dayobs_to_mjd(dayobs: str | float) -> float:
    assert False
    try:
        day_obs_mjd = Time(dayobs).mjd
    except ValueError:
        try:
            day_obs_mjd = Time(datetime.strptime(str(dayobs), "%Y%m%d"))
        except ValueError:
            day_obs_mjd = Time(dayobs, format="mjd")

    if not isinstance(day_obs_mjd, float):
        raise ValueError

    return day_obs_mjd


def prenight_sim_cli(cli_args: list = []) -> None:
    assert False
    parser = argparse.ArgumentParser(description="Run prenight simulations")
    default_time = Time(int(_mjd_now() - 0.5), format="mjd")
    parser.add_argument(
        "--dayobs",
        type=str,
        default=default_time.iso,
        help="The day_obs of the night to simulate.",
    )
    parser.add_argument("--scheduler", type=str, default=None, help="pickle file of the scheduler to run.")
    parser.add_argument("--config_version", type=str, default=None, help="Version of ts_config_ocs used.")
    parser.add_argument("--telescope", type=str, default=None, help="Telescope scheduled.")

    # Configure logging
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    log_handlers = [stream_handler]

    log_file = os.environ.get("PRENIGHT_LOG_FILE", None)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.DEBUG)
        log_handlers.append(file_handler)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        handlers=[stream_handler, file_handler],
    )

    # FIXME
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="IntRounded being used with a potentially too-small scale factor.",
    )

    add_make_scheduler_snapshot_args(parser)

    args = parser.parse_args() if len(cli_args) == 0 else parser.parse_args(cli_args)

    day_obs_mjd = _parse_dayobs_to_mjd(args.dayobs)

    scheduler_file = args.scheduler
    opsim_metadata = {"telescope": args.telescope}
    if args.repo is not None:
        if os.path.exists(scheduler_file):
            raise ValueError(f"File {scheduler_file} already exists!")

        if args.config_version is not None:
            scheduler: CoreScheduler = get_scheduler(config_script=args.script, visits_db=args.visits)
            save_scheduler(scheduler, scheduler_file)
            opsim_metadata.update(
                {
                    "opsim_config_repository": args.repo,
                    "opsim_config_script": args.script,
                    "opsim_config_version": args.config_version,
                }
            )
        elif args.branch is not None:
            scheduler: CoreScheduler = get_scheduler(
                args.repo, args.script, args.branch, visits_db=args.visits
            )
            save_scheduler(scheduler, scheduler_file)

            opsim_metadata.update(
                {
                    "opsim_config_repository": args.repo,
                    "opsim_config_script": args.script,
                    "opsim_config_branch": args.branch,
                }
            )
        else:
            raise ValueError("Either the branch or the version of ts_ocs_config must be specified")

    run_prenights(
        day_obs_mjd,
        scheduler_file,
        minutes_delays=(0, 1, 10, 60),
        anomalous_overhead_seeds=(101, 102),
        opsim_metadata=opsim_metadata,
    )


def compute_sim_start_and_end(day_obs: int, sim_nights: int, delay: float = 0) -> tuple(float, float):
    """Compute simulation start and end time for a prenight simulation.

    Parameters
    ----------
    day_obs: `int`
        The starting dayobs as YYYYMMDD.
    sim_nights: `int`
        The number of nights to simulate
    delay: `float`
        A delayed start, in minutes.

    Return
    ------
    sim_start_mjd, sim_duration : `tuple`
        The starting MJD and simulation duration in days.
    """
    # Find the start of observing for the specified day_obs.
    # Almanac.get_sunset_info does not use day_obs, so
    # translate to DJD, Dublin Julian Date.
    # Like plain Julian Date, DJD rolls over at the
    # same time as day_obs, but unlike JD, the year
    # of the epoch works with python's datetime.date.
    # Note that MJD - DJD = 15019.5
    # and has epoch 1899-12-31T12:00:00.
    sunsets = Almanac().sunsets
    djd_mjd_offset = 15019.5
    djd0 = datetime.date(1899, 12, 31)

    night0_djd = int(np.floor(sunsets["sunset"][0] - djd_mjd_offset))
    day_obs_date = dayobs_to_date(day_obs)
    day_obs_djd = (day_obs_date - djd0).days
    day_obs_start_night = day_obs_djd - night0_djd
    sim_start_mjd = sunsets["sun_n12_setting"][day_obs_start_night] + delay / (24 * 60.0)
    day_obs_end_night = day_obs_start_night + sim_nights - 1
    sim_end_mjd = sunsets["sunrise"][day_obs_end_night]
    sim_duration = sim_end_mjd - sim_start_mjd
    assert np.ceil(sim_duration) == sim_nights
    return sim_start_mjd, sim_duration


def run_prenight_sim_cli(cli_args: list = []) -> int:
    parser = argparse.ArgumentParser(description="Run an SV simulation.")
    parser.add_argument("scheduler", type=str, help="scheduler pickle file.")
    parser.add_argument("observatory", type=str, help="model observatory pickle file.")
    parser.add_argument("day_obs", type=int, help="start day obs.")
    parser.add_argument("sim_nights", type=int, help="number of nights to run.")
    parser.add_argument("--keep_rewards", action="store_true", help="Compute rewards data.")
    parser.add_argument("--telescope", type=str, default="simonyi", help="The telescope simulated.")
    parser.add_argument("--label", type=str, default="", help="The tags on the simulation.")
    parser.add_argument("--delay", type=float, default=0.0, help="Minutes after nominal to start.")
    parser.add_argument("--anom_overhead_scale", type=float, default=0.0, help="scale of scatter in the slew")
    parser.add_argument(
        "--anom_overhead_seed",
        type=int,
        default=1,
        help="random number seed for anomalous scatter in overhead",
    )
    parser.add_argument("--tags", type=str, default=[], nargs="*", help="The tags on the simulation.")
    parser.add_argument("--results", type=str, default="", help="Results directory.")
    args = parser.parse_args() if len(cli_args) == 0 else parser.parse_args(cli_args)
    day_obs = args.day_obs
    sim_nights = args.sim_nights
    keep_rewards = args.keep_rewards
    tags = args.tags
    label = args.label
    telescope = args.telescope
    delay = args.delay
    anom_overhead_scale = args.anom_overhead_scale
    anom_overhead_seed = args.anom_overhead_seed
    results_dir = args.results if len(args.results) > 0 else None

    with open(args.scheduler, "rb") as scheduler_io:
        scheduler = pickle.load(scheduler_io)

    with open(args.observatory, "rb") as observatory_io:
        observatory = pickle.load(observatory_io)

    if anom_overhead_scale > 0:
        anomalous_overhead_func = AnomalousOverheadFunc(anom_overhead_seed, anom_overhead_scale)
    else:
        anomalous_overhead_func = None

    if keep_rewards:
        scheduler.keep_rewards = keep_rewards

    sim_start_mjd, sim_duration = compute_sim_start_and_end(day_obs, sim_nights, delay)
    observatory.mjd = sim_start_mjd

    LOGGER.info("Starting simulation")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        results = sim_runner(
            observatory,
            scheduler,
            sim_start_mjd=sim_start_mjd,
            sim_duration=sim_duration,
            record_rewards=keep_rewards,
            verbose=True,
            anomalous_overhead_func=anomalous_overhead_func,
        )
    observatory, scheduler, obs = results[:3]
    rewards, obs_rewards = (None, None) if len(results) < 5 else results[3:5]
    LOGGER.info("Simulation complete.")

    LOGGER.info("Writing simulation results")
    data_path = make_sim_data_dir(
        obs,
        rewards,
        obs_rewards,
        in_files={"scheduler": args.scheduler, "observatory": args.observatory},
        tags=tags,
        label=label,
        opsim_metadata={"telescope": telescope},
        data_path=results_dir,
    )
    output_dirname: str = "Unknown"
    if isinstance(data_path, Path):
        assert isinstance(data_path, Path)
        output_dirname = data_path.name
    else:
        assert isinstance(data_path, str)
        output_dirname = data_path

    LOGGER.info(f"Wrote results in directory: {output_dirname}")

    return 0


if __name__ == "__main__":
    prenight_sim_cli()
