"""Tools for running the set of simulations used for a pre-night briefing."""

__all__ = ["AnomalousOverheadFunc", "run_prenight_sim_cli"]

import argparse
import logging
import pickle
import warnings
from datetime import date
from pathlib import Path

import numpy as np
from matplotlib.pylab import Generator
from rubin_scheduler.scheduler import sim_runner
from rubin_scheduler.site_models import Almanac

from rubin_sim.sim_archive import make_sim_data_dir

from .util import dayobs_to_date

LOGGER = logging.getLogger(__name__)


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


def compute_sim_start_and_end(day_obs: int, sim_nights: int, delay: float = 0) -> tuple[float, float]:
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
    djd0 = date(1899, 12, 31)

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
