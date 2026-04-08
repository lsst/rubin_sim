"""Tools for running the set of simulations used for a pre-night briefing."""

__all__ = ["AnomalousOverheadFunc", "run_prenight_sim_cli"]

import argparse
import logging
import pickle
import warnings
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np
import scipy.stats
from matplotlib.pylab import Generator
from rubin_scheduler.scheduler import sim_runner
from rubin_scheduler.site_models import Almanac

from rubin_sim.sim_archive import make_sim_data_dir

from .util import dayobs_to_date

LOGGER = logging.getLogger(__name__)


# Type hint to match scipy.stats style distribution classes
class ContinuousDistribution(Protocol):
    def rvs(
        self,
        size: int | tuple[int, ...] | None = None,
        random_state: np.random.RandomState | np.random.Generator | int | None = None,
    ) -> float: ...


class AnomalousOverheadFunc:
    """Callable to return random overhead.

    Parameters
    ----------
    seed : `int`
        Random number seed.
    slew_scale : `float`
        The scale for the scatter in the slew offest (seconds).
        Defaults to 0.0.
    visit_scale : `float`, optional
        The scale for the scatter in the visit overhead offset (seconds).
        Defaults to 0.0.
    slew_loc : `float`, optional
        The location of the scatter in the slew offest (seconds).
        It is unlikely that this should ever be non-zero.
        Defaults to 0.0.
    visit_loc : `float`, optional
        The location of the scatter in the visit offset (seconds).
        It is unlikely that this should ever be non-zero.
        Defaults to 0.0.
    scatter_distribution: `str` or `None`, optional
        The distribution from which the scatter should be taken.
        This must be the name of a method of `numpy.random.Generator`.
        If `None`, the distribution is taken from
        `DEFAULT_OVERHEAD_SCATTER_DIST`.
        Defaults to ``None``.
    scatter_kwargs: `dict` or `None`, optional
        Dictionary of arguments passed to ``scatter_distribution``.
        If `None`, the argumentns taken from
        `DEFAULT_OVERHEAD_SCATTER_KWARGS`.
    min_overhead : `float`, optional
        The minimum possible overhead

    Notes
    -----
    - The ``slew_scale`` and ``slew_loc`` parameters introduce an offset
      proportional to the slew time following a normal distribution.
      A non-zero ``slew_loc`` will sysetmatically move the center of
      the distribution.
    - The ``visit_scale`` and ``visit_loc`` parameters introduce an offset
      proportional to the visit time following a normal distribution.
      A non-zero ``visit_loc`` will sysetmatically move the center of
      the distribution.
    - The ``scatter_distribution`` and ``scatter_kwargs`` introduce an
      offset independent of the modeled slew and visit times, and support
      any distribution offered by `numpy.random.Generator`.

    """

    default_overhead_scatter_dist: ContinuousDistribution = scipy.stats.halfnorm
    default_overhead_scatter_kwargs = {"scale": 0.0, "loc": 0.0}

    def __init__(
        self,
        seed: int,
        slew_scale: float = 0.0,
        visit_scale: float = 0.0,
        slew_loc: float = 0.0,
        visit_loc: float = 0.0,
        min_overhead: float = 0.0,
        scatter_distribution: str | ContinuousDistribution | None = None,
        scatter_kwargs: dict | None = None,
    ) -> None:
        self.rng: Generator = np.random.default_rng(seed)
        self.visit_loc: float = visit_loc
        self.visit_scale: float = visit_scale
        self.slew_loc: float = slew_loc
        self.slew_scale: float = slew_scale

        if scatter_kwargs is None:
            scatter_kwargs = self.default_overhead_scatter_kwargs
        assert isinstance(scatter_kwargs, dict)
        self.scatter_kwargs = deepcopy(scatter_kwargs)

        self.scatter_dist_func: Callable
        if scatter_distribution is None:
            scatter_distribution = self.default_overhead_scatter_dist

        if isinstance(scatter_distribution, str):
            # If scatter_distribution is a string, interpret it
            # as a numpy random number distribution.
            try:
                maybe_scatter_dist_func = getattr(self.rng, scatter_distribution)
                # If scatter_distribution is a plain attribute
                # rather than a true
                # method, it won't work for us, and confuses the type checker.
            except AttributeError as exc:
                raise AttributeError(
                    "'scatter_distribution' must be valid NumPy random Generator method, "
                    f"and '{scatter_distribution}' is not."
                ) from exc
            assert callable(maybe_scatter_dist_func), "'scatter_distribution' must be callable"
            self.scatter_dist_func = maybe_scatter_dist_func
        else:
            maybe_scatter_dist_func = getattr(scatter_distribution, "rvs", None)
            assert callable(
                maybe_scatter_dist_func
            ), f"Cannot use scatter distribution of type {str(scatter_distribution.__class__.__name__)}"

            if "random_state" not in self.scatter_kwargs:
                self.scatter_kwargs["random_state"] = self.rng

            self.scatter_dist_func = maybe_scatter_dist_func

        self.min_overhead: float = min_overhead

        if self.visit_scale > 0:
            raise NotImplementedError("Visit scale scatter is no longer implemented.")

    def __call__(
        self,
        visittime: float | np.ndarray | None = None,
        slewtime: float | np.ndarray | None = None,
        obs: Mapping[str, Any] | None = None,
    ) -> float:
        """Return a randomized offset for the visit overhead.

        Parameters
        ----------
        visittime : `float` or `None`
            The visit time (seconds). No longer does anything,
            and will be removed.
        slewtime : `float` or `None`
            The slew time (seconds). If the ``slew_scale`` attribute
            is greater than zero, either ``slewtime`` must not be
            ``None`` or ``obs`` (below) must have a ``slewtime`` key.
            The use of ``slewtime`` is deprecated in favor of using
            ``obs``.
        obs : `Mapping[str, Any]`
            Observation record.

        Returns
        -------
        offset: `float`
            The offset (in seconds) between the modeled overhead
            between exposures and the overhead to be applied.
            "Overhead" is the difference in start times between
            succesive expusures, minus the exposure times:
            ``overhead = exp_start_2 - exp_start_1 - exptime_1``.
            So, the ``offset`` here is time to be added to
            the modeled start of an exposure.
        """
        if visittime is not None:
            warnings.warn(
                "The visittime argument is deprecated and does nothing.", DeprecationWarning, stacklevel=2
            )

        slew_overhead_offset: float
        if self.slew_scale > 0 or self.slew_loc > 0:
            # we are actually using slew time scattering, so
            # get the model slewtime.
            if slewtime is not None:
                if isinstance(slewtime, np.ndarray):
                    assert slewtime.shape == (1,)
                    slewtime = float(slewtime.item())
                assert isinstance(slewtime, float)
                warnings.warn(
                    "The visittime argument is deprecated and unused", DeprecationWarning, stacklevel=2
                )
            else:
                assert isinstance(obs, Mapping)
                assert isinstance(obs["slewtime"], float)
                slewtime = obs["slewtime"]

            slew_overhead_offset = slewtime * self.rng.normal(self.slew_loc, self.slew_scale)
            # Slew might be faster than expected,
            # but it will never take negative time.
            if (slewtime + slew_overhead_offset) < 0:
                slew_overhead_offset = -1 * slewtime
        else:
            slew_overhead_offset = 0.0

        scatter_offset: float = self.scatter_dist_func(**self.scatter_kwargs)

        overhead_offset: float = slew_overhead_offset + scatter_offset

        # If we have the available parameters, make sure the final overhead
        # is more than min_overhead
        if obs is not None and "visittime" in obs and "exptime" in obs and "slewtime" in obs:
            model_overhead = obs["slewtime"] + obs["visittime"] - obs["exptime"]
            after_offset_overhead = model_overhead + overhead_offset
            if after_offset_overhead < self.min_overhead:
                overhead_offset = self.min_overhead - model_overhead
        else:
            warnings.warn("Could not verify that the overhead was greater than the minimum.")

        return overhead_offset


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
        "--anom_overhead_scatter", type=float, default=0.0, help="absolute scatter in the overhead"
    )
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
    anom_overhead_scatter = args.anom_overhead_scatter
    anom_overhead_seed = args.anom_overhead_seed
    results_dir = args.results if len(args.results) > 0 else None

    with open(args.scheduler, "rb") as scheduler_io:
        scheduler = pickle.load(scheduler_io)

    with open(args.observatory, "rb") as observatory_io:
        observatory = pickle.load(observatory_io)

    anomalous_overhead_func: Callable | None = None
    anom_overhead_args = {}
    if anom_overhead_scale != 0.0:
        anom_overhead_args["slew_scale"] = anom_overhead_scale
    if anom_overhead_scatter != 0.0:
        anom_overhead_args["scatter_kwargs"] = {"scale": anom_overhead_scatter}
    if len(anom_overhead_args) > 0:
        anom_overhead_args["seed"] = anom_overhead_seed
        anomalous_overhead_func = AnomalousOverheadFunc(**anom_overhead_args)

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
