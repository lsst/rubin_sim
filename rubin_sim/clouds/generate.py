"""Generate monthly representative cloud sequences from CTIO records.

The generator builds a synthetic cloud record in which every calendar month
is "typical": its distribution of cloud states and its transition counts
match the historical statistics for that month, to within the specified
monthly tolerances.
"""

import calendar
from collections.abc import Sequence
from datetime import date
from os import PathLike
from typing import Any

import click
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from .data import (
    CloudDistribution,
    Clouds,
    TransitionMatrix,
    _validate_cloud_distribution,
    _validate_transition_matrix,
    compute_transition_matrix,
    count_cloud_states,
    make_cloud_frame,
    plot_cloud_histogram,
    plot_transition_histograms,
    read_historical_clouds,
    save_clouds,
)

__all__ = ("generate_average_clouds", "cloud_generation_workflow", "generate_average_clouds_cli")

# Origin states to try, in priority order, when sampling the next state from
# a given current state (the row index). Each row starts with the state
# itself, then falls back to the nearest states. This follows the historical
# fallback priority, except that here the state itself is tried first: it is
# only skipped when the budget of transitions out of it is exhausted.
_ORIGINS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8),
    (1, 2, 0, 3, 4, 5, 6, 7, 8),
    (2, 1, 0, 3, 4, 5, 6, 7, 8),
    (3, 4, 5, 6, 7, 8, 2, 1, 0),
    (4, 3, 5, 6, 7, 8, 2, 1, 0),
    (5, 6, 7, 4, 8, 3, 2, 1, 0),
    (6, 5, 7, 4, 8, 3, 2, 1, 0),
    (7, 6, 5, 8, 4, 3, 2, 1, 0),
    (8, 7, 6, 5, 4, 3, 2, 1, 0),
)


def _generate_month(
    transitions: npt.NDArray[np.float64],
    distribution: npt.NDArray[np.float64],
    num_quarters: int,
    prior_state: int | None,
    rng: np.random.Generator,
    max_steps: int,
) -> npt.NDArray[np.uint8]:
    """Sample one month of cloud states and verify its monthly statistics.

    The sequence is drawn by walking a transition budget: each draw of a
    (origin, destination) pair consumes one unit of that pair's expected
    count, and each destination state is capped by its expected number of
    quarters. This keeps the result close to the historical statistics
    without having to rejection-sample entire months.

    Parameters
    ----------
    transitions : `numpy.ndarray`, (9, 9)
        Historical transition counts for this month, indexed by origin
        then destination.
    distribution : `numpy.ndarray`, (9,)
        Historical counts of quarters in each cloud state for this month.
    num_quarters : `int`
        Number of quarters in the month to generate.
    prior_state : `int` or `None`
        Cloud state of the quarter immediately preceding the month, or
        `None` if there is none (the start of the simulation).
    rng : `numpy.random.Generator`
        Random number generator. It is shared across attempts so that
        retries explore new sequences rather than repeating one.
    max_steps : `int`
        Maximum number of attempts to choose a state for any one quarter.

    Returns
    -------
    sequence : `numpy.ndarray`, (num_quarters,)
        Generated cloud state, in eighths, for each quarter.

    Raises
    ------
    RuntimeError
        If no state can be chosen for a quarter within ``max_steps``, or if
        the completed sequence fails the monthly distribution or transition
        count checks.
    """
    # Integer state counts must sum to num_quarters, so floor everything and
    # then hand the leftover quarters to states chosen with probability
    # proportional to their fractional remainder.
    expected_states = num_quarters * distribution / distribution.sum()
    remaining_states = np.floor(expected_states).astype(int)
    fractions = expected_states - remaining_states
    num_ceil = num_quarters - remaining_states.sum()
    if num_ceil:
        choices = rng.choice(9, size=num_ceil, replace=False, p=fractions / fractions.sum())
        remaining_states[choices] += 1

    # With no prior state, the first quarter has no incoming transition (the
    # seed state of 0 is a placeholder), so one fewer transition occurs.
    num_transitions = num_quarters - (prior_state is None)
    expected_transitions = transitions * num_transitions / transitions.sum()
    remaining_transitions = expected_transitions.copy()
    # Never plan a transition into a state that has no quarters left to fill.
    remaining_transitions[:, remaining_states == 0] = 0

    sequence = np.empty(num_quarters, dtype=np.uint8)
    state = 0 if prior_state is None else prior_state

    for position in range(num_quarters):
        for step in range(max_steps):
            # Use the first origin in the fallback order that still has
            # transition budget left.
            weights = np.zeros(9)
            origin = state
            for origin in _ORIGINS[state]:
                weights = remaining_transitions[origin]
                if weights.sum() > 0:
                    break

            total = weights.sum()
            if total > 0:
                # Inverse-CDF sampling over the remaining budget.
                destination = int(np.searchsorted(np.cumsum(weights), rng.random() * total, side="right"))
                break

            # The budget is exhausted everywhere (rounding in the expected
            # counts can cause this), so replenish it in proportion to the
            # quarters still to be filled and try again.
            remaining_transitions = transitions * (num_quarters - position) / transitions.sum()
            remaining_transitions[:, remaining_states == 0] = 0
        else:
            raise RuntimeError(f"No progress after {max_steps} steps at quarter {position + 1}")

        sequence[position] = destination

        # Spend the budget used by this choice.
        remaining_states[destination] -= 1
        if remaining_states[destination] == 0:
            remaining_transitions[:, destination] = 0
        remaining_transitions[origin, destination] = max(remaining_transitions[origin, destination] - 1, 0)
        state = destination

    # State counts must be within one quarter of the monthly expectation.
    counts = np.bincount(sequence, minlength=9)
    if np.any(np.abs(counts - expected_states) > 1 + 1e-12):
        raise RuntimeError("Cloud distribution does not meet the monthly tolerance")

    # Build (origin, destination) pairs; the initial placeholder state of
    # the simulation is excluded, but a real prior state is included.
    origins = sequence[:-1]
    destinations = sequence[1:]
    if prior_state is not None:
        origins = np.concatenate(([prior_state], origins))
        destinations = sequence
    # Encode each pair as origin * 9 + destination so one bincount tallies
    # the whole 9x9 matrix.
    counts = np.bincount(origins.astype(int) * 9 + destinations, minlength=81).reshape(9, 9)

    # Transition counts must be within sqrt(expected), or within one quarter
    # when fewer than one transition is expected. Small tolerances of 1e-12
    # guard against floating point noise at the boundary.
    tolerance = np.where(expected_transitions < 1, 1, np.sqrt(expected_transitions))
    if np.any(np.abs(counts - expected_transitions) > tolerance + 1e-12):
        raise RuntimeError("Transition counts do not meet the monthly tolerance")

    return sequence


def generate_average_clouds(
    historical_transition_matrix: TransitionMatrix,
    historical_distribution: CloudDistribution,
    num_years: int,
    *,
    seed: int | None = 6563,
    max_retries: int = 10000,
    max_steps: int = 100,
) -> Clouds:
    """Generate clouds starting in 1975, validating each calendar month.

    Months are generated one at a time and chained: the final state of one
    month is the prior state of the next, so that transitions across month
    boundaries are counted in the month in which they end.

    Parameters
    ----------
    historical_transition_matrix : `TransitionMatrix`
        Historical transition counts indexed by month and origin.
    historical_distribution : `CloudDistribution`
        Historical cloud state counts indexed by month.
    num_years : `int`
        Minimum duration in Julian years. An extra month is added if needed.
    seed : `int` or `None`, optional
        Seed for a single generator retained across all attempts and months.
    max_retries : `int`, optional
        Maximum candidate sequences attempted per month.
    max_steps : `int`, optional
        Maximum attempts to advance at any one quarter in a candidate.

    Returns
    -------
    clouds : `Clouds`
        Generated eighths, fractional clouds, and quarter-center times.

    Raises
    ------
    RuntimeError
        If a month cannot meet the statistical requirements within the limits.
    ValueError
        If limits or historical counts are invalid.
    """
    _validate_transition_matrix(historical_transition_matrix)
    _validate_cloud_distribution(historical_distribution)

    if set(historical_distribution.index) != set(range(1, 13)):
        raise ValueError("Generation requires historical cloud counts for all twelve months")

    for name, value in (("num_years", num_years), ("max_retries", max_retries), ("max_steps", max_steps)):
        # bool is a subclass of int, but True/False are never intended here.
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    # Fail early, with the month named, rather than partway through generation.
    for month in range(1, 13):
        distribution = historical_distribution.loc[month].to_numpy(dtype=float)
        transitions = historical_transition_matrix.loc[month].to_numpy(dtype=float)
        for name, values, shape in (
            ("cloud state", distribution, (9,)),
            ("transition", transitions, (9, 9)),
        ):
            if values.shape != shape or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Invalid {name} counts for month {month}")
            if values.sum() == 0:
                raise ValueError(f"No historical {name} counts for month {month}")

    # One generator for the whole run: reseeding per retry would reproduce
    # the same failing sequence.
    rng = np.random.default_rng(seed)

    dates: list[date] = []
    sequences: list[npt.NDArray[np.uint8]] = []
    prior_state: int | None = None

    # Loop one month past num_years * 12: that final pass is only used as a
    # check on whether an extra month is needed, never to generate one
    # unless the span is too short.
    for offset in range(num_years * 12 + 1):
        year, month = 1975 + offset // 12, offset % 12 + 1

        if offset == num_years * 12:
            # CloudData wraps the database using a whole number of Julian
            # years, so the data must span at least num_years of them.
            clouds = make_cloud_frame(dates, np.concatenate(sequences))
            if int(clouds.c_date.iloc[-1]) - int(clouds.c_date.iloc[0]) >= num_years * 31557600:
                return clouds

        days = calendar.monthrange(year, month)[1]

        # Retry with the shared generator: each attempt draws fresh random
        # numbers, so a failed candidate is not repeated.
        for attempt in range(max_retries):
            try:
                sequence = _generate_month(
                    historical_transition_matrix.loc[month].to_numpy(dtype=float),
                    historical_distribution.loc[month].to_numpy(dtype=float),
                    days * 4,
                    prior_state,
                    rng,
                    max_steps,
                )
            except RuntimeError as error:
                last_error = error
            else:
                break
        else:
            # Only reached if every attempt failed.
            raise RuntimeError(
                f"Could not generate {year}-{month:02d} after {max_retries} attempts: {last_error}"
            ) from last_error

        sequences.append(sequence)
        dates.extend(date(year, month, day) for day in range(1, days + 1))
        prior_state = int(sequence[-1])

    return make_cloud_frame(dates, np.concatenate(sequences))


def cloud_generation_workflow(
    historical_paths: str | PathLike | Sequence[str | PathLike],
    output_database: str | PathLike,
    num_years: int = 20,
    *,
    cloud_histogram: str | PathLike | None = None,
    transition_histograms: str | PathLike | None = None,
    seed: int | None = 6563,
    max_retries: int = 10000,
    max_steps: int = 100,
) -> Clouds:
    """Generate and save clouds with optional diagnostics.

    Runs the full pipeline: read historical records, compute their
    statistics, generate simulated clouds, write the database, and
    optionally write diagnostic plots.

    Parameters
    ----------
    historical_paths : `str`, `os.PathLike`, or sequence of those
        Historical CTIO record file or files.
    output_database : `str` or `os.PathLike`
        Path of the SQLite database to write.
    num_years : `int`, optional
        Minimum number of years to simulate.
    cloud_histogram : `str` or `os.PathLike`, optional
        If given, path at which to save the cloud state histograms.
    transition_histograms : `str` or `os.PathLike`, optional
        If given, path at which to save the transition count histograms.
    seed : `int` or `None`, optional
        Seed for the random number generator.
    max_retries : `int`, optional
        Maximum candidate sequences attempted per month.
    max_steps : `int`, optional
        Maximum attempts to advance at any one quarter in a candidate.

    Returns
    -------
    simulated : `Clouds`
        The simulated clouds that were written to ``output_database``.
    """
    historical = read_historical_clouds(historical_paths)
    distribution = count_cloud_states(historical)
    transitions = compute_transition_matrix(historical)

    simulated = generate_average_clouds(
        transitions, distribution, num_years, seed=seed, max_retries=max_retries, max_steps=max_steps
    )
    save_clouds(output_database, simulated)

    # Close each figure after saving: pyplot keeps figures alive until
    # closed, which would leak memory in long-lived sessions.
    if cloud_histogram is not None:
        figure = plot_cloud_histogram(distribution, count_cloud_states(simulated))
        try:
            figure.savefig(cloud_histogram)
        finally:
            plt.close(figure)

    if transition_histograms is not None:
        figure = plot_transition_histograms(transitions, compute_transition_matrix(simulated))
        try:
            figure.savefig(transition_histograms)
        finally:
            plt.close(figure)

    return simulated


@click.command()
@click.argument("historical_paths", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-database", "-o", required=True, type=click.Path(dir_okay=False))
@click.option("--num-years", default=20, show_default=True, type=click.IntRange(min=1))
@click.option("--seed", default=6563, show_default=True, type=int)
@click.option("--max-retries", default=10000, show_default=True, type=click.IntRange(min=1))
@click.option("--max-steps", default=100, show_default=True, type=click.IntRange(min=1))
@click.option("--cloud-histogram", type=click.Path(dir_okay=False))
@click.option("--transition-histograms", type=click.Path(dir_okay=False))
def generate_average_clouds_cli(**kwargs: Any) -> None:
    """Generate a scheduler cloud database from CTIO record files.

    HISTORICAL_PATHS are one or more whitespace-separated CTIO record
    files. The options map directly onto `cloud_generation_workflow`.
    """
    try:
        cloud_generation_workflow(**kwargs)
    except (ValueError, RuntimeError, OSError) as error:
        # Present expected failures as a clean message, not a traceback.
        raise click.ClickException(str(error)) from error
