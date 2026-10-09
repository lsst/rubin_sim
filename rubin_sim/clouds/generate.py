"""Generate monthly representative cloud sequences from CTIO records."""

import calendar
from datetime import date
from os import PathLike
from typing import Sequence

import click
import matplotlib.pyplot as plt
import numpy as np

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

_ORIGINS = (
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
    transitions: np.ndarray,
    distribution: np.ndarray,
    num_quarters: int,
    prior_state: int | None,
    rng: np.random.Generator,
    max_steps: int,
) -> np.ndarray:
    """Sample a count-constrained sequence and verify its transitions."""
    expected_states = num_quarters * distribution / distribution.sum()
    remaining_states = np.floor(expected_states).astype(int)
    fractions = expected_states - remaining_states
    num_ceil = num_quarters - remaining_states.sum()
    if num_ceil:
        choices = rng.choice(9, size=num_ceil, replace=False, p=fractions / fractions.sum())
        remaining_states[choices] += 1

    num_transitions = num_quarters - (prior_state is None)
    expected_transitions = transitions * num_transitions / transitions.sum()
    remaining_transitions = expected_transitions.copy()
    remaining_transitions[:, remaining_states == 0] = 0
    sequence = np.empty(num_quarters, dtype=np.uint8)
    state = 0 if prior_state is None else prior_state
    for position in range(num_quarters):
        for step in range(max_steps):
            weights = np.zeros(9)
            origin = state
            for origin in _ORIGINS[state]:
                weights = remaining_transitions[origin]
                if weights.sum() > 0:
                    break
            total = weights.sum()
            if total > 0:
                destination = int(np.searchsorted(np.cumsum(weights), rng.random() * total, side="right"))
                break
            remaining_transitions = transitions * (num_quarters - position) / transitions.sum()
            remaining_transitions[:, remaining_states == 0] = 0
        else:
            raise RuntimeError(f"No progress after {max_steps} steps at quarter {position + 1}")
        sequence[position] = destination
        remaining_states[destination] -= 1
        if remaining_states[destination] == 0:
            remaining_transitions[:, destination] = 0
        remaining_transitions[origin, destination] = max(remaining_transitions[origin, destination] - 1, 0)
        state = destination

    counts = np.bincount(sequence, minlength=9)
    if np.any(np.abs(counts - expected_states) > 1 + 1e-12):
        raise RuntimeError("Cloud distribution does not meet R-3")
    origins = sequence[:-1]
    destinations = sequence[1:]
    if prior_state is not None:
        origins = np.concatenate(([prior_state], origins))
        destinations = sequence
    counts = np.bincount(origins.astype(int) * 9 + destinations, minlength=81).reshape(9, 9)
    # Rare pairs allow one-quarter deviations, including zero expectation.
    tolerance = np.where(expected_transitions < 1, 1, np.sqrt(expected_transitions))
    if np.any(np.abs(counts - expected_transitions) > tolerance + 1e-12):
        raise RuntimeError("Transition counts do not meet R-4")
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

    Parameters
    ----------
    historical_transition_matrix : pandas.DataFrame
        Historical transition counts indexed by month and origin.
    historical_distribution : pandas.DataFrame
        Historical cloud state counts indexed by month.
    num_years : int
        Minimum duration in Julian years. An extra month is added if needed.
    seed : int or None, optional
        Seed for a single generator retained across all attempts and months.
    max_retries : int, optional
        Maximum candidate sequences attempted per month.
    max_steps : int, optional
        Maximum attempts to advance at any one quarter in a candidate.

    Returns
    -------
    clouds : pandas.DataFrame
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
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
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

    rng = np.random.default_rng(seed)
    dates = []
    sequences = []
    prior_state = None
    for offset in range(num_years * 12 + 1):
        year, month = 1975 + offset // 12, offset % 12 + 1
        if offset == num_years * 12:
            clouds = make_cloud_frame(dates, np.concatenate(sequences))
            if int(clouds.c_date.iloc[-1]) - int(clouds.c_date.iloc[0]) >= num_years * 31557600:
                return clouds
        days = calendar.monthrange(year, month)[1]
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
    """Generate and save clouds with optional diagnostics."""
    historical = read_historical_clouds(historical_paths)
    distribution = count_cloud_states(historical)
    transitions = compute_transition_matrix(historical)
    simulated = generate_average_clouds(
        transitions, distribution, num_years, seed=seed, max_retries=max_retries, max_steps=max_steps
    )
    save_clouds(output_database, simulated)
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
def generate_average_clouds_cli(**kwargs) -> None:
    """Generate a scheduler cloud database from CTIO record files."""
    try:
        cloud_generation_workflow(**kwargs)
    except (ValueError, RuntimeError, OSError) as error:
        raise click.ClickException(str(error)) from error
