"""Regression tests for chimera and snapshot progress tracking.

All visits are small synthetic data built here.
"""

import os
import tempfile
import unittest
from importlib.metadata import distribution
from unittest.mock import patch

import numpy as np
import pandas as pd
from astropy.time import Time
from click.testing import CliRunner

from rubin_sim.maf.progress import (
    build_chimera,
    build_chimeras,
    build_chimeras_cmd,
    dayobs_range,
)

BANDS = "ugrizy"
NSIDE = 16


def _make_visits(first_dayobs, n_nights, per_night, seed):
    """Make deterministic synthetic visits, without a ``dayObs`` column.

    Visits start 0.1 to 0.4 days after 00:00 UTC on the calendar day
    after each ``dayObs`` date, which is within that observing day
    (UTC-12). Only columns the progress batches fetch are included.
    """
    rng = np.random.default_rng(seed)
    start = str(first_dayobs)
    mjd0 = Time(f"{start[:4]}-{start[4:6]}-{start[6:]}T00:00:00").mjd + 1.0
    n_visits = n_nights * per_night
    night = np.repeat(np.arange(n_nights), per_night)
    return pd.DataFrame(
        {
            "observationId": np.arange(n_visits) + 100000 * seed,
            "observationStartMJD": mjd0 + night + rng.uniform(0.1, 0.4, n_visits),
            "fieldRA": rng.uniform(0, 360, n_visits),
            "fieldDec": np.degrees(np.arcsin(rng.uniform(-1, 0.2, n_visits))),
            "band": np.tile(list(BANDS), n_visits // 6 + 1)[:n_visits],
            "fiveSigmaDepth": 24.0 + rng.normal(0, 0.3, n_visits),
            "visitExposureTime": 30.0,
            "rotSkyPos": 0.0,
        }
    )


class TestChimeraConstruction(unittest.TestCase):
    """Construction and date-range tests for chimera visit sequences."""

    def test_build_chimera(self):
        dates = {"start_dayobs": 20260102, "transition_dayobs": 20260104, "end_dayobs": 20260107}
        completed = pd.DataFrame(
            [
                (1, 20260101, 24.0),  # Before start.
                (2, 20260102, 24.0),  # At start: included.
                (3, 20260103, np.nan),
                (4, 20260104, 24.0),  # At transition: included.
                (5, 20260104, 0.0),
                (6, 20260104, -1.0),
                (7, 20260105, 24.0),  # After transition.
            ],
            columns=["observationId", "dayObs", "fiveSigmaDepth"],
        ).assign(band="g", completed_only=range(7))
        baseline = pd.DataFrame(
            [
                (101, 20260104, 23.0),  # At transition: excluded.
                (102, 20260105, 23.0),  # After transition: included.
                (103, 20260105, np.nan),
                (104, 20260106, 0.0),
                (105, 20260107, 23.0),  # At target: included.
                (106, 20260108, 23.0),  # After target.
                (107, 20260105, -1.0),
            ],
            columns=["observationId", "dayObs", "fiveSigmaDepth"],
        ).assign(band="r", exposures=2, baseline_only=range(7))
        result = build_chimera(completed, baseline, **dates)

        # Boundaries (S, T, E inclusive; T exclusive for baseline) and
        # depth cuts (NaN, zero, negative) on both sources.
        self.assertEqual(result["observationId"].tolist(), [2, 4, 102, 105])
        self.assertEqual(result["simulated"].tolist(), [False, False, True, True])
        # Missing completed-visits exposures is 1; baseline value is kept.
        self.assertEqual(result["exposures"].tolist(), [1, 1, 2, 2])
        # Only columns common to both inputs, plus simulated.
        self.assertEqual(
            set(result.columns),
            {"observationId", "dayObs", "fiveSigmaDepth", "band", "exposures", "simulated"},
        )

        # Null exposures in an existing completed-visits column become 1.
        with_exposures = completed.assign(exposures=5.0)
        with_exposures.loc[with_exposures["observationId"] == 2, "exposures"] = 3.0
        with_exposures.loc[with_exposures["observationId"] == 4, "exposures"] = np.nan
        result = build_chimera(with_exposures, baseline, **dates)
        self.assertEqual(result["exposures"].tolist(), [3, 1, 2, 2])

        # A cutoff equal to the baseline depth excludes baseline visits.
        with patch("rubin_sim.maf.progress.FIVE_SIGMA_DEPTH_LIMIT", 23.0):
            result = build_chimera(completed, baseline, **dates)
        self.assertEqual(result["observationId"].tolist(), [2, 4])

    def test_build_chimeras_series(self):
        # Completed visits run past end_dayobs, and the step does not
        # land on the last completed date. Dates cross a month boundary.
        completed = pd.DataFrame(
            {
                "observationId": range(12),
                "dayObs": np.repeat([20260129, 20260130, 20260131, 20260201, 20260202, 20260203], 2),
                "fiveSigmaDepth": 24.0,
            }
        )
        baseline = pd.DataFrame(
            {
                "observationId": range(100, 118),
                "dayObs": np.repeat(
                    [
                        20260129,
                        20260130,
                        20260131,
                        20260201,
                        20260202,
                        20260203,
                        20260204,
                        20260205,
                        20260206,
                    ],
                    2,
                ),
                "fiveSigmaDepth": 23.0,
            }
        )
        with tempfile.TemporaryDirectory() as out_dir:
            specs = build_chimeras(completed, baseline, 20260129, 20260201, step=2, out_dir=out_dir)
            self.assertEqual([t for t, _ in specs], [20260129, 20260131, 20260202, 20260203])
            for transition, path in specs:
                self.assertEqual(os.path.basename(path), f"chimera_{transition}.h5")
            chimeras = {t: pd.read_hdf(path, key="observations") for t, path in specs}

        first, last = chimeras[20260129], chimeras[20260203]
        # Baseline visits stop at end_dayobs ...
        self.assertEqual(first.loc[first["simulated"], "dayObs"].max(), 20260201)
        # ... but completed visits do not, and nothing is simulated
        # once the transition is after end_dayobs.
        self.assertEqual(len(last), len(completed))
        self.assertFalse(last["simulated"].any())
        self.assertEqual(last["dayObs"].max(), 20260203)

    def test_dayobs_range(self):
        self.assertEqual(dayobs_range(20260130, 20260202), [20260130, 20260131, 20260201, 20260202])
        with self.assertRaises(ValueError):
            dayobs_range(20260101, 20260105, 0)

