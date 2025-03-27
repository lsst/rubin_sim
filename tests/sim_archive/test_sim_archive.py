import importlib
import lzma
import pickle
import unittest
import urllib
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from astropy.time import Time
from rubin_scheduler.scheduler import sim_runner
from rubin_scheduler.scheduler.example import example_scheduler
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.utils import SURVEY_START_MJD

HAVE_LSST_RESOURCES = importlib.util.find_spec("lsst") and importlib.util.find_spec("lsst.resources")
if HAVE_LSST_RESOURCES:
    from lsst.resources import ResourcePath

    from rubin_sim.sim_archive.sim_archive import (
        check_opsim_archive_resource,
        compile_sim_metadata,
        fetch_latest_prenight_sim_for_nights,
        fetch_obsloctap_visits,
        find_latest_prenight_sim_for_nights,
        make_sim_archive_cli,
        make_sim_archive_dir,
        read_archived_sim_metadata,
        read_sim_metadata_from_hdf,
        transfer_archive_dir,
        verify_compiled_sim_metadata,
    )


class TestSimArchive(unittest.TestCase):
    @unittest.skipIf(not HAVE_LSST_RESOURCES, "No lsst.resources")
    def test_sim_archive(self):
        # Begin by running a short simulation
        sim_start_mjd = SURVEY_START_MJD
        sim_duration = 1  # days
        scheduler = example_scheduler(mjd_start=sim_start_mjd)
        scheduler.keep_rewards = True
        observatory = ModelObservatory(mjd_start=sim_start_mjd)

        # Record the state of the scheduler at the start of the sim.
        data_dir = TemporaryDirectory()
        data_path = Path(data_dir.name)

        scheduler_fname = data_path.joinpath("scheduler.pickle.xz")
        with lzma.open(scheduler_fname, "wb", format=lzma.FORMAT_XZ) as pio:
            pickle.dump(scheduler, pio)

        files_to_archive = {"scheduler": scheduler_fname}

        # Run the simulation
        sim_runner_kwargs = {
            "sim_start_mjd": sim_start_mjd,
            "sim_duration": sim_duration,
            "record_rewards": True,
        }

        observatory, scheduler, observations, reward_df, obs_rewards = sim_runner(
            observatory, scheduler, **sim_runner_kwargs
        )

        # Make the scratch sim archive
        make_sim_archive_dir(
            observations,
            reward_df=reward_df,
            obs_rewards=obs_rewards,
            in_files=files_to_archive,
            sim_runner_kwargs=sim_runner_kwargs,
            tags=["test"],
            label="test",
            data_path=data_path,
        )

        # Move the scratch sim archive to a test resource
        test_resource_dir = TemporaryDirectory()
        test_resource_uri = "file://" + test_resource_dir.name
        sim_archive_uri = transfer_archive_dir(data_dir.name, test_resource_uri)

        # Check the saved archive
        archive_check = check_opsim_archive_resource(sim_archive_uri)
        self.assertEqual(
            archive_check.keys(),
            set(
                [
                    "opsim.db",
                    "rewards.h5",
                    "scheduler.pickle.xz",
                    "obs_stats.txt",
                    "environment.txt",
                    "pypi.json",
                ]
            ),
        )
        for value in archive_check.values():
            self.assertTrue(value)

        # Read back the metadata
        archive_metadata = read_archived_sim_metadata(test_resource_uri)
        base = sim_archive_uri.dirname().geturl().removeprefix(test_resource_uri).rstrip("/").lstrip("/")
        expected_label = f"{base} test"
        self.assertEqual(archive_metadata[sim_archive_uri.geturl()]["label"], expected_label)

        # Cache the metadata
        test_compiled_metadata_uri = test_resource_uri + "/compiled_metadata_cache.h5"

        # Test reading from cached metadata
        compile_sim_metadata(test_resource_uri, test_compiled_metadata_uri)
        read_sim_metadata_from_hdf(test_compiled_metadata_uri)
        read_archived_sim_metadata(test_resource_uri, compilation_resource=test_compiled_metadata_uri)
        verify_compiled_sim_metadata(test_resource_uri, test_compiled_metadata_uri)

    @unittest.skipIf(not HAVE_LSST_RESOURCES, "No lsst.resources")
    @unittest.skipIf(importlib.util.find_spec("schedview") is None, "No schedview")
    def test_cli(self):
        test_resource_path = ResourcePath("resource://schedview/data/")
        with test_resource_path.join("sample_opsim.db").as_local() as local_rp:
            opsim = urllib.parse.urlparse(local_rp.geturl()).path

        with test_resource_path.join("sample_rewards.h5").as_local() as local_rp:
            rewards = urllib.parse.urlparse(local_rp.geturl()).path

        with test_resource_path.join("sample_scheduler.pickle.xz").as_local() as local_rp:
            scheduler = urllib.parse.urlparse(local_rp.geturl()).path

        with TemporaryDirectory() as test_archive_dir:
            test_archive_uri = f"file://{test_archive_dir}/"
            make_sim_archive_cli(
                "Test",
                opsim,
                "--rewards",
                rewards,
                "--scheduler",
                scheduler,
                "--archive_base_uri",
                test_archive_uri,
            )

    @unittest.skipIf(not HAVE_LSST_RESOURCES, "No lsst.resources")
    def test_find_latest_prenight_sim_for_night(self):
        day_obs = "2025-03-25"
        max_simulation_age = int(np.ceil(Time.now().mjd - Time(day_obs).mjd)) + 1
        sim_metadata = find_latest_prenight_sim_for_nights(day_obs, max_simulation_age=max_simulation_age)
        assert sim_metadata["simulated_dates"]["first"] <= day_obs <= sim_metadata["simulated_dates"]["last"]

    @unittest.skipIf(not HAVE_LSST_RESOURCES, "No lsst.resources")
    def test_fetch_latest_prenight_sim_for_night(self):
        day_obs = "2025-03-25"
        max_simulation_age = int(np.ceil(Time.now().mjd - Time(day_obs).mjd)) + 1
        visits = fetch_latest_prenight_sim_for_nights(day_obs, max_simulation_age=max_simulation_age)
        assert len(visits) > 0

    @unittest.skipIf(not HAVE_LSST_RESOURCES, "No lsst.resources")
    def test_fetch_obsloctap_visits(self):
        day_obs = "2025-03-25"
        num_nights = 2
        visits = pd.DataFrame(fetch_obsloctap_visits(day_obs, nights=num_nights))
        assert np.floor(visits["observationStartMJD"].min() - 0.5) == Time(day_obs).mjd
        assert np.floor(visits["observationStartMJD"].max() - 0.5) == Time(day_obs).mjd + num_nights - 1


if __name__ == "__main__":
    unittest.main()
