import numpy as np
import unittest
import rubin_sim.scheduler.basis_functions as bf
from rubin_sim.scheduler.utils import standard_goals, calc_norm_factor
from rubin_sim.scheduler.surveys import (
    generate_dd_surveys,
    GreedySurvey,
    BlobSurvey,
)
from rubin_sim.scheduler.schedulers import CoreScheduler
from rubin_sim.scheduler import sim_runner
from rubin_sim.scheduler.model_observatory import ModelObservatory
import rubin_sim.scheduler.detailers as detailers
from rubin_sim.scheduler.example import example_scheduler


def gen_greedy_surveys(nside):
    """
    Make a quick set of greedy surveys
    """
    target_map = standard_goals(nside=nside)
    filters = ["g", "r", "i", "z", "y"]
    surveys = []

    for filtername in filters:
        bfs = []
        bfs.append(bf.M5DiffBasisFunction(filtername=filtername, nside=nside))
        bfs.append(
            bf.TargetMapBasisFunction(
                filtername=filtername,
                target_map=target_map[filtername],
                out_of_bounds_val=np.nan,
                nside=nside,
            )
        )
        bfs.append(bf.SlewtimeBasisFunction(filtername=filtername, nside=nside))
        bfs.append(bf.StrictFilterBasisFunction(filtername=filtername))
        # Masks, give these 0 weight
        bfs.append(
            bf.ZenithShadowMaskBasisFunction(
                nside=nside, shadow_minutes=60.0, max_alt=76.0
            )
        )
        bfs.append(bf.MoonAvoidanceBasisFunction(nside=nside, moon_distance=30.0))
        bfs.append(bf.CloudedOutBasisFunction())

        bfs.append(bf.FilterLoadedBasisFunction(filternames=filtername))
        bfs.append(bf.PlanetMaskBasisFunction(nside=nside))

        weights = np.array([3.0, 0.3, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        surveys.append(
            GreedySurvey(
                bfs,
                weights,
                block_size=1,
                filtername=filtername,
                dither=True,
                nside=nside,
            )
        )
    return surveys


def gen_blob_surveys(nside):
    """
    make a quick set of blob surveys
    """
    target_map = standard_goals(nside=nside)
    norm_factor = calc_norm_factor(target_map)

    filter1s = ["g"]  # , 'r', 'i', 'z', 'y']
    filter2s = ["g"]  # , 'r', 'i', None, None]

    pair_surveys = []
    for filtername, filtername2 in zip(filter1s, filter2s):
        detailer_list = []
        bfs = []
        bfs.append(bf.M5DiffBasisFunction(filtername=filtername, nside=nside))
        if filtername2 is not None:
            bfs.append(bf.M5DiffBasisFunction(filtername=filtername2, nside=nside))
        bfs.append(
            bf.TargetMapBasisFunction(
                filtername=filtername,
                target_map=target_map[filtername],
                out_of_bounds_val=np.nan,
                nside=nside,
                norm_factor=norm_factor,
            )
        )
        if filtername2 is not None:
            bfs.append(
                bf.TargetMapBasisFunction(
                    filtername=filtername2,
                    target_map=target_map[filtername2],
                    out_of_bounds_val=np.nan,
                    nside=nside,
                    norm_factor=norm_factor,
                )
            )
        bfs.append(bf.SlewtimeBasisFunction(filtername=filtername, nside=nside))
        bfs.append(bf.StrictFilterBasisFunction(filtername=filtername))
        # Masks, give these 0 weight
        bfs.append(
            bf.ZenithShadowMaskBasisFunction(
                nside=nside, shadow_minutes=60.0, max_alt=76.0
            )
        )
        bfs.append(bf.MoonAvoidanceBasisFunction(nside=nside, moon_distance=30.0))
        bfs.append(bf.CloudedOutBasisFunction())
        # feasibility basis fucntions. Also give zero weight.
        filternames = [fn for fn in [filtername, filtername2] if fn is not None]
        bfs.append(bf.FilterLoadedBasisFunction(filternames=filternames))
        bfs.append(bf.TimeToTwilightBasisFunction(time_needed=22.0))
        bfs.append(bf.NotTwilightBasisFunction())
        bfs.append(bf.PlanetMaskBasisFunction(nside=nside))

        weights = np.array(
            [3.0, 3.0, 0.3, 0.3, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        if filtername2 is None:
            # Need to scale weights up so filter balancing still works properly.
            weights = np.array([6.0, 0.6, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if filtername2 is None:
            survey_name = "blob, %s" % filtername
        else:
            survey_name = "blob, %s%s" % (filtername, filtername2)
        if filtername2 is not None:
            detailer_list.append(detailers.TakeAsPairsDetailer(filtername=filtername2))
        pair_surveys.append(
            BlobSurvey(
                bfs,
                weights,
                filtername1=filtername,
                filtername2=filtername2,
                survey_note=survey_name,
                ignore_obs="DD",
                detailers=detailer_list,
                nside=nside,
            )
        )
    return pair_surveys


class TestExample(unittest.TestCase):
    def test_example(self):
        """Try out the example scheduler."""
        nside = 32
        survey_length = 2.0  # days
        scheduler = example_scheduler(nside=nside)
        observatory = ModelObservatory(nside=nside)
        observatory, scheduler, observations = sim_runner(
            observatory, scheduler, survey_length=survey_length, filename=None
        )
        # Check that greedy observed some
        assert "greedy" in observations["note"]
        # Make sure lots of observations executed
        assert observations.size > 1000
        # Make sure nothing tried to look through the earth
        assert np.min(observations["alt"]) > 0


class TestFeatures(unittest.TestCase):
    def test_greedy(self):
        """
        Set up a greedy survey and run for a few days. A crude way to touch lots of code.
        """
        nside = 32
        survey_length = 2.0  # days

        surveys = gen_greedy_surveys(nside)
        # Depricating Pairs_survey_scripted
        # surveys.append(Pairs_survey_scripted(None, ignore_obs='DD'))

        # Set up the DD
        dd_surveys = generate_dd_surveys(nside=nside)
        surveys.extend(dd_surveys)

        scheduler = CoreScheduler(surveys, nside=nside)
        observatory = ModelObservatory(nside=nside)
        observatory, scheduler, observations = sim_runner(
            observatory, scheduler, survey_length=survey_length, filename=None
        )

        # Check that greedy observed some
        assert "" in observations["note"]
        # Make sure lots of observations executed
        assert observations.size > 1000
        # Make sure nothing tried to look through the earth
        assert np.min(observations["alt"]) > 0

    def test_blobs(self):
        """
        Set up a blob selection survey
        """
        nside = 32
        survey_length = 2.0  # days

        surveys = []
        # Set up the DD
        dd_surveys = generate_dd_surveys(nside=nside)
        surveys.append(dd_surveys)

        surveys.append(gen_blob_surveys(nside))
        surveys.append(gen_greedy_surveys(nside))

        scheduler = CoreScheduler(surveys, nside=nside)
        observatory = ModelObservatory(nside=nside)
        observatory, scheduler, observations = sim_runner(
            observatory, scheduler, survey_length=survey_length, filename=None
        )

        # Make sure some blobs executed
        assert "blob, gg, b" in observations["note"]
        assert "blob, gg, a" in observations["note"]
        # Make sure some greedy executed
        assert "" in observations["note"]
        # Make sure lots of observations executed
        assert observations.size > 1000
        # Make sure nothing tried to look through the earth
        assert np.min(observations["alt"]) > 0

    def test_nside(self):
        """
        test running at higher nside
        """
        nside = 64
        survey_length = 2.0  # days

        surveys = []
        # Set up the DD
        dd_surveys = generate_dd_surveys(nside=nside)
        surveys.append(dd_surveys)

        surveys.append(gen_blob_surveys(nside))
        surveys.append(gen_greedy_surveys(nside))

        scheduler = CoreScheduler(surveys, nside=nside)
        observatory = ModelObservatory(nside=nside)
        observatory, scheduler, observations = sim_runner(
            observatory, scheduler, survey_length=survey_length, filename=None
        )

        # Make sure some blobs executed
        assert "blob, gg, b" in observations["note"]
        assert "blob, gg, a" in observations["note"]
        # Make sure some greedy executed
        assert "" in observations["note"]
        # Make sure lots of observations executed
        assert observations.size > 1000
        # Make sure nothing tried to look through the earth
        assert np.min(observations["alt"]) > 0


if __name__ == "__main__":
    unittest.main()
