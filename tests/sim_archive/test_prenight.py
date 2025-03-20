import importlib.util
import unittest
from tempfile import TemporaryDirectory

try:
    from lsst.resources import ResourcePath

    HAVE_RESOURCES = True
except ModuleNotFoundError:
    HAVE_RESOURCES = False

if HAVE_RESOURCES:
    from rubin_sim.sim_archive import prenight_sim_cli

# We need rubin_sim to get the baseline sim
# Tooling prefers checking that it exists using importlib rather
# than importing it and not actually using it.
HAVE_RUBIN_SIM = importlib.util.find_spec("rubin_sim")


class TestPrenight(unittest.TestCase):
    @unittest.skip("Too slow")
    @unittest.skipIf(not HAVE_RESOURCES, "No lsst.resources")
    @unittest.skipIf(not HAVE_RUBIN_SIM, "No rubin_sim, needed for rubin_sim.data.get_baseline")
    def test_prenight(self):
        with TemporaryDirectory() as test_archive_dir:
            archive_uri = ResourcePath(test_archive_dir).geturl()  # type: ignore
            prenight_sim_cli("--archive", archive_uri)
