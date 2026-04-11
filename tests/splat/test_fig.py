import glob
import shutil
import unittest

import matplotlib.pylab as plt
import pandas as pd

import rubin_sim.splat as maf


class TestFig(unittest.TestCase):

    def tearDown(self):

        shutil.rmtree("maf_figs")

    def test_fig_saver(self):

        fs = maf.FigSaver()

        info = {}
        info["metric: name"] = "test name"
        info["slicer: nside"] = 42

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])

        fs(fig, info=info)

        # check that we made pdf and png
        f1 = glob.glob("maf_figs/*.pdf")
        assert len(f1) > 0
        f2 = glob.glob("maf_figs/*.png")
        assert len(f2) > 0

        # Check that we wrote to the default tracking file
        f3 = glob.glob("maf_figs/*.db")
        assert len(f3) > 0

        stats = {"value": [12, 12], "run_name": ["test_name", "ack"]}
        stats = pd.DataFrame.from_dict(stats)

        fs.save_stats(stats)

        # If we have a single row, use from_records
        stats = {"value": 12, "run_name": "test_name"}
        stats = pd.DataFrame.from_records([stats])

        fs.save_stats(stats)


if __name__ == "__main__":
    unittest.main()
