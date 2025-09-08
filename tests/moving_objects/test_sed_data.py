import os
import unittest

from rubin_scheduler.data import get_data_dir

import rubin_sim.phot_utils as phot_utils


class SedDataUnitTest(unittest.TestCase):

    def setUp(self) -> None:
        # Read the lsst bandpasses
        lsst = {}
        lsst_filterlist = "ugrizy"
        for b in lsst_filterlist:
            lsst[b] = phot_utils.Bandpass()
            lsst[b].read_throughput(os.path.join(get_data_dir(), "throughputs", "baseline", f"total_{b}.dat"))
        self.lsst = lsst

    def test_sun(self) -> None:
        sun = phot_utils.Sed()
        sun.read_sed_flambda(os.path.join(get_data_dir(), "movingObjects", "kurucz_sun.gz"))
        expected_mags = {"u": -25.3, "g": -26.5, "r": -27.0, "i": -27.0, "z": -27.1, "y": -27.1}
        for b in "ugrizy":
            mag = sun.calc_mag(self.lsst[b])
            self.assertTrue(abs(mag - expected_mags[b]) < 0.2)

    def test_seds(self):
        # We have some sample asteroid spectra in rubin_sim_data_dir -
        # However, those are 'observed' spectra
        # (already includes a sample solar spectra)
        ast_templates = [
            k.replace(".dat.gz", "")
            for k in os.listdir(os.path.join(get_data_dir(), "movingObjects"))
            if "kurucz" not in k and "harris" not in k
        ]
        # Read the sso seds from disk (using rubin_sim.photUtils.Sed)
        sso = {}
        for k in ast_templates:
            sso[k] = phot_utils.Sed()
            sso[k].read_sed_flambda(os.path.join(get_data_dir(), "movingObjects", f"{k}.dat.gz"))

        # Calculate the colors for each type:
        # referenc e bandpass first
        refband_name = "r"
        refband = self.lsst[refband_name]
        # Calculate the reference bandpass magnitudes
        refmags = {}
        for k in sso:
            refmags[k] = sso[k].calc_mag(refband)

        # And now the colors
        colors = {}
        for k in sso:
            colors[k] = {}
            for f in "ugrizy":
                colors[k][f"lsst_{f}"] = sso[k].calc_mag(self.lsst[f]) - refmags[k]


if __name__ == "__main__":
    unittest.main()
