import unittest

import numpy as np

import rubin_sim.maf.metrics as metrics


class TestTemplateMetrics(unittest.TestCase):
    def test_incremental_template_metric(self):
        """
        Test the incremental template metric
        """
        npoints = 10
        names = ["seeingFwhmEff", "fiveSigmaDepth", "observationStartMJD", "night", "filter"]
        types = [float, float, float, int, str]
        data = np.zeros(npoints, dtype=list(zip(names, types)))
        data["seeingFwhmEff"] = [0.7, 0.5, 1.5, 1.2, 0.6, 0.65, 1.2, 1.2, 1.2, 1.2]
        data["fiveSigmaDepth"] = [24.5, 24.5, 23, 23.5, 23.5, 24.5, 23, 23, 23, 23]
        data["observationStartMJD"] = np.arange(npoints)
        data["night"] = np.arange(npoints)
        data["filter"] = ["r" for i in range(npoints)]

        ttm = metrics.TemplateTime(n_images_in_template=3, seeing_percentile=50, m5_percentile=50)
        metric_val = ttm.run(data)

        night_created = ttm.reduce_Night_template_created(metric_val)
        n_nights_no_template = ttm.reduce_N_nights_without_template(metric_val)
        n_images_until_template = ttm.reduce_N_images_until_template(metric_val)
        n_images_with_template = ttm.reduce_N_images_with_template(metric_val)
        template_m5 = ttm.reduce_Template_m5(metric_val)
        template_seeing = ttm.reduce_Template_seeing(metric_val)

        self.assertEqual(night_created, 4)
        self.assertEqual(n_nights_no_template, 4)
        self.assertEqual(n_images_until_template, 5)
        self.assertEqual(n_images_with_template, 5)
        self.assertAlmostEqual(template_m5, 24.917687379997627)
        self.assertEqual(template_seeing, 0.607672394977614)

        # alternative values

        ttm = metrics.TemplateTime(n_images_in_template=4, seeing_percentile=75, m5_percentile=75)
        metric_val = ttm.run(data)

        night_created = ttm.reduce_Night_template_created(metric_val)
        n_nights_no_template = ttm.reduce_N_nights_without_template(metric_val)
        n_images_until_template = ttm.reduce_N_images_until_template(metric_val)
        n_images_with_template = ttm.reduce_N_images_with_template(metric_val)
        template_m5 = ttm.reduce_Template_m5(metric_val)
        template_seeing = ttm.reduce_Template_seeing(metric_val)

        self.assertEqual(night_created, 5)
        self.assertEqual(n_nights_no_template, 5)
        self.assertEqual(n_images_until_template, 6)
        self.assertEqual(n_images_with_template, 4)
        self.assertAlmostEqual(template_m5, 25.124349265777216)
        self.assertEqual(template_seeing, 0.6213856145706174)

        # test no template created

        ttm = metrics.TemplateTime(n_images_in_template=3, seeing_percentile=25, m5_percentile=25)
        metric_val = ttm.run(data)

        self.assertEqual(metric_val, -666)


if __name__ == "__main__":
    unittest.main()
