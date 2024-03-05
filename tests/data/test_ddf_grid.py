import unittest

from rubin_scheduler.scheduler.surveys import generate_ddf_grid


class GenerateDDFTest(unittest.TestCase):
    def testGenDDF(self):
        """
        Test the DDF grid generator over in rubin_scheduler
        """
        # This triggers several RunTimeErrors (intentionally?).
        result = generate_ddf_grid(survey_length=0.01, verbose=False)
        assert result is not None


if __name__ == "__main__":
    unittest.main()
