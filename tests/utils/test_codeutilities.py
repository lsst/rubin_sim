import unittest

import numpy as np

from rubin_sim.utils.code_utilities import _validate_inputs


class CodeUtilsTest(unittest.TestCase):
    def test_validate_inputs(self):
        """
        test that _validate_inputs returns expected values
        """

        # test that it correctly identifies numpy array inputs
        self.assertTrue(_validate_inputs([np.array([1, 2]), np.array([3, 4])], ["a", "b"], "dummy_method"))

        # test that it correctly identifies inputs that are numbers
        self.assertFalse(_validate_inputs([1, 2, 3], ["a", "b", "c"], "dummy"))

        # test that the correct exception is raised when you pass in
        # a number a numpy array
        with self.assertRaises(RuntimeError) as ee:
            _validate_inputs([1, np.array([2, 3])], ["a", "b"], "dummy")

        self.assertIn("and the same type", ee.exception.args[0])

        # test that the correct exception is raised when you pass in
        # numpy arrays of different length
        with self.assertRaises(RuntimeError) as ee:
            _validate_inputs([np.array([1, 2]), np.array([1, 2, 3])], ["a", "b"], "dummy")

        self.assertIn("same length", ee.exception.args[0])

        # test that an exception is raised if lists (rather than numpy
        # arrays) are passed in
        with self.assertRaises(RuntimeError) as ee:
            _validate_inputs([[1, 2], [3, 4]], ["a", "b"], "dummy")

        self.assertIn("either a number or a numpy array", ee.exception.args[0])


if __name__ == "__main__":
    unittest.main()
