import unittest


class CleanUpTestCase(unittest.TestCase):
    def test_clean_up(self):
        """
        Test that sims_clean_up behaves as it should by importing a test module
        with some dummy caches, adding things to them, and then deleting them.
        """
        from testModules.dummyModule import a_dict_cache
        from testModules.dummyModule import a_list_cache
        from rubin_sim.utils.CodeUtilities import sims_clean_up

        self.assertEqual(len(sims_clean_up.targets), 2)

        a_dict_cache["a"] = 1
        a_dict_cache["b"] = 2
        a_list_cache.append("alpha")
        a_list_cache.append("beta")

        self.assertEqual(len(a_dict_cache), 2)
        self.assertEqual(len(a_list_cache), 2)

        sims_clean_up()

        self.assertEqual(len(a_dict_cache), 0)
        self.assertEqual(len(a_list_cache), 0)
        self.assertEqual(len(sims_clean_up.targets), 2)

        # make sure that re-importing caches does not add second copies
        # to sims_clean_up.targets
        from testModules.dummyModule import a_list_cache

        self.assertEqual(len(sims_clean_up.targets), 2)


if __name__ == "__main__":
    unittest.main()
