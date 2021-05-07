import numpy as np
import unittest
from rubin_sim.utils import SpatialBounds, CircleBounds, BoxBounds


class SpatialBoundsTest(unittest.TestCase):

    def testExceptions(self):
        """
        Test that the spatial bound classes raise exceptions when you
        give them improperly formatted arguments
        """

        with self.assertRaises(RuntimeError):
            CircleBounds(1.0, 2.0, [3.0, 4.0])

        with self.assertRaises(RuntimeError):
            CircleBounds('a', 2.0, 3.0)

        with self.assertRaises(RuntimeError):
            CircleBounds(1.0, 'b', 4.0)

        CircleBounds(1.0, 2.0, 3)

        with self.assertRaises(RuntimeError):
            BoxBounds(1.0, 2.0, 'abcde')

        with self.assertRaises(RuntimeError):
            BoxBounds('a', 2, 3.0)

        with self.assertRaises(RuntimeError):
            BoxBounds(1.0, 'b', 4.0)

        BoxBounds(1, 2, 3)
        BoxBounds(1, 2, [3, 5])

    def testCircle(self):
        myFov = SpatialBounds.getSpatialBounds('circle', 1.0, 2.0, 1.0)
        self.assertEqual(myFov.RA, 1.0)
        self.assertEqual(myFov.DEC, 2.0)
        self.assertEqual(myFov.radius, 1.0)

    def testSquare(self):
        myFov1 = SpatialBounds.getSpatialBounds('box', 1.0, 2.0, 1.0)
        self.assertEqual(myFov1.RA, 1.0)
        self.assertEqual(myFov1.DEC, 2.0)
        self.assertEqual(myFov1.RAmaxDeg, np.degrees(2.0))
        self.assertEqual(myFov1.RAminDeg, np.degrees(0.0))
        self.assertEqual(myFov1.DECmaxDeg, np.degrees(3.0))
        self.assertEqual(myFov1.DECminDeg, np.degrees(1.0))

        length = [1.0]
        myFov2 = SpatialBounds.getSpatialBounds('box', 1.0, 2.0, length)
        self.assertEqual(myFov2.RA, 1.0)
        self.assertEqual(myFov2.DEC, 2.0)
        self.assertEqual(myFov2.RAmaxDeg, np.degrees(2.0))
        self.assertEqual(myFov2.RAminDeg, np.degrees(0.0))
        self.assertEqual(myFov2.DECmaxDeg, np.degrees(3.0))
        self.assertEqual(myFov2.DECminDeg, np.degrees(1.0))

        length = (1.0)
        myFov3 = SpatialBounds.getSpatialBounds('box', 1.0, 2.0, length)
        self.assertEqual(myFov3.RA, 1.0)
        self.assertEqual(myFov3.DEC, 2.0)
        self.assertEqual(myFov3.RAmaxDeg, np.degrees(2.0))
        self.assertEqual(myFov3.RAminDeg, np.degrees(0.0))
        self.assertEqual(myFov3.DECmaxDeg, np.degrees(3.0))
        self.assertEqual(myFov3.DECminDeg, np.degrees(1.0))

        length = np.array([1.0])
        myFov4 = SpatialBounds.getSpatialBounds('box', 1.0, 2.0, length)
        self.assertEqual(myFov4.RA, 1.0)
        self.assertEqual(myFov4.DEC, 2.0)
        self.assertEqual(myFov4.RAmaxDeg, np.degrees(2.0))
        self.assertEqual(myFov4.RAminDeg, np.degrees(0.0))
        self.assertEqual(myFov4.DECmaxDeg, np.degrees(3.0))
        self.assertEqual(myFov4.DECminDeg, np.degrees(1.0))

        self.assertRaises(RuntimeError, SpatialBounds.getSpatialBounds,
                          'utterNonsense', 1.0, 2.0, length)

    def testRectangle(self):

        length = [1.0, 2.0]
        myFov2 = SpatialBounds.getSpatialBounds('box', 1.0, 2.0, length)
        self.assertEqual(myFov2.RA, 1.0)
        self.assertEqual(myFov2.DEC, 2.0)
        self.assertEqual(myFov2.RAmaxDeg, np.degrees(2.0))
        self.assertEqual(myFov2.RAminDeg, np.degrees(0.0))
        self.assertEqual(myFov2.DECmaxDeg, np.degrees(4.0))
        self.assertEqual(myFov2.DECminDeg, np.degrees(0.0))

        length = (1.0, 2.0)
        myFov3 = SpatialBounds.getSpatialBounds('box', 1.0, 2.0, length)
        self.assertEqual(myFov3.RA, 1.0)
        self.assertEqual(myFov3.DEC, 2.0)
        self.assertEqual(myFov3.RAmaxDeg, np.degrees(2.0))
        self.assertEqual(myFov3.RAminDeg, np.degrees(0.0))
        self.assertEqual(myFov3.DECmaxDeg, np.degrees(4.0))
        self.assertEqual(myFov3.DECminDeg, np.degrees(0.0))

        length = np.array([1.0, 2.0])
        myFov4 = SpatialBounds.getSpatialBounds('box', 1.0, 2.0, length)
        self.assertEqual(myFov4.RA, 1.0)
        self.assertEqual(myFov4.DEC, 2.0)
        self.assertEqual(myFov4.RAmaxDeg, np.degrees(2.0))
        self.assertEqual(myFov4.RAminDeg, np.degrees(0.0))
        self.assertEqual(myFov4.DECmaxDeg, np.degrees(4.0))
        self.assertEqual(myFov4.DECminDeg, np.degrees(0.0))

        self.assertRaises(RuntimeError, SpatialBounds.getSpatialBounds,
                          'box', 1.0, 2.0, 'moreUtterNonsense')

    def test_eq(self):
        """
        Test that we have implemented __eq__and __ne__ correctly
        """
        ref_circle = CircleBounds(113.1, -20.1, 1.56)
        other_circle = CircleBounds(113.1, -20.1, 1.56)
        self.assertEqual(ref_circle, other_circle)
        self.assertTrue(ref_circle == other_circle)
        self.assertFalse(ref_circle != other_circle)

        other_circle = CircleBounds(113.2, -20.1, 1.56)
        self.assertNotEqual(ref_circle, other_circle)
        self.assertFalse(ref_circle == other_circle)
        self.assertTrue(ref_circle != other_circle)

        other_circle = CircleBounds(113.1, -20.2, 1.56)
        self.assertNotEqual(ref_circle, other_circle)
        self.assertFalse(ref_circle == other_circle)
        self.assertTrue(ref_circle != other_circle)

        other_circle = CircleBounds(113.1, -20.1, 1.57)
        self.assertNotEqual(ref_circle, other_circle)
        self.assertFalse(ref_circle == other_circle)
        self.assertTrue(ref_circle != other_circle)

        ref_square = BoxBounds(113.1, -20.1, 1.56)
        self.assertNotEqual(ref_circle, ref_square)
        self.assertFalse(ref_circle == ref_square)
        self.assertTrue(ref_circle != ref_square)

        other_square = BoxBounds(113.1, -20.1, 1.56)
        self.assertEqual(ref_square, other_square)
        self.assertTrue(ref_square == other_square)
        self.assertFalse(ref_square != other_square)

        other_square = BoxBounds(113.2, -20.1, 1.56)
        self.assertNotEqual(ref_square, other_square)
        self.assertFalse(ref_square == other_square)
        self.assertTrue(ref_square != other_square)

        other_square = BoxBounds(113.1, -20.2, 1.56)
        self.assertNotEqual(ref_square, other_square)
        self.assertFalse(ref_square == other_square)
        self.assertTrue(ref_square != other_square)

        other_square = BoxBounds(113.1, -20.1, 1.57)
        self.assertNotEqual(ref_square, other_square)
        self.assertFalse(ref_square == other_square)
        self.assertTrue(ref_square != other_square)

        ref_rect = BoxBounds(113.1, -20.1, [1.56, 1.56])
        self.assertEqual(ref_rect, ref_square)
        self.assertTrue(ref_rect == ref_square)
        self.assertFalse(ref_rect != ref_square)

        self.assertNotEqual(ref_rect, ref_circle)
        self.assertFalse(ref_rect == ref_circle)
        self.assertTrue(ref_rect != ref_circle)

        ref_rect = BoxBounds(113.1, -20.1, [1.56, 1.52])
        self.assertNotEqual(ref_rect, ref_square)
        self.assertFalse(ref_rect == ref_square)
        self.assertTrue(ref_rect != ref_square)

        other_rect = BoxBounds(113.1, -20.1, [1.56, 1.52])
        self.assertEqual(ref_rect, other_rect)
        self.assertTrue(ref_rect == other_rect)
        self.assertFalse(ref_rect != other_rect)

        other_rect = BoxBounds(113.1, -20.1, np.array([1.56, 1.52]))
        self.assertEqual(ref_rect, other_rect)
        self.assertTrue(ref_rect == other_rect)
        self.assertFalse(ref_rect != other_rect)

        other_rect = BoxBounds(113.1, -20.1, (1.56, 1.52))
        self.assertEqual(ref_rect, other_rect)
        self.assertTrue(ref_rect == other_rect)
        self.assertFalse(ref_rect != other_rect)

        other_rect = BoxBounds(113.2, -20.1, (1.56, 1.52))
        self.assertNotEqual(ref_rect, other_rect)
        self.assertFalse(ref_rect == other_rect)
        self.assertTrue(ref_rect != other_rect)

        other_rect = BoxBounds(113.1, -20.2, (1.56, 1.52))
        self.assertNotEqual(ref_rect, other_rect)
        self.assertFalse(ref_rect == other_rect)
        self.assertTrue(ref_rect != other_rect)

        other_rect = BoxBounds(113.1, -20.1, (1.57, 1.52))
        self.assertNotEqual(ref_rect, other_rect)
        self.assertFalse(ref_rect == other_rect)
        self.assertTrue(ref_rect != other_rect)

        other_rect = BoxBounds(113.1, -20.1, (1.56, 1.51))
        self.assertNotEqual(ref_rect, other_rect)
        self.assertFalse(ref_rect == other_rect)
        self.assertTrue(ref_rect != other_rect)


if __name__ == "__main__":
    unittest.main()
