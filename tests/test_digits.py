import unittest
import numpy as np
from binvid.digits import DigitField, ScrollField

class TestDigits(unittest.TestCase):

    def test_values_are_only_zero_or_one(self):
        df = DigitField(rows=20, cols=30, seed=42, refresh=4)
        for f in range(10):
            arr = df.for_frame(f)
            self.assertEqual(arr.shape, (20, 30))
            self.assertEqual(arr.dtype, np.uint8)
            unique_vals = set(np.unique(arr))
            self.assertTrue(unique_vals.issubset({0, 1}))
        sf = ScrollField(rows=20, cols=30, seed=42)
        for f in range(10):
            arr = sf.for_frame(f)
            self.assertEqual(arr.shape, (20, 30))
            self.assertEqual(arr.dtype, np.uint8)
            unique_vals = set(np.unique(arr))
            self.assertTrue(unique_vals.issubset({0, 1}))

    def test_same_seed_produces_identical_output(self):
        df1 = DigitField(rows=15, cols=25, seed=123, refresh=5, partial=False)
        df2 = DigitField(rows=15, cols=25, seed=123, refresh=5, partial=False)
        for f in range(20):
            np.testing.assert_array_equal(df1.for_frame(f), df2.for_frame(f))
        df_p1 = DigitField(rows=15, cols=25, seed=999, refresh=3, partial=True)
        df_p2 = DigitField(rows=15, cols=25, seed=999, refresh=3, partial=True)
        for f in range(20):
            np.testing.assert_array_equal(df_p1.for_frame(f), df_p2.for_frame(f))
        sf1 = ScrollField(rows=15, cols=25, seed=777)
        sf2 = ScrollField(rows=15, cols=25, seed=777)
        for f in range(20):
            np.testing.assert_array_equal(sf1.for_frame(f), sf2.for_frame(f))

    def test_different_seeds_produce_different_output(self):
        df1 = DigitField(rows=20, cols=20, seed=1, refresh=0)
        df2 = DigitField(rows=20, cols=20, seed=2, refresh=0)
        self.assertFalse(np.array_equal(df1.for_frame(0), df2.for_frame(0)))
        sf1 = ScrollField(rows=20, cols=20, seed=1)
        sf2 = ScrollField(rows=20, cols=20, seed=2)
        self.assertFalse(np.array_equal(sf1.for_frame(0), sf2.for_frame(0)))

    def test_refresh_zero_returns_identical_object_and_contents(self):
        df = DigitField(rows=10, cols=10, seed=42, refresh=0)
        first_frame = df.for_frame(0)
        for f in [1, 2, 5, 10, 100]:
            next_frame = df.for_frame(f)
            self.assertIs(first_frame, next_frame)
            np.testing.assert_array_equal(first_frame, next_frame)

    def test_refresh_n_behavior(self):
        df = DigitField(rows=20, cols=20, seed=42, refresh=4, partial=False)
        f0 = df.for_frame(0)
        f1 = df.for_frame(1)
        f3 = df.for_frame(3)
        self.assertIs(f0, f1)
        self.assertIs(f0, f3)
        f4 = df.for_frame(4)
        self.assertFalse(np.array_equal(f0, f4))
        f5 = df.for_frame(5)
        self.assertIs(f4, f5)

    def test_partial_refresh_ratio(self):
        rows, cols = (100, 100)
        df = DigitField(rows=rows, cols=cols, seed=42, refresh=5, partial=True, flip_ratio=0.15)
        f0 = df.for_frame(0)
        f5 = df.for_frame(5)
        flipped_count = np.count_nonzero(f0 != f5)
        total_cells = rows * cols
        ratio = flipped_count / total_cells
        self.assertGreaterEqual(ratio, 0.1)
        self.assertLessEqual(ratio, 0.2)

    def test_scrollfield_drifts_across_frames(self):
        sf = ScrollField(rows=20, cols=20, seed=42, min_speed=1.0, max_speed=2.0)
        f0 = sf.for_frame(0)
        f5 = sf.for_frame(5)
        f10 = sf.for_frame(10)
        self.assertFalse(np.array_equal(f0, f5))
        self.assertFalse(np.array_equal(f5, f10))

    def test_interchangeable_interface(self):
        fields = [DigitField(rows=12, cols=16, seed=42, refresh=8), DigitField(rows=12, cols=16, seed=42, refresh=8, partial=True), ScrollField(rows=12, cols=16, seed=42)]
        for field in fields:
            out = field.for_frame(0)
            self.assertEqual(out.shape, (12, 16))
            self.assertEqual(out.dtype, np.uint8)
            self.assertTrue(set(np.unique(out)).issubset({0, 1}))

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            DigitField(rows=0, cols=10)
        with self.assertRaises(ValueError):
            DigitField(rows=10, cols=-1)
        with self.assertRaises(ValueError):
            DigitField(rows=10, cols=10, refresh=-1)
        with self.assertRaises(ValueError):
            DigitField(rows=10, cols=10, flip_ratio=-0.1)
        with self.assertRaises(ValueError):
            DigitField(rows=10, cols=10).for_frame(-1)
        with self.assertRaises(ValueError):
            ScrollField(rows=10, cols=10, min_speed=-1.0)
        with self.assertRaises(ValueError):
            ScrollField(rows=10, cols=10, min_speed=2.0, max_speed=1.0)
        with self.assertRaises(ValueError):
            ScrollField(rows=10, cols=10).for_frame(-1)
if __name__ == '__main__':
    unittest.main()
