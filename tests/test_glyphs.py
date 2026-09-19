import os
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from binvid.environment import find_monospace_font
from binvid.glyphs import build_atlas, main, preview_atlas

class TestGlyphs(unittest.TestCase):
    font_path: str

    @classmethod
    def setUpClass(cls):
        cls.font_path = find_monospace_font()

    def test_build_atlas_shape_and_dtype(self):
        cell_w, cell_h = (8, 14)
        atlas = build_atlas(self.font_path, font_size=12, cell_w=cell_w, cell_h=cell_h)
        self.assertIsInstance(atlas, np.ndarray)
        self.assertEqual(atlas.shape, (2, cell_h, cell_w))
        self.assertEqual(atlas.dtype, np.uint8)
        self.assertGreaterEqual(int(atlas.min()), 0)
        self.assertLessEqual(int(atlas.max()), 255)

    def test_glyphs_distinct_and_nonempty(self):
        atlas = build_atlas(self.font_path, font_size=12, cell_w=8, cell_h=14)
        glyph_0 = atlas[0]
        glyph_1 = atlas[1]
        self.assertGreater(np.count_nonzero(glyph_0), 0)
        self.assertGreater(np.count_nonzero(glyph_1), 0)
        self.assertFalse(np.array_equal(glyph_0, glyph_1))

    def test_centering_within_cell(self):
        cell_w, cell_h = (10, 18)
        atlas = build_atlas(self.font_path, font_size=12, cell_w=cell_w, cell_h=cell_h)
        for idx, name in enumerate(['0', '1']):
            glyph = atlas[idx]
            y_indices, x_indices = np.where(glyph > 0)
            self.assertTrue(len(y_indices) > 0)
            self.assertTrue(len(x_indices) > 0)
            min_y, max_y = (int(y_indices.min()), int(y_indices.max()))
            min_x, max_x = (int(x_indices.min()), int(x_indices.max()))
            top_margin = min_y
            bottom_margin = cell_h - 1 - max_y
            self.assertLess(abs(top_margin - bottom_margin), 4, f"Vertical alignment skewed for '{name}'")
            left_margin = min_x
            right_margin = cell_w - 1 - max_x
            self.assertLess(abs(left_margin - right_margin), 4, f"Horizontal alignment skewed for '{name}'")

    def test_invalid_parameters_raise(self):
        with self.assertRaises(FileNotFoundError):
            build_atlas('non_existent_font_file.ttf', 12, 8, 14)
        with self.assertRaises(ValueError):
            build_atlas(self.font_path, font_size=0, cell_w=8, cell_h=14)
        with self.assertRaises(ValueError):
            build_atlas(self.font_path, font_size=12, cell_w=0, cell_h=14)
        with self.assertRaises(ValueError):
            build_atlas(self.font_path, font_size=12, cell_w=8, cell_h=-5)

    def test_preview_atlas(self):
        atlas = build_atlas(self.font_path, font_size=12, cell_w=8, cell_h=14)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, 'test_preview.png')
            result_path = preview_atlas(atlas, scale=4, output_path=out_file)
            self.assertEqual(result_path, os.path.abspath(out_file))
            self.assertTrue(os.path.isfile(out_file))
            with Image.open(out_file) as img:
                self.assertEqual(img.size, (8 * 2 * 4, 14 * 4))

    def test_preview_atlas_invalid_input(self):
        with self.assertRaises(ValueError):
            preview_atlas(np.zeros((1, 10, 10), dtype=np.uint8))
        with self.assertRaises(ValueError):
            preview_atlas(np.zeros((2, 10, 10), dtype=np.uint8), scale=0)

    def test_main_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            preview_path = os.path.join(tmpdir, 'atlas_preview.png')
            with patch('binvid.glyphs.preview_atlas', wraps=preview_atlas) as mock_prev:
                main()
                self.assertTrue(mock_prev.called)
if __name__ == '__main__':
    unittest.main()
