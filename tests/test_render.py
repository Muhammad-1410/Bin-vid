import os
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from binvid.digits import DigitField
from binvid.environment import find_monospace_font
from binvid.geometry import GridSpec, compute_grid
from binvid.glyphs import build_atlas
from binvid.render import main, render_frame

class TestRender(unittest.TestCase):
    font_path: str
    grid: GridSpec
    atlas: np.ndarray
    digits: np.ndarray
    test_frame: np.ndarray

    @classmethod
    def setUpClass(cls):
        cls.font_path = find_monospace_font()
        cls.grid = compute_grid(src_w=320, src_h=240, cols=40, cell_w=8, cell_h=14)
        cls.atlas = build_atlas(cls.font_path, font_size=12, cell_w=cls.grid.cell_w, cell_h=cls.grid.cell_h)
        df = DigitField(rows=cls.grid.rows, cols=cls.grid.cols, seed=42, refresh=0)
        cls.digits = df.for_frame(0)
        cls.test_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cls.test_frame[:120, :160] = 255
        cls.test_frame[120:, 160:] = (0, 0, 255)

    def test_render_color_mode(self):
        out = render_frame(self.test_frame, self.grid, self.atlas, self.digits, mode='color')
        self.assertEqual(out.shape, (self.grid.out_h, self.grid.out_w, 3))
        self.assertEqual(out.dtype, np.uint8)
        cell_0 = out[:self.grid.cell_h, :self.grid.cell_w]
        cell_glyph = self.atlas[self.digits[0, 0]]
        self.assertEqual(int(cell_0[cell_glyph == 0].max()), 0)
        self.assertGreater(int(cell_0[cell_glyph > 0].max()), 100)

    def test_render_mono_mode(self):
        tint = (0, 255, 70)
        out = render_frame(self.test_frame, self.grid, self.atlas, self.digits, mode='mono', tint_color=tint)
        self.assertEqual(out.shape, (self.grid.out_h, self.grid.out_w, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(int(out[:, :, 0].max()), 0)
        self.assertGreater(int(out[:, :, 1].max()), 100)

    def test_render_flat_mode(self):
        tint = (0, 255, 70)
        out = render_frame(self.test_frame, self.grid, self.atlas, self.digits, mode='flat', tint_color=tint, threshold=128)
        self.assertEqual(out.shape, (self.grid.out_h, self.grid.out_w, 3))
        self.assertEqual(out.dtype, np.uint8)
        bottom_left = out[12 * self.grid.cell_h:, :15 * self.grid.cell_w]
        self.assertEqual(int(bottom_left.max()), 0)
        cell_0 = out[:self.grid.cell_h, :self.grid.cell_w]
        cell_glyph = self.atlas[self.digits[0, 0]]
        self.assertGreater(int(cell_0[cell_glyph > 0, 1].max()), 200)

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            render_frame(self.test_frame, self.grid, self.atlas, self.digits, mode='unknown_mode')
        with self.assertRaises(ValueError):
            bad_digits = np.zeros((10, 10), dtype=np.uint8)
            render_frame(self.test_frame, self.grid, self.atlas, bad_digits, mode='color')
        with self.assertRaises(ValueError):
            bad_atlas = np.zeros((2, 5, 5), dtype=np.uint8)
            render_frame(self.test_frame, self.grid, bad_atlas, self.digits, mode='color')

    def test_render_invert(self):
        normal = render_frame(self.test_frame, self.grid, self.atlas, self.digits, invert=False)
        inverted = render_frame(self.test_frame, self.grid, self.atlas, self.digits, invert=True)
        cell_0_norm = normal[:self.grid.cell_h, :self.grid.cell_w]
        cell_0_inv = inverted[:self.grid.cell_h, :self.grid.cell_w]
        cell_glyph = self.atlas[self.digits[0, 0]]
        stroke_mask = cell_glyph > 200
        if np.any(stroke_mask):
            self.assertGreater(cell_0_norm[stroke_mask].mean(), cell_0_inv[stroke_mask].mean())

    def test_render_bg_color(self):
        bg = (100, 0, 0)
        out = render_frame(self.test_frame, self.grid, self.atlas, self.digits, bg_color=bg)
        cell_0 = out[:self.grid.cell_h, :self.grid.cell_w]
        cell_glyph = self.atlas[self.digits[0, 0]]
        bg_pixels = cell_0[cell_glyph == 0]
        self.assertAlmostEqual(int(bg_pixels[:, 0].mean()), 100, delta=5)

    def test_render_gamma(self):
        dark_frame = np.full((240, 320, 3), 30, dtype=np.uint8)
        norm_out = render_frame(dark_frame, self.grid, self.atlas, self.digits, gamma=1.0)
        boost_out = render_frame(dark_frame, self.grid, self.atlas, self.digits, gamma=0.5)
        self.assertGreater(boost_out.mean(), norm_out.mean())

    def test_render_contrast_boost(self):
        low_contrast = np.random.randint(50, 80, (240, 320, 3), dtype=np.uint8)
        normal = render_frame(low_contrast, self.grid, self.atlas, self.digits, contrast_boost=False)
        boosted = render_frame(low_contrast, self.grid, self.atlas, self.digits, contrast_boost=True)
        norm_range = normal.max() - normal.min()
        boost_range = boosted.max() - boosted.min()
        self.assertGreaterEqual(boost_range, norm_range)

    def test_render_scanlines(self):
        normal = render_frame(self.test_frame, self.grid, self.atlas, self.digits, scanlines=False)
        scan = render_frame(self.test_frame, self.grid, self.atlas, self.digits, scanlines=True)
        odd_rows_norm = normal[1::2, :].mean()
        odd_rows_scan = scan[1::2, :].mean()
        self.assertLess(odd_rows_scan, odd_rows_norm)

    def test_render_edge_emphasis(self):
        edge_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        edge_frame[:, 160:] = 255
        out = render_frame(edge_frame, self.grid, self.atlas, self.digits, edge_emphasis=True)
        self.assertEqual(out.shape, (self.grid.out_h, self.grid.out_w, 3))

    def test_main_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, 'sample.mp4')
            preview_path = os.path.join(tmpdir, 'frame_preview.png')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, 10.0, (160, 120))
            writer.write(np.full((120, 160, 3), 150, dtype=np.uint8))
            writer.release()
            cli_args = ['binvid.render', video_path, '--cols', '30', '--mode', 'color', '--output', preview_path]
            with patch('sys.argv', cli_args):
                main()
            self.assertTrue(os.path.isfile(preview_path))
            saved_img = cv2.imread(preview_path)
            self.assertIsNotNone(saved_img)
            self.assertEqual(saved_img.shape[2], 3)
if __name__ == '__main__':
    unittest.main()
