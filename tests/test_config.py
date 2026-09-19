import os
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from binvid.config import RenderConfig

class TestRenderConfig(unittest.TestCase):

    def test_default_values(self):
        cfg = RenderConfig(source_path='input.mp4', output_path='output.mp4')
        self.assertEqual(cfg.source_path, 'input.mp4')
        self.assertEqual(cfg.output_path, 'output.mp4')
        self.assertEqual(cfg.cols, 200)
        self.assertEqual(cfg.cell_w, 8)
        self.assertEqual(cfg.cell_h, 14)
        self.assertIsNone(cfg.font_path)
        self.assertEqual(cfg.font_size, 12)
        self.assertEqual(cfg.digit_refresh, 8)
        self.assertEqual(cfg.seed, 42)
        self.assertEqual(cfg.crf, 18)

    def test_frozen_immutability(self):
        cfg = RenderConfig(source_path='in.mp4', output_path='out.mp4')
        with self.assertRaises(FrozenInstanceError):
            cfg.cols = 100

    def test_valid_config_passes(self):
        cfg = RenderConfig(source_path='valid_input.mp4', output_path='valid_output.mp4')
        cfg.validate()

    def test_empty_source_path_raises(self):
        with self.assertRaises(ValueError) as ctx:
            RenderConfig(source_path='', output_path='out.mp4').validate()
        self.assertIn('source_path', str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            RenderConfig(source_path='   ', output_path='out.mp4').validate()
        self.assertIn('source_path', str(ctx.exception))

    def test_empty_output_path_raises(self):
        with self.assertRaises(ValueError) as ctx:
            RenderConfig(source_path='in.mp4', output_path='').validate()
        self.assertIn('output_path', str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            RenderConfig(source_path='in.mp4', output_path='   ').validate()
        self.assertIn('output_path', str(ctx.exception))

    def test_cols_validation(self):
        with self.assertRaises(ValueError) as ctx:
            RenderConfig(source_path='in.mp4', output_path='out.mp4', cols=9).validate()
        self.assertIn('cols', str(ctx.exception))
        RenderConfig(source_path='in.mp4', output_path='out.mp4', cols=10).validate()

    def test_cell_dimensions_validation(self):
        with self.assertRaises(ValueError) as ctx:
            RenderConfig(source_path='in.mp4', output_path='out.mp4', cell_w=3).validate()
        self.assertIn('cell_w', str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            RenderConfig(source_path='in.mp4', output_path='out.mp4', cell_h=3).validate()
        self.assertIn('cell_h', str(ctx.exception))
        RenderConfig(source_path='in.mp4', output_path='out.mp4', cell_w=4, cell_h=4).validate()

    def test_font_size_validation(self):
        with self.assertRaises(ValueError) as ctx:
            RenderConfig(source_path='in.mp4', output_path='out.mp4', font_size=0).validate()
        self.assertIn('font_size', str(ctx.exception))

    def test_digit_refresh_validation(self):
        with self.assertRaises(ValueError) as ctx:
            RenderConfig(source_path='in.mp4', output_path='out.mp4', digit_refresh=-1).validate()
        self.assertIn('digit_refresh', str(ctx.exception))
        RenderConfig(source_path='in.mp4', output_path='out.mp4', digit_refresh=0).validate()

    def test_crf_validation(self):
        with self.assertRaises(ValueError) as ctx:
            RenderConfig(source_path='in.mp4', output_path='out.mp4', crf=-1).validate()
        self.assertIn('crf', str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            RenderConfig(source_path='in.mp4', output_path='out.mp4', crf=52).validate()
        self.assertIn('crf', str(ctx.exception))
        RenderConfig(source_path='in.mp4', output_path='out.mp4', crf=0).validate()
        RenderConfig(source_path='in.mp4', output_path='out.mp4', crf=51).validate()

    def test_font_path_validation(self):
        with self.assertRaises(ValueError) as ctx:
            RenderConfig(source_path='in.mp4', output_path='out.mp4', font_path='non_existent_font_12345.ttf').validate()
        self.assertIn('font_path', str(ctx.exception))
        with tempfile.NamedTemporaryFile(suffix='.ttf', delete=False) as f:
            temp_path = f.name
        try:
            cfg = RenderConfig(source_path='in.mp4', output_path='out.mp4', font_path=temp_path)
            cfg.validate()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_workers_validation(self):
        with self.assertRaises(ValueError) as ctx:
            RenderConfig(source_path='in.mp4', output_path='out.mp4', workers=0).validate()
        self.assertIn('workers', str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            RenderConfig(source_path='in.mp4', output_path='out.mp4', workers=-2).validate()
        self.assertIn('workers', str(ctx.exception))
        RenderConfig(source_path='in.mp4', output_path='out.mp4', workers=1).validate()
        RenderConfig(source_path='in.mp4', output_path='out.mp4', workers=8).validate()

    def test_limit_validation(self):
        with self.assertRaises(ValueError) as ctx:
            RenderConfig(source_path='in.mp4', output_path='out.mp4', limit=0).validate()
        self.assertIn('limit', str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            RenderConfig(source_path='in.mp4', output_path='out.mp4', limit=-5).validate()
        self.assertIn('limit', str(ctx.exception))
        RenderConfig(source_path='in.mp4', output_path='out.mp4', limit=None).validate()
        RenderConfig(source_path='in.mp4', output_path='out.mp4', limit=100).validate()

    def test_gamma_validation(self):
        with self.assertRaises(ValueError) as ctx:
            RenderConfig(source_path='in.mp4', output_path='out.mp4', gamma=0).validate()
        self.assertIn('gamma', str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            RenderConfig(source_path='in.mp4', output_path='out.mp4', gamma=-1.2).validate()
        self.assertIn('gamma', str(ctx.exception))
        RenderConfig(source_path='in.mp4', output_path='out.mp4', gamma=0.7).validate()

    def test_bg_color_validation_and_parsing(self):
        cfg1 = RenderConfig(source_path='in.mp4', output_path='out.mp4', bg='#ffffff')
        cfg1.validate()
        self.assertEqual(cfg1.get_bg_bgr(), (255, 255, 255))
        cfg2 = RenderConfig(source_path='in.mp4', output_path='out.mp4', bg='102030')
        cfg2.validate()
        self.assertEqual(cfg2.get_bg_bgr(), (48, 32, 16))
        cfg3 = RenderConfig(source_path='in.mp4', output_path='out.mp4', bg='255, 128, 0')
        cfg3.validate()
        self.assertEqual(cfg3.get_bg_bgr(), (0, 128, 255))
        with self.assertRaises(ValueError):
            RenderConfig(source_path='in.mp4', output_path='out.mp4', bg='invalid').validate()
        with self.assertRaises(ValueError):
            RenderConfig(source_path='in.mp4', output_path='out.mp4', bg='300,0,0').validate()
if __name__ == '__main__':
    unittest.main()
