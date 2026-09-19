import io
import os
import sys
import unittest
from unittest.mock import patch

from binvid.environment import find_ffmpeg, find_monospace_font, main


class TestEnvironment(unittest.TestCase):
    def test_find_ffmpeg_success(self):
        # On this environment, find_ffmpeg should find a valid executable
        path = find_ffmpeg()
        self.assertIsInstance(path, str)
        self.assertTrue(os.path.isfile(path))

    @patch("shutil.which", return_value=None)
    @patch.dict(sys.modules, {"imageio_ffmpeg": None})
    def test_find_ffmpeg_not_found_raises(self, _mock_which):
        with self.assertRaises(RuntimeError) as ctx:
            find_ffmpeg()
        self.assertIn("FFmpeg binary not found", str(ctx.exception))
        self.assertIn("imageio-ffmpeg", str(ctx.exception))

    @patch("shutil.which", return_value=None)
    def test_find_ffmpeg_fallback_to_imageio(self, _mock_which):
        class DummyImageio:
            @staticmethod
            def get_ffmpeg_exe():
                return __file__  # valid existing file for test

        with patch.dict(sys.modules, {"imageio_ffmpeg": DummyImageio}):
            path = find_ffmpeg()
            self.assertEqual(path, os.path.abspath(__file__))

    def test_find_monospace_font_success(self):
        # On this environment, a valid font should be discovered
        font_path = find_monospace_font()
        self.assertIsInstance(font_path, str)
        self.assertTrue(os.path.isfile(font_path))

    @patch("os.path.isfile", return_value=False)
    @patch.dict(sys.modules, {"matplotlib": None})
    def test_find_monospace_font_not_found_raises(self, _mock_isfile):
        with self.assertRaises(RuntimeError) as ctx:
            find_monospace_font()
        self.assertIn("No monospace font found", str(ctx.exception))
        self.assertIn("Candidate paths searched:", str(ctx.exception))

    def test_find_ffmpeg_frozen(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            bin_dir = Path(tmpdir) / "bin"
            bin_dir.mkdir(parents=True)
            dummy_ffmpeg = bin_dir / "ffmpeg.exe"
            dummy_ffmpeg.write_text("dummy")

            with patch.object(sys, "frozen", True, create=True), \
                 patch.object(sys, "_MEIPASS", tmpdir, create=True):
                path = find_ffmpeg()
                self.assertEqual(path, str(dummy_ffmpeg.resolve()))

    def test_find_monospace_font_frozen(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            fonts_dir = Path(tmpdir) / "fonts"
            fonts_dir.mkdir(parents=True)
            dummy_font = fonts_dir / "consola.ttf"
            dummy_font.write_text("dummy font")

            with patch.object(sys, "frozen", True, create=True), \
                 patch.object(sys, "_MEIPASS", tmpdir, create=True):
                path = find_monospace_font()
                self.assertEqual(path, str(dummy_font.resolve()))

    def test_environment_main_output(self):
        # Verify that main() runs and prints verification headers
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            main()
        output = captured.getvalue()
        self.assertIn("=== binvid Environment Verification ===", output)
        self.assertIn("FFmpeg binary :", output)
        self.assertIn("Monospace font:", output)


if __name__ == "__main__":
    unittest.main()

