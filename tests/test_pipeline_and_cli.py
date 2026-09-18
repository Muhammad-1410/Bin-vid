import os
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from binvid.cli import build_parser, main as cli_main
from binvid.config import RenderConfig
from binvid.environment import find_ffmpeg
from binvid.pipeline import convert
from binvid.probe import probe


class TestPipelineAndCLI(unittest.TestCase):
    ffmpeg_bin: str
    temp_dir: str
    src_video: str

    @classmethod
    def setUpClass(cls):
        cls.ffmpeg_bin = find_ffmpeg()
        cls.temp_dir = tempfile.mkdtemp()

        # Generate a 1-second 30fps video with audio (30 frames)
        cls.src_video = os.path.join(cls.temp_dir, "clip_with_audio.mp4")
        subprocess.run(
            [
                cls.ffmpeg_bin,
                "-y",
                "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=30",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                "-c:v", "libx264", "-c:a", "aac",
                cls.src_video,
            ],
            capture_output=True,
            check=True,
        )

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_convert_end_to_end_color(self):
        out_path = os.path.join(self.temp_dir, "out_color.mp4")
        config = RenderConfig(
            source_path=self.src_video,
            output_path=out_path,
            cols=40,
            cell_w=8,
            cell_h=14,
            mode="color",
            digit_mode="refresh",
            digit_refresh=8,
        )

        convert(config, progress=False)

        self.assertTrue(os.path.isfile(out_path))
        info = probe(out_path)
        self.assertEqual(info.frame_count, 30)
        self.assertTrue(info.has_audio, "Original audio track must be intact")
        self.assertEqual(info.width % 2, 0)
        self.assertEqual(info.height % 2, 0)

    def test_convert_end_to_end_mono_and_scroll(self):
        out_path = os.path.join(self.temp_dir, "out_mono_scroll.mp4")
        config = RenderConfig(
            source_path=self.src_video,
            output_path=out_path,
            cols=35,
            cell_w=8,
            cell_h=14,
            mode="mono",
            digit_mode="scroll",
        )

        convert(config, progress=False)

        self.assertTrue(os.path.isfile(out_path))
        info = probe(out_path)
        self.assertEqual(info.frame_count, 30)
        self.assertTrue(info.has_audio)

    def test_convert_keyboard_interrupt_handles_cleanly(self):
        out_path = os.path.join(self.temp_dir, "out_interrupted.mp4")
        config = RenderConfig(
            source_path=self.src_video,
            output_path=out_path,
            cols=30,
            cell_w=8,
            cell_h=14,
        )

        # Simulate KeyboardInterrupt on the 5th frame
        call_count = 0
        from binvid.encoder import FrameSink
        original_write = FrameSink.write

        def interrupt_write(self, frame):
            nonlocal call_count
            call_count += 1
            if call_count >= 5:
                raise KeyboardInterrupt("Simulated user Ctrl+C")
            return original_write(self, frame)

        with patch.object(FrameSink, "write", interrupt_write):
            convert(config, progress=False)

        # Confirm partial file exists and was closed cleanly
        self.assertTrue(os.path.isfile(out_path))

    def test_convert_with_limit(self):
        out_path = os.path.join(self.temp_dir, "out_limited.mp4")
        config = RenderConfig(
            source_path=self.src_video,
            output_path=out_path,
            cols=30,
            cell_w=8,
            cell_h=14,
            limit=12,
        )
        convert(config, progress=False)
        self.assertTrue(os.path.isfile(out_path))
        info = probe(out_path)
        self.assertEqual(info.frame_count, 12)

    def test_convert_multiprocessing_workers(self):
        out_path = os.path.join(self.temp_dir, "out_mp_workers.mp4")
        config = RenderConfig(
            source_path=self.src_video,
            output_path=out_path,
            cols=30,
            cell_w=8,
            cell_h=14,
            workers=2,
            limit=15,
        )
        convert(config, progress=False)
        self.assertTrue(os.path.isfile(out_path))
        info = probe(out_path)
        self.assertEqual(info.frame_count, 15)
        self.assertTrue(info.has_audio)

    def test_cli_parser_defaults_and_options(self):
        parser = build_parser()
        args = parser.parse_args(["video.mp4"])
        self.assertEqual(args.input, "video.mp4")
        self.assertIsNone(args.output)
        self.assertEqual(args.cols, 200)
        self.assertEqual(args.mode, "color")
        self.assertEqual(args.digit_mode, "refresh")
        self.assertEqual(args.workers, 1)
        self.assertIsNone(args.limit)

        args2 = parser.parse_args([
            "input.mp4",
            "-o", "custom.mp4",
            "--cols", "150",
            "--mode", "mono",
            "--digit-mode", "scroll",
            "--crf", "22",
            "--workers", "4",
            "--limit", "50",
        ])
        self.assertEqual(args2.output, "custom.mp4")
        self.assertEqual(args2.cols, 150)
        self.assertEqual(args2.mode, "mono")
        self.assertEqual(args2.digit_mode, "scroll")
        self.assertEqual(args2.crf, 22)
        self.assertEqual(args2.workers, 4)
        self.assertEqual(args2.limit, 50)

    def test_cli_main_execution_with_workers_and_limit(self):
        cli_out = os.path.join(self.temp_dir, "cli_mp_test_out.mp4")
        argv = [
            self.src_video,
            "-o", cli_out,
            "--cols", "30",
            "--workers", "2",
            "--limit", "10",
            "--no-progress",
        ]
        ret = cli_main(argv)
        self.assertEqual(ret, 0)
        self.assertTrue(os.path.isfile(cli_out))
        info = probe(cli_out)
        self.assertEqual(info.frame_count, 10)

    def test_cli_main_execution(self):
        cli_out = os.path.join(self.temp_dir, "cli_test_out.mp4")
        argv = [
            self.src_video,
            "-o", cli_out,
            "--cols", "30",
            "--mode", "flat",
            "--no-progress",
        ]
        ret = cli_main(argv)
        self.assertEqual(ret, 0)
        self.assertTrue(os.path.isfile(cli_out))
        info = probe(cli_out)
        self.assertEqual(info.frame_count, 30)
        self.assertTrue(info.has_audio)


    def test_permission_error_on_unwritable_destination(self):
        config = RenderConfig(
            source_path=self.src_video,
            output_path="/non_existent_root_dir_xyz/cannot_write/out.mp4",
            cols=30,
        )
        with patch("os.access", return_value=False):
            with self.assertRaises(PermissionError):
                convert(config, progress=False)

    def test_convert_with_styles(self):
        out_path = os.path.join(self.temp_dir, "out_styles.mp4")
        config = RenderConfig(
            source_path=self.src_video,
            output_path=out_path,
            cols=30,
            cell_w=8,
            cell_h=14,
            limit=5,
            invert=True,
            bg="#101820",
            gamma=0.8,
            contrast_boost=True,
            scanlines=True,
            edge_emphasis=True,
        )
        convert(config, progress=False)
        self.assertTrue(os.path.isfile(out_path))
        info = probe(out_path)
        self.assertEqual(info.frame_count, 5)

    def test_cli_friendly_errors(self):
        # Missing file error
        with patch("sys.stderr") as mock_err:
            ret = cli_main(["non_existent_file_xyz.mp4"])
            self.assertEqual(ret, 1)

        # Unwritable directory error
        with patch("os.access", return_value=False):
            with patch("sys.stderr") as mock_err:
                ret = cli_main([self.src_video, "-o", "/unwritable/out.mp4"])
                self.assertEqual(ret, 1)


if __name__ == "__main__":
    unittest.main()
