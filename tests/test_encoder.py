import os
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from binvid.encoder import FrameSink
from binvid.environment import find_ffmpeg
from binvid.probe import probe


class TestEncoder(unittest.TestCase):
    ffmpeg_bin: str
    temp_dir: str
    src_with_audio: str
    src_silent: str

    @classmethod
    def setUpClass(cls):
        cls.ffmpeg_bin = find_ffmpeg()
        cls.temp_dir = tempfile.mkdtemp()

        # 1. Generate 1s reference video with audio
        cls.src_with_audio = os.path.join(cls.temp_dir, "ref_with_audio.mp4")
        subprocess.run(
            [
                cls.ffmpeg_bin,
                "-y",
                "-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=30",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                "-c:v", "libx264", "-c:a", "aac",
                cls.src_with_audio,
            ],
            capture_output=True,
            check=True,
        )

        # 2. Generate 1s silent video
        cls.src_silent = os.path.join(cls.temp_dir, "ref_silent.mp4")
        subprocess.run(
            [
                cls.ffmpeg_bin,
                "-y",
                "-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=30",
                "-c:v", "libx264",
                cls.src_silent,
            ],
            capture_output=True,
            check=True,
        )

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_encode_with_audio_preservation(self):
        out_path = os.path.join(self.temp_dir, "out_with_audio.mp4")
        width, height, fps = 160, 120, 30.0
        num_frames = 30

        with FrameSink(self.src_with_audio, out_path, width=width, height=height, fps=fps, crf=18) as sink:
            for i in range(num_frames):
                frame = np.full((height, width, 3), i * 8, dtype=np.uint8)
                sink.write(frame)

        self.assertTrue(os.path.isfile(out_path))
        info = probe(out_path)
        self.assertEqual(info.width, width)
        self.assertEqual(info.height, height)
        self.assertEqual(info.frame_count, num_frames)
        self.assertTrue(info.has_audio)

    def test_encode_silent_video(self):
        out_path = os.path.join(self.temp_dir, "out_silent.mp4")
        width, height, fps = 160, 120, 30.0
        num_frames = 15

        with FrameSink(self.src_silent, out_path, width=width, height=height, fps=fps, crf=18) as sink:
            for i in range(num_frames):
                frame = np.full((height, width, 3), 100, dtype=np.uint8)
                sink.write(frame)

        self.assertTrue(os.path.isfile(out_path))
        info = probe(out_path)
        self.assertEqual(info.width, width)
        self.assertEqual(info.height, height)
        self.assertEqual(info.frame_count, num_frames)
        self.assertFalse(info.has_audio)

    def test_frame_validation(self):
        sink = FrameSink(self.src_silent, "dummy.mp4", width=100, height=80)
        # Not active
        with self.assertRaises(RuntimeError):
            sink.write(np.zeros((80, 100, 3), dtype=np.uint8))

        # Test parameter validations
        with self.assertRaises(ValueError):
            FrameSink(self.src_silent, "dummy.mp4", width=0, height=80)
        with self.assertRaises(ValueError):
            FrameSink(self.src_silent, "dummy.mp4", width=100, height=-10)
        with self.assertRaises(ValueError):
            FrameSink(self.src_silent, "dummy.mp4", width=100, height=80, fps=0)
        with self.assertRaises(ValueError):
            FrameSink(self.src_silent, "dummy.mp4", width=100, height=80, crf=-1)
        with self.assertRaises(ValueError):
            FrameSink(self.src_silent, "dummy.mp4", width=100, height=80, crf=52)

    def test_frame_content_validation_inside_context(self):
        out_path = os.path.join(self.temp_dir, "test_validation.mp4")
        with FrameSink(self.src_silent, out_path, width=160, height=120) as sink:
            # Wrong type
            with self.assertRaises(TypeError):
                sink.write("not an array")  # type: ignore

            # Wrong dtype
            with self.assertRaises(ValueError):
                sink.write(np.zeros((120, 160, 3), dtype=np.float32))

            # Wrong shape
            with self.assertRaises(ValueError):
                sink.write(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_ffmpeg_failure_surfaces_stderr(self):
        # Provoke FFmpeg failure by setting an invalid source path
        bad_out = os.path.join(self.temp_dir, "bad_out.mp4")
        with self.assertRaises(RuntimeError) as ctx:
            with FrameSink("non_existent_source_path_9999.mp4", bad_out, width=160, height=120) as sink:
                sink.write(np.zeros((120, 160, 3), dtype=np.uint8))

        err_msg = str(ctx.exception)
        self.assertIn("FFmpeg", err_msg)

    def test_broken_pipe_surfaces_stderr(self):
        out_path = os.path.join(self.temp_dir, "broken_pipe.mp4")
        with FrameSink(self.src_silent, out_path, width=160, height=120) as sink:
            # Safely close original stdin and replace with mock simulating BrokenPipeError
            if sink._proc.stdin:
                sink._proc.stdin.close()
            mock_stdin = MagicMock()
            mock_stdin.write.side_effect = BrokenPipeError("Simulated pipe break")
            mock_stdin.closed = True
            sink._proc.stdin = mock_stdin

            with self.assertRaises(RuntimeError) as ctx:
                sink.write(np.zeros((120, 160, 3), dtype=np.uint8))

            self.assertIn("Broken pipe", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
