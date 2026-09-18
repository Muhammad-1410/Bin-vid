import io
import os
import shutil
import subprocess
import tempfile
import unittest
import warnings
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from binvid.probe import VideoInfo, frame_iter, main, probe


class TestProbe(unittest.TestCase):
    temp_dir: str
    video_no_audio: str
    video_with_audio: str

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()

        # 1. Create a 5-frame video without audio using OpenCV
        cls.video_no_audio = os.path.join(cls.temp_dir, "test_no_audio.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(cls.video_no_audio, fourcc, 10.0, (64, 48))
        for i in range(5):
            frame = np.full((48, 64, 3), i * 50, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        # 2. Create a 5-frame video with audio using FFmpeg if ffmpeg is available
        cls.video_with_audio = os.path.join(cls.temp_dir, "test_with_audio.mp4")
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            subprocess.run(
                [
                    ffmpeg_bin,
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=duration=0.5:size=64x48:rate=10",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=1000:duration=0.5",
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "aac",
                    cls.video_with_audio,
                ],
                capture_output=True,
                check=False,
            )

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_probe_no_audio(self):
        info = probe(self.video_no_audio)
        self.assertIsInstance(info, VideoInfo)
        self.assertEqual(info.width, 64)
        self.assertEqual(info.height, 48)
        self.assertAlmostEqual(info.fps, 10.0, places=1)
        self.assertEqual(info.frame_count, 5)
        self.assertAlmostEqual(info.duration_seconds, 0.5, places=1)
        self.assertFalse(info.has_audio)

    def test_probe_with_audio(self):
        if not os.path.isfile(self.video_with_audio):
            self.skipTest("FFmpeg generated audio test video not available")
        info = probe(self.video_with_audio)
        self.assertTrue(info.has_audio)
        self.assertEqual(info.width, 64)
        self.assertEqual(info.height, 48)

    def test_frame_iter(self):
        frames = list(frame_iter(self.video_no_audio))
        self.assertEqual(len(frames), 5)
        for frame in frames:
            self.assertIsInstance(frame, np.ndarray)
            self.assertEqual(frame.shape, (48, 64, 3))

    def test_frame_iter_early_exit_releases_capture(self):
        # Break out of iteration early to ensure cleanup works
        gen = frame_iter(self.video_no_audio)
        first_frame = next(gen)
        self.assertEqual(first_frame.shape, (48, 64, 3))
        gen.close()

    def test_nonexistent_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            probe("non_existent_file_12345.mp4")

        with self.assertRaises(FileNotFoundError):
            list(frame_iter("non_existent_file_12345.mp4"))

    def test_unopenable_file_raises(self):
        corrupt_file = os.path.join(self.temp_dir, "corrupt.mp4")
        with open(corrupt_file, "w") as f:
            f.write("not a valid video")

        with self.assertRaises(ValueError):
            probe(corrupt_file)

        with self.assertRaises(ValueError):
            list(frame_iter(corrupt_file))

    def test_bad_metadata_fps_zero_falls_back(self):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        # width, height, fps, frame_count
        prop_map = {
            cv2.CAP_PROP_FRAME_WIDTH: 100.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 50.0,
            cv2.CAP_PROP_FPS: 0.0,  # bad fps
            cv2.CAP_PROP_FRAME_COUNT: 60.0,
        }
        mock_cap.get.side_effect = lambda prop: prop_map.get(prop, 0.0)

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with warnings.catch_warnings(record=True) as recorded_warnings:
                warnings.simplefilter("always")
                info = probe(self.video_no_audio)

                # Expect warning for FPS
                self.assertTrue(any("FPS" in str(w.message) for w in recorded_warnings))
                self.assertEqual(info.fps, 30.0)
                self.assertEqual(info.frame_count, 60)
                self.assertAlmostEqual(info.duration_seconds, 2.0)

    def test_bad_metadata_frame_count_negative_falls_back(self):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        prop_map = {
            cv2.CAP_PROP_FRAME_WIDTH: 100.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 50.0,
            cv2.CAP_PROP_FPS: 25.0,
            cv2.CAP_PROP_FRAME_COUNT: -1.0,  # bad frame count
        }
        mock_cap.get.side_effect = lambda prop: prop_map.get(prop, 0.0)

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with warnings.catch_warnings(record=True) as recorded_warnings:
                warnings.simplefilter("always")
                info = probe(self.video_no_audio)

                # Expect warning for frame count
                self.assertTrue(any("frame count" in str(w.message) for w in recorded_warnings))
                self.assertEqual(info.frame_count, 0)
                self.assertEqual(info.duration_seconds, 0.0)

    def test_probe_and_frame_iter_rotation(self):
        # When rotation is 90 degrees, dimensions are swapped to display orientation
        with patch("binvid.probe.get_video_rotation", return_value=90):
            info = probe(self.video_no_audio)
            self.assertEqual(info.rotation, 90)
            self.assertEqual(info.width, 48)
            self.assertEqual(info.height, 64)

            # frame_iter rotates frames accordingly
            frame = next(frame_iter(self.video_no_audio))
            self.assertEqual(frame.shape, (64, 48, 3))

    def test_main_cli(self):
        captured = io.StringIO()
        with patch("sys.argv", ["binvid.probe", self.video_no_audio]):
            with patch("sys.stdout", captured):
                main()
        output = captured.getvalue()
        self.assertIn("Probing video:", output)
        self.assertIn("--- VideoInfo ---", output)
        self.assertIn("--- Frame Count Verification ---", output)
        self.assertIn("Match           : YES", output)


if __name__ == "__main__":
    unittest.main()
