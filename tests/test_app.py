from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import MagicMock

import cv2
import gradio as gr
import numpy as np

from binvid.app import (
    _PREVIEW_FRAME_CACHE,
    build_app,
    cleanup_temp_files,
    convert_video_gradio,
    extract_preview_frame,
    render_preview_still,
    resolve_video_path,
    run_conversion,
)


class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp(prefix="binvid_app_test_")
        cls.sample_video = os.path.join(cls.test_dir, "sample.mp4")

        # Create a synthetic 20-frame video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(cls.sample_video, fourcc, 25.0, (160, 120))
        for i in range(20):
            frame = np.full((120, 160, 3), fill_value=int(10 + i * 10), dtype=np.uint8)
            writer.write(frame)
        writer.release()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def setUp(self):
        _PREVIEW_FRAME_CACHE.clear()

    def test_cleanup_temp_files(self):
        """Test temp directory file cleanup by age and file count limit."""
        temp_dir = tempfile.mkdtemp(prefix="cleanup_test_")
        try:
            now = time.time()
            # File 1: Old video (2 hours ago)
            old_file = os.path.join(temp_dir, "old_video.mp4")
            with open(old_file, "w") as f:
                f.write("test")
            os.utime(old_file, (now - 7200, now - 7200))

            # File 2: Recent video (1 minute ago)
            recent_file1 = os.path.join(temp_dir, "recent1.mp4")
            with open(recent_file1, "w") as f:
                f.write("test")
            os.utime(recent_file1, (now - 60, now - 60))

            # File 3: Recent video (now)
            recent_file2 = os.path.join(temp_dir, "recent2.mp4")
            with open(recent_file2, "w") as f:
                f.write("test")

            # File 4: Non-video file (should be ignored)
            non_video = os.path.join(temp_dir, "document.txt")
            with open(non_video, "w") as f:
                f.write("text")
            os.utime(non_video, (now - 7200, now - 7200))

            # Run cleanup with max_age = 3600 (1 hour) and max_files = 5
            deleted = cleanup_temp_files(temp_dir, max_age_seconds=3600, max_files=5)
            self.assertEqual(deleted, 1)
            self.assertFalse(os.path.exists(old_file))
            self.assertTrue(os.path.exists(recent_file1))
            self.assertTrue(os.path.exists(recent_file2))
            self.assertTrue(os.path.exists(non_video))

            # Test max_files truncation
            deleted_excess = cleanup_temp_files(temp_dir, max_age_seconds=86400, max_files=1)
            self.assertEqual(deleted_excess, 1)
            self.assertFalse(os.path.exists(recent_file1))
            self.assertTrue(os.path.exists(recent_file2))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_extract_preview_frame(self):
        """Test extraction and caching of the 25% frame."""
        res = extract_preview_frame(self.sample_video)
        self.assertIsNotNone(res)
        frame, w, h = res
        self.assertEqual(w, 160)
        self.assertEqual(h, 120)
        self.assertEqual(frame.shape, (120, 160, 3))

        # Check caching
        self.assertIn(self.sample_video, _PREVIEW_FRAME_CACHE)
        # Call again, should use cache
        res2 = extract_preview_frame(self.sample_video)
        self.assertIs(res[0], res2[0])

        # Test non-existent file
        self.assertIsNone(extract_preview_frame("non_existent_file.mp4"))

    def test_render_preview_still_color(self):
        """Test rendering live preview still in color mode."""
        img_rgb, info_text = render_preview_still(
            video_path=self.sample_video,
            cols=30,
            font_size=12,
            gamma=1.0,
            mode="color",
            digit_mode="refresh",
            tint_color="#00FF46",
            invert=False,
            bg_color="#000000",
            contrast_boost=False,
            scanlines=False,
            edge_emphasis=False,
        )
        self.assertIsNotNone(img_rgb)
        self.assertIsInstance(img_rgb, np.ndarray)
        self.assertEqual(img_rgb.ndim, 3)
        self.assertEqual(img_rgb.shape[2], 3)
        self.assertIn("30 cols", info_text)
        self.assertIn("Output Resolution", info_text)

    def test_render_preview_still_styles(self):
        """Test rendering live preview still with styles (mono, invert, scanlines, etc.)."""
        img_rgb, info_text = render_preview_still(
            video_path=self.sample_video,
            cols=25,
            font_size=10,
            gamma=0.7,
            mode="mono",
            digit_mode="scroll",
            tint_color="#00FF00",
            invert=True,
            bg_color="#101020",
            contrast_boost=True,
            scanlines=True,
            edge_emphasis=True,
        )
        self.assertIsNotNone(img_rgb)
        self.assertIsInstance(img_rgb, np.ndarray)
        self.assertEqual(img_rgb.shape[2], 3)
        self.assertIn("25 cols", info_text)

    def test_render_preview_still_no_video(self):
        """Test preview handling when no video is provided."""
        img, info = render_preview_still(
            video_path=None,
            cols=30,
            font_size=12,
            gamma=1.0,
            mode="color",
            digit_mode="refresh",
            tint_color="#00FF46",
            invert=False,
            bg_color="#000000",
            contrast_boost=False,
            scanlines=False,
            edge_emphasis=False,
        )
        self.assertIsNone(img)
        self.assertIn("Upload a video", info)

    def test_convert_video_gradio(self):
        """Test conversion flow invoked through the Gradio handler."""
        mock_progress = MagicMock()

        out_path, summary_md, download_btn = run_conversion(
            video_path=self.sample_video,
            cols=30,
            font_size=12,
            gamma=1.0,
            mode="color",
            digit_mode="static",
            tint_color="#00FF46",
            invert=False,
            bg_color="#000000",
            contrast_boost=False,
            scanlines=False,
            edge_emphasis=False,
            refresh=8,
            crf=23,
            workers=1,
            progress=mock_progress,
        )

        self.assertTrue(os.path.isfile(out_path))
        self.assertGreater(os.path.getsize(out_path), 0)
        self.assertTrue(mock_progress.called)
        self.assertIn("Conversion Complete", summary_md)
        self.assertIsInstance(download_btn, gr.DownloadButton)
        if isinstance(download_btn.value, dict):
            self.assertEqual(download_btn.value.get("orig_name"), os.path.basename(out_path))
        else:
            self.assertEqual(download_btn.value, str(out_path))
        self.assertTrue(download_btn.visible)

    def test_convert_video_gradio_missing_video(self):
        """Test that missing video raises gr.Error."""
        with self.assertRaises(gr.Error):
            convert_video_gradio(
                video_path=None,
                cols=30,
                font_size=12,
                gamma=1.0,
                mode="color",
                digit_mode="static",
                tint_color="#00FF46",
                invert=False,
                bg_color="#000000",
                contrast_boost=False,
                scanlines=False,
                edge_emphasis=False,
                refresh=8,
                crf=23,
                workers=1,
            )

    def test_resolve_video_path(self):
        """Test robust resolution of file paths from string, dict, or object."""
        self.assertEqual(resolve_video_path(self.sample_video), os.path.abspath(self.sample_video))
        self.assertEqual(resolve_video_path({"path": self.sample_video}), os.path.abspath(self.sample_video))
        self.assertEqual(resolve_video_path({"video": self.sample_video}), os.path.abspath(self.sample_video))

        class MockFile:
            def __init__(self, path):
                self.path = path

        self.assertEqual(resolve_video_path(MockFile(self.sample_video)), os.path.abspath(self.sample_video))
        self.assertIsNone(resolve_video_path(None))
        self.assertIsNone(resolve_video_path({}))
        self.assertIsNone(resolve_video_path("non_existent.mp4"))

    def test_convert_video_gradio_dict_payload(self):
        """Test conversion flow when Gradio delivers a dict payload."""
        mock_progress = MagicMock()
        out_path, summary_md, download_btn = run_conversion(
            video_path={"path": self.sample_video},
            cols=30,
            font_size=12,
            gamma=1.0,
            mode="color",
            digit_mode="static",
            tint_color="#00FF46",
            invert=False,
            bg_color="#000000",
            contrast_boost=False,
            scanlines=False,
            edge_emphasis=False,
            refresh=8,
            crf=23,
            workers=1,
            progress=mock_progress,
        )
        self.assertTrue(os.path.isfile(out_path))
        self.assertIn("Conversion Complete", summary_md)

    def test_build_app(self):
        """Test that build_app successfully returns a valid Gradio Blocks instance."""
        demo = build_app()
        self.assertIsInstance(demo, gr.Blocks)


if __name__ == "__main__":
    unittest.main()
