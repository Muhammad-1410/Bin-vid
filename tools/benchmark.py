#!/usr/bin/env python3
"""Stage-by-stage benchmarking and optimization analysis tool for binvid.

Times each pipeline stage over 200 frames:
- Decode: VideoCapture / frame_iter
- Downsample: cv2.resize with INTER_AREA
- Mask construction: atlas[digits] fancy indexing and reshape
- Colourise & Multiply:
    * cv2.INTER_NEAREST vs np.repeat comparison
    * In-place pre-allocated buffers vs new array allocation
- Encoder write: streaming to FFmpeg via FrameSink pipe

Also benchmarks end-to-end FPS across different worker counts (1, 2, 4, 8)
and cv2.setNumThreads settings.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

import cv2
import numpy as np

# Ensure binvid is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from binvid.config import RenderConfig
from binvid.digits import DigitField
from binvid.encoder import FrameSink
from binvid.environment import find_ffmpeg, find_monospace_font
from binvid.geometry import compute_grid
from binvid.glyphs import build_atlas
from binvid.pipeline import convert
from binvid.probe import frame_iter, probe
from binvid.render import render_frame


def generate_synthetic_video(path: str, num_frames: int = 200, width: int = 1920, height: int = 1080) -> None:
    """Generate a synthetic 1080p test video with moving gradients and geometric shapes."""
    print(f"Generating synthetic 1080p test video ({num_frames} frames, {width}x{height})...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open cv2.VideoWriter for {path}")

    try:
        # Precompute gradient background
        x = np.linspace(0, 255, width, dtype=np.uint8)
        y = np.linspace(0, 255, height, dtype=np.uint8)
        xx, yy = np.meshgrid(x, y)
        base = np.stack([xx, yy, 255 - xx], axis=-1)

        for i in range(num_frames):
            frame = base.copy()
            # Moving circle
            cx = int((width / 2) + np.sin(i * 0.1) * (width * 0.3))
            cy = int((height / 2) + np.cos(i * 0.1) * (height * 0.3))
            cv2.circle(frame, (cx, cy), 120, (255, 255, 255), -1)
            # Frame counter text
            cv2.putText(
                frame,
                f"Benchmark Frame {i:03d}",
                (80, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.5,
                (0, 0, 255),
                4,
            )
            writer.write(frame)
    finally:
        writer.release()
    print(f"Synthetic video generated at: {path}")


def benchmark_stages(
    video_path: str,
    cols: int = 200,
    cell_w: int = 8,
    cell_h: int = 14,
    num_frames: int = 200,
) -> dict[str, float]:
    """Time each stage separately over 200 frames and print percentage breakdown."""
    info = probe(video_path)
    grid = compute_grid(info.width, info.height, cols, cell_w, cell_h)
    font_path = find_monospace_font()
    atlas = build_atlas(font_path, font_size=12, cell_w=grid.cell_w, cell_h=grid.cell_h)
    field = DigitField(grid.rows, grid.cols, seed=42, refresh=8)

    print("\n" + "=" * 75)
    print(f"  STAGE-BY-STAGE BENCHMARK OVER {num_frames} FRAMES")
    print(f"  Resolution: {info.width}x{info.height} -> Output: {grid.out_w}x{grid.out_h}")
    print(f"  Grid: {grid.cols} cols x {grid.rows} rows (cells: {grid.cell_w}x{grid.cell_h} px)")
    print("=" * 75)

    # 1. Decode Stage
    frames: list[np.ndarray] = []
    t0 = time.perf_counter()
    for idx, f in enumerate(frame_iter(video_path)):
        if idx >= num_frames:
            break
        frames.append(f)
    t_decode = time.perf_counter() - t0
    actual_frames = len(frames)
    if actual_frames == 0:
        raise RuntimeError("No frames could be decoded from the video.")

    # 2. Downsample Stage (cv2.INTER_AREA)
    small_frames: list[np.ndarray] = []
    t0 = time.perf_counter()
    for f in frames:
        s = cv2.resize(f, (grid.cols, grid.rows), interpolation=cv2.INTER_AREA)
        small_frames.append(s)
    t_downsample = time.perf_counter() - t0

    # 3. Mask Construction Stage (fancy indexing + transpose + reshape)
    masks: list[np.ndarray] = []
    digit_fields = [field.for_frame(i) for i in range(actual_frames)]
    t0 = time.perf_counter()
    for digits in digit_fields:
        tiles = atlas[digits]
        m = tiles.transpose(0, 2, 1, 3).reshape(grid.out_h, grid.out_w)
        masks.append(m)
    t_mask = time.perf_counter() - t0

    # 4. Colourise / Upscale Stage: INTER_NEAREST vs np.repeat comparison
    t0 = time.perf_counter()
    upscaled_nearest: list[np.ndarray] = []
    for s in small_frames:
        u = cv2.resize(s, (grid.out_w, grid.out_h), interpolation=cv2.INTER_NEAREST)
        upscaled_nearest.append(u)
    t_upscale_nearest = time.perf_counter() - t0

    t0 = time.perf_counter()
    for s in small_frames:
        _ = np.repeat(np.repeat(s, grid.cell_h, axis=0), grid.cell_w, axis=1)
    t_upscale_repeat = time.perf_counter() - t0

    # Multiply & Combine: Allocating new array per frame
    t0 = time.perf_counter()
    for m, u in zip(masks, upscaled_nearest):
        _ = ((m[..., None].astype(np.uint16) * u) // 255).astype(np.uint8)
    t_multiply_allocated = time.perf_counter() - t0

    # Multiply & Combine: In-place pre-allocated buffer
    out_buf = np.empty((grid.out_h, grid.out_w, 3), dtype=np.uint8)
    temp_u16_buf = np.empty((grid.out_h, grid.out_w, 3), dtype=np.uint16)
    t0 = time.perf_counter()
    rendered_frames: list[np.ndarray] = []
    for m, u in zip(masks, upscaled_nearest):
        np.multiply(m[..., None], u, out=temp_u16_buf, dtype=np.uint16)
        np.floor_divide(temp_u16_buf, 255, out=temp_u16_buf)
        out_buf[...] = temp_u16_buf
        rendered_frames.append(out_buf.copy())
    t_multiply_inplace = time.perf_counter() - t0

    # Total colourise stage (upscale + multiply) with pre-allocation
    t_colourise_opt = t_upscale_nearest + t_multiply_inplace
    t_colourise_baseline = t_upscale_nearest + t_multiply_allocated

    # 5. Encoder Write Stage (pipe to FrameSink)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_out:
        tmp_out_path = tmp_out.name

    try:
        ffmpeg_bin = find_ffmpeg()
        sink = FrameSink(
            source_path=video_path,
            output_path=tmp_out_path,
            width=grid.out_w,
            height=grid.out_h,
            fps=info.fps,
            crf=18,
            ffmpeg_path=ffmpeg_bin,
        )
        t0 = time.perf_counter()
        with sink:
            for rf in rendered_frames:
                sink.write(rf)
        t_encoder = time.perf_counter() - t0
    finally:
        if os.path.exists(tmp_out_path):
            os.remove(tmp_out_path)

    # Calculate percentages for the optimized single-process pipeline
    total_time = t_decode + t_downsample + t_mask + t_colourise_opt + t_encoder
    stages = [
        ("1. Video Decode", t_decode),
        ("2. Downsample (cv2.INTER_AREA)", t_downsample),
        ("3. Mask Construction (atlas[digits])", t_mask),
        ("4. Colourise & Multiply (in-place)", t_colourise_opt),
        ("5. Encoder Write (FFmpeg pipe)", t_encoder),
    ]

    print(f"\n{'Stage':<38} | {'Total Time':>10} | {'Per Frame':>11} | {'% of Total':>10}")
    print("-" * 77)
    for name, dur in stages:
        pct = (dur / total_time) * 100.0 if total_time > 0 else 0.0
        ms_per_frame = (dur / actual_frames) * 1000.0
        print(f"{name:<38} | {dur:>9.3f}s | {ms_per_frame:>9.2f}ms | {pct:>9.1f}%")
    print("-" * 77)
    print(f"{'Total (Pipeline Stages Sum)':<38} | {total_time:>9.3f}s | {(total_time / actual_frames) * 1000:>9.2f}ms | 100.0%")
    effective_fps = actual_frames / total_time if total_time > 0 else 0.0
    print(f"Effective Cumulative Speed: {effective_fps:.1f} FPS\n")

    # Detailed Micro-Optimization Comparisons
    print("-" * 75)
    print("  MICRO-OPTIMIZATION COMPARISONS:")
    print("-" * 75)
    print(
        f"  * Upscaling to full grid ({grid.out_w}x{grid.out_h}):\n"
        f"      cv2.resize (INTER_NEAREST) : {t_upscale_nearest:.3f}s ({(t_upscale_nearest / actual_frames) * 1000:.2f} ms/frame)\n"
        f"      np.repeat on both axes     : {t_upscale_repeat:.3f}s ({(t_upscale_repeat / actual_frames) * 1000:.2f} ms/frame)\n"
        f"      Speedup: {t_upscale_repeat / t_upscale_nearest:.2f}x faster using cv2.INTER_NEAREST\n"
    )
    print(
        f"  * Multiply & Combine buffer allocation:\n"
        f"      Allocating new array/frame : {t_multiply_allocated:.3f}s ({(t_multiply_allocated / actual_frames) * 1000:.2f} ms/frame)\n"
        f"      In-place pre-allocated buf : {t_multiply_inplace:.3f}s ({(t_multiply_inplace / actual_frames) * 1000:.2f} ms/frame)\n"
        f"      Speedup: {t_multiply_allocated / t_multiply_inplace:.2f}x faster with pre-allocated buffer\n"
    )

    return {
        "t_decode": t_decode,
        "t_downsample": t_downsample,
        "t_mask": t_mask,
        "t_colourise_opt": t_colourise_opt,
        "t_colourise_baseline": t_colourise_baseline,
        "t_upscale_nearest": t_upscale_nearest,
        "t_upscale_repeat": t_upscale_repeat,
        "t_multiply_allocated": t_multiply_allocated,
        "t_multiply_inplace": t_multiply_inplace,
        "t_encoder": t_encoder,
        "total_time": total_time,
    }


def benchmark_multiprocessing(video_path: str, num_frames: int = 200, cols: int = 200) -> None:
    """Benchmark end-to-end pipeline throughput across different worker pool counts."""
    print("=" * 75)
    print(f"  END-TO-END PIPELINE THROUGHPUT (1 vs 2 vs 4 vs 8 workers)")
    print(f"  Processing {num_frames} frames with cols={cols}...")
    print("=" * 75)

    worker_counts = [1, 2, 4, 8]
    results: list[tuple[int, float, float]] = []

    for w in worker_counts:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_out:
            tmp_out_path = tmp_out.name

        try:
            config = RenderConfig(
                source_path=video_path,
                output_path=tmp_out_path,
                cols=cols,
                workers=w,
                limit=num_frames,
            )
            t0 = time.perf_counter()
            convert(config, progress=False)
            elapsed = time.perf_counter() - t0
            fps = num_frames / elapsed if elapsed > 0 else 0.0
            results.append((w, elapsed, fps))
        finally:
            if os.path.exists(tmp_out_path):
                os.remove(tmp_out_path)

    baseline_fps = results[0][2]
    print(f"\n{'Workers':<10} | {'Total Time':>12} | {'FPS':>10} | {'Speedup':>10}")
    print("-" * 52)
    for w, elapsed, fps in results:
        speedup = fps / baseline_fps if baseline_fps > 0 else 1.0
        print(f"{w:<10} | {elapsed:>11.2f}s | {fps:>9.1f} | {speedup:>9.2f}x")
    print("-" * 52)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark binvid performance across pipeline stages and worker counts.")
    parser.add_argument("video", nargs="?", default=None, help="Input video path (defaults to generated synthetic 1080p video)")
    parser.add_argument("--frames", type=int, default=200, help="Number of frames to benchmark over (default: 200)")
    parser.add_argument("--cols", type=int, default=200, help="Grid column count (default: 200)")
    parser.add_argument("--skip-mp", action="store_true", help="Skip multiprocessing scaling benchmark")
    args = parser.parse_args()

    temp_video = None
    video_path = args.video

    if video_path is None or not os.path.isfile(video_path):
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_file.close()
        temp_video = temp_file.name
        generate_synthetic_video(temp_video, num_frames=args.frames, width=1920, height=1080)
        video_path = temp_video

    try:
        benchmark_stages(
            video_path=video_path,
            cols=args.cols,
            num_frames=args.frames,
        )
        if not args.skip_mp:
            benchmark_multiprocessing(
                video_path=video_path,
                num_frames=args.frames,
                cols=args.cols,
            )
    finally:
        if temp_video and os.path.exists(temp_video):
            os.remove(temp_video)


if __name__ == "__main__":
    main()
