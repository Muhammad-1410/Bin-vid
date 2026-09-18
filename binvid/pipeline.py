from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import Callable, Iterator

import cv2
import numpy as np
from tqdm import tqdm

from binvid.config import RenderConfig
from binvid.digits import DigitField, ScrollField
from binvid.encoder import FrameSink
from binvid.environment import find_ffmpeg, find_monospace_font
from binvid.geometry import GridSpec, compute_grid
from binvid.glyphs import build_atlas
from binvid.probe import frame_iter, probe
from binvid.render import render_frame

_WORKER_CTX: dict = {}


def _init_render_worker(
    grid: GridSpec,
    atlas: np.ndarray,
    mode: str,
    tint_color: tuple[int, int, int] = (0, 255, 70),
    bg_color: tuple[int, int, int] = (0, 0, 0),
    invert: bool = False,
    gamma: float = 1.0,
    contrast_boost: bool = False,
    scanlines: bool = False,
    edge_emphasis: bool = False,
) -> None:
    """Initialize worker process: set single-threaded cv2 and allocate local buffers."""
    cv2.setNumThreads(1)
    _WORKER_CTX.update({
        "grid": grid,
        "atlas": atlas,
        "mode": mode,
        "tint_color": tint_color,
        "bg_color": bg_color,
        "invert": invert,
        "gamma": gamma,
        "contrast_boost": contrast_boost,
        "scanlines": scanlines,
        "edge_emphasis": edge_emphasis,
        "out": np.empty((grid.out_h, grid.out_w, 3), dtype=np.uint8),
        "temp_u16": np.empty((grid.out_h, grid.out_w, 3), dtype=np.uint16),
    })


def _render_worker_task(item: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Worker process task: render frame using cached atlas/grid/styles and return a copy."""
    frame, digits = item
    return render_frame(
        bgr_frame=frame,
        digits=digits,
        **_WORKER_CTX,
    ).copy()


def _frame_task_generator(
    source_path: str,
    digit_field: DigitField | ScrollField,
    limit: int | None = None,
    downsample_size: tuple[int, int] | None = None,
    rotation: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield pairs of (frame, digits) up to the specified limit.

    If downsample_size=(cols, rows) is provided, downsamples the frame to the
    character grid resolution, avoiding multi-megabyte IPC transfer overhead
    across multiprocessing worker process boundaries.
    """
    for frame_idx, frame in enumerate(frame_iter(source_path, rotation=rotation)):
        if limit is not None and frame_idx >= limit:
            break
        digits = digit_field.for_frame(frame_idx)
        if downsample_size is not None:
            cols, rows = downsample_size
            frame = cv2.resize(frame, (cols, rows), interpolation=cv2.INTER_AREA)
        yield (frame, digits)


def convert(
    config: RenderConfig,
    progress: bool = True,
    progress_callback: Callable[[int, int | None], None] | None = None,
) -> None:
    """Convert a video file into ASCII-style binary art made of 0 and 1 digits.

    Validates settings, verifies write permissions, probes geometry/rates/rotation,
    pre-renders the glyph atlas, initializes the digit field, and streams processed
    frames into FFmpeg in strict sequence. Supports single-threaded pre-allocated
    buffering or a multi-worker process pool feeding the encoder.
    """
    # 1. Validate config and resolve ffmpeg binary and monospace font
    config.validate()

    # Pre-flight output path write permission check
    out_dir = os.path.dirname(os.path.abspath(config.output_path))
    if out_dir and not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as exc:
            raise PermissionError(f"Cannot create output directory '{out_dir}': {exc}") from exc
    if out_dir and not os.access(out_dir, os.W_OK):
        raise PermissionError(f"No write permission for output directory: '{out_dir}'.")
    if os.path.exists(config.output_path) and not os.access(config.output_path, os.W_OK):
        raise PermissionError(f"No write permission for output file: '{config.output_path}'.")

    ffmpeg_bin = find_ffmpeg()
    font_path = config.font_path if config.font_path is not None else find_monospace_font()

    # 2. Probe source video metadata (including orientation/rotation)
    info = probe(config.source_path)

    # 3. Compute GridSpec and build the glyph atlas once
    grid = compute_grid(info.width, info.height, config.cols, config.cell_w, config.cell_h)
    atlas = build_atlas(font_path, config.font_size, grid.cell_w, grid.cell_h)

    # 4. Construct the chosen digit field strategy
    if config.digit_mode == "scroll":
        digit_field: DigitField | ScrollField = ScrollField(grid.rows, grid.cols, seed=config.seed)
    elif config.digit_mode == "static":
        digit_field = DigitField(grid.rows, grid.cols, seed=config.seed, refresh=0)
    else:  # "refresh"
        digit_field = DigitField(
            grid.rows,
            grid.cols,
            seed=config.seed,
            refresh=config.digit_refresh,
            partial=False,
        )

    # Resolve background and tint colors as (B, G, R)
    bg_color = config.get_bg_bgr()
    tint_color = config.get_tint_bgr()

    # 5. Open FrameSink and stream frames one at a time
    sink = FrameSink(
        source_path=config.source_path,
        output_path=config.output_path,
        width=grid.out_w,
        height=grid.out_h,
        fps=info.fps,
        crf=config.crf,
        ffmpeg_path=ffmpeg_bin,
    )

    frames_written = 0
    start_time = time.perf_counter()
    interrupted = False

    # 6. Determine total frames for progress bar
    total_frames: int | None
    if config.limit is not None:
        if info.frame_count > 0:
            total_frames = min(info.frame_count, config.limit)
        else:
            total_frames = config.limit
    else:
        total_frames = info.frame_count if info.frame_count > 0 else None

    pbar = tqdm(
        total=total_frames,
        disable=not progress,
        unit="frame",
        desc="Rendering",
    )

    try:
        with sink:
            if config.workers > 1:
                # Multiprocessing pool mode:
                # Downsample in generator to minimize IPC socket transfer (38 KB vs 6.2 MB per frame)
                # and use pool.imap to guarantee strict sequential ordering into sink
                task_stream = _frame_task_generator(
                    config.source_path,
                    digit_field,
                    config.limit,
                    downsample_size=(grid.cols, grid.rows),
                    rotation=info.rotation,
                )
                with mp.Pool(
                    processes=config.workers,
                    initializer=_init_render_worker,
                    initargs=(
                        grid,
                        atlas,
                        config.mode,
                        tint_color,
                        bg_color,
                        config.invert,
                        config.gamma,
                        config.contrast_boost,
                        config.scanlines,
                        config.edge_emphasis,
                    ),
                ) as pool:
                    for rendered in pool.imap(_render_worker_task, task_stream, chunksize=4):
                        sink.write(rendered)
                        frames_written += 1
                        pbar.update(1)
                        if progress_callback is not None:
                            progress_callback(frames_written, total_frames)
            else:
                # Single-process mode: pre-allocate output and temp buffers once in-place
                task_stream = _frame_task_generator(
                    config.source_path,
                    digit_field,
                    config.limit,
                    downsample_size=None,
                    rotation=info.rotation,
                )
                cv2.setNumThreads(1)
                out_buf = np.empty((grid.out_h, grid.out_w, 3), dtype=np.uint8)
                temp_u16_buf = np.empty((grid.out_h, grid.out_w, 3), dtype=np.uint16)
                for frame, digits in task_stream:
                    rendered = render_frame(
                        frame,
                        grid,
                        atlas,
                        digits,
                        mode=config.mode,
                        tint_color=tint_color,
                        bg_color=bg_color,
                        invert=config.invert,
                        gamma=config.gamma,
                        contrast_boost=config.contrast_boost,
                        scanlines=config.scanlines,
                        edge_emphasis=config.edge_emphasis,
                        out=out_buf,
                        temp_u16=temp_u16_buf,
                    )
                    sink.write(rendered)
                    frames_written += 1
                    pbar.update(1)
                    if progress_callback is not None:
                        progress_callback(frames_written, total_frames)
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n[Interrupted] Encoding stopped early by user.")
        print(f"Partial playable video saved to: {os.path.abspath(config.output_path)}")
    finally:
        pbar.close()

    # 7. Log summary at the end
    elapsed = time.perf_counter() - start_time
    avg_fps = frames_written / elapsed if elapsed > 0 else 0.0

    print("\n=== binvid Conversion Summary ===")
    print(f"  Source Video       : {config.source_path}")
    print(f"  Output Video       : {config.output_path}")
    print(f"  Grid Dimensions    : {grid.cols}x{grid.rows} characters")
    print(f"  Output Resolution  : {grid.out_w}x{grid.out_h} px")
    print(f"  Frames Written     : {frames_written}")
    print(f"  Workers Used       : {config.workers}")
    print(f"  Wall-Clock Time    : {elapsed:.2f} s")
    print(f"  Average Speed      : {avg_fps:.1f} FPS")
    if interrupted:
        print(f"  Status             : PARTIAL (Interrupted by user)")
    else:
        print(f"  Status             : COMPLETE")
