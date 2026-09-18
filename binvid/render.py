from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from binvid.geometry import GridSpec


_GAMMA_LUTS: dict[float, np.ndarray] = {}


def _get_gamma_lut(gamma: float) -> np.ndarray:
    """Return precomputed 256-entry uint8 lookup table for gamma correction."""
    key = round(gamma, 4)
    if key not in _GAMMA_LUTS:
        lut = np.array([np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255) for i in range(256)], dtype=np.uint8)
        _GAMMA_LUTS[key] = lut
    return _GAMMA_LUTS[key]


def render_frame(
    bgr_frame: np.ndarray,
    grid: GridSpec,
    atlas: np.ndarray,
    digits: np.ndarray,
    mode: str = "color",
    tint_color: tuple[int, int, int] = (0, 255, 70),
    threshold: int = 128,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    invert: bool = False,
    gamma: float = 1.0,
    contrast_boost: bool = False,
    scanlines: bool = False,
    edge_emphasis: bool = False,
    out: np.ndarray | None = None,
    temp_u16: np.ndarray | None = None,
) -> np.ndarray:
    """Render a single video frame into binary ASCII art with configurable visual styling.

    Parameters:
      bgr_frame: Input BGR image (full size or pre-downsampled to (grid.rows, grid.cols, 3)).
      grid: GridSpec containing grid layout and output dimensions.
      atlas: uint8 pre-rendered glyph atlas of shape (2, cell_h, cell_w).
      digits: uint8 array of shape (grid.rows, grid.cols) with values 0 or 1.
      mode: Rendering style:
        - "color": full-color average cell tinting.
        - "mono": grayscale brightness tinted with tint_color (default terminal green).
        - "flat": high-contrast thresholded glyphs at full tint_color brightness.
      tint_color: (B, G, R) color tuple for "mono" and "flat" modes (default (0, 255, 70)).
      threshold: Brightness threshold [0-255] for "flat" mode.
      bg_color: (B, G, R) background canvas color (default pure black (0, 0, 0)).
      invert: When True, renders dark digits on a light/colored background.
      gamma: Gamma correction applied to cell brightness (> 0; ~0.7 boosts dark footage).
      contrast_boost: When True, equalizes luminance histogram to preserve mid-tones.
      scanlines: When True, darkens alternating pixel rows slightly (CRT style).
      edge_emphasis: When True, runs a Sobel filter on downsampled grayscale and biases
        high-gradient cells toward digit '1'.
      out: Optional pre-allocated uint8 BGR array of shape (grid.out_h, grid.out_w, 3).
      temp_u16: Optional pre-allocated uint16 array of shape (grid.out_h, grid.out_w, 3).

    Returns:
      uint8 BGR array of shape (grid.out_h, grid.out_w, 3).
    """
    if mode not in ("color", "mono", "flat"):
        raise ValueError(f"Invalid mode '{mode}'. Supported modes: 'color', 'mono', 'flat'.")

    if digits.shape != (grid.rows, grid.cols):
        raise ValueError(f"digits shape {digits.shape} does not match grid ({grid.rows}, {grid.cols}).")

    if atlas.shape != (2, grid.cell_h, grid.cell_w):
        raise ValueError(f"atlas shape {atlas.shape} does not match grid cells ({grid.cell_h}, {grid.cell_w}).")

    if out is None:
        out = np.empty((grid.out_h, grid.out_w, 3), dtype=np.uint8)
    else:
        if out.shape != (grid.out_h, grid.out_w, 3) or out.dtype != np.uint8:
            raise ValueError(
                f"out buffer must have shape ({grid.out_h}, {grid.out_w}, 3) and uint8 dtype, "
                f"got shape {out.shape} and dtype {out.dtype}."
            )

    if temp_u16 is None:
        temp_u16 = np.empty((grid.out_h, grid.out_w, 3), dtype=np.uint16)
    else:
        if temp_u16.shape != (grid.out_h, grid.out_w, 3) or temp_u16.dtype != np.uint16:
            raise ValueError(
                f"temp_u16 buffer must have shape ({grid.out_h}, {grid.out_w}, 3) and uint16 dtype, "
                f"got shape {temp_u16.shape} and dtype {temp_u16.dtype}."
            )

    # 1. Downsample the source frame to (grid.cols, grid.rows) using cv2.INTER_AREA
    if bgr_frame.shape[:2] == (grid.rows, grid.cols):
        small = bgr_frame
    else:
        small = cv2.resize(bgr_frame, (grid.cols, grid.rows), interpolation=cv2.INTER_AREA)

    # 2. Apply optional contrast boost (histogram equalization on luminance)
    if contrast_boost:
        ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        small = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # 3. Apply optional gamma correction
    if abs(gamma - 1.0) > 1e-4:
        small = cv2.LUT(small, _get_gamma_lut(gamma))

    # 4. Apply optional edge emphasis (Sobel filter biasing high-gradient cells to '1')
    if edge_emphasis:
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray_small, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_small, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        p75 = float(np.percentile(mag, 75))
        threshold_val = max(20.0, p75)
        edge_mask = mag >= threshold_val
        if np.any(edge_mask):
            digits = digits.copy()
            digits[edge_mask] = 1

    # 5. Build the full-resolution glyph mask using numpy fancy indexing
    tiles = atlas[digits]
    mask = tiles.transpose(0, 2, 1, 3).reshape(grid.out_h, grid.out_w)

    # 6. Determine cell colors and upscale to full canvas size with cv2.INTER_NEAREST
    if mode == "color":
        colour_full = cv2.resize(small, (grid.out_w, grid.out_h), interpolation=cv2.INTER_NEAREST)

    elif mode == "mono":
        # Grayscale brightness tinted by tint_color
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        tint_arr = np.array(tint_color, dtype=np.uint16)
        small_tinted = ((gray[..., None].astype(np.uint16) * tint_arr) // 255).astype(np.uint8)
        colour_full = cv2.resize(small_tinted, (grid.out_w, grid.out_h), interpolation=cv2.INTER_NEAREST)

    else:  # mode == "flat"
        # Harsh high-contrast look: glyphs above threshold are full brightness tint_color
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        tint_arr = np.array(tint_color, dtype=np.uint8)
        small_flat = np.where(gray[..., None] >= threshold, tint_arr, 0).astype(np.uint8)
        colour_full = cv2.resize(small_flat, (grid.out_w, grid.out_h), interpolation=cv2.INTER_NEAREST)

    # 7. Blend glyph strokes and background canvas considering invert & bg_color
    m = mask[..., None].astype(np.uint16)
    if invert:
        # Inverted: digit stroke takes bg_color, cell background takes colour_full
        stroke_col = np.array(bg_color, dtype=np.uint16)
        canvas_col = colour_full
    else:
        # Normal: digit stroke takes colour_full, cell background takes bg_color
        stroke_col = colour_full
        canvas_col = np.array(bg_color, dtype=np.uint16)

    m_inv = 255 - m
    temp_u16[...] = (m * stroke_col + m_inv * canvas_col) // 255
    out[...] = temp_u16

    # 8. Apply optional CRT scanlines
    if scanlines:
        cv2.convertScaleAbs(out[1::2, :], dst=out[1::2, :], alpha=0.75)

    return out


def main() -> None:
    """CLI to render a single frame from a video file and save frame_preview.png."""
    parser = argparse.ArgumentParser(description="Render a single frame from a video to binary ASCII.")
    parser.add_argument("path", help="Path to input video file")
    parser.add_argument("--cols", type=int, default=200, help="Number of character columns (default: 200)")
    parser.add_argument("--cell-w", type=int, default=8, help="Cell width in pixels (default: 8)")
    parser.add_argument("--cell-h", type=int, default=14, help="Cell height in pixels (default: 14)")
    parser.add_argument("--font-size", type=int, default=12, help="Font size in pt (default: 12)")
    parser.add_argument("--mode", choices=["color", "mono", "flat"], default="color", help="Render mode (default: color)")
    parser.add_argument("--output", default="frame_preview.png", help="Output preview image path (default: frame_preview.png)")
    args = parser.parse_args()

    from binvid.digits import DigitField
    from binvid.environment import find_monospace_font
    from binvid.geometry import compute_grid
    from binvid.glyphs import build_atlas
    from binvid.probe import frame_iter, probe

    video_info = probe(args.path)
    grid = compute_grid(video_info.width, video_info.height, args.cols, args.cell_w, args.cell_h)

    font_path = find_monospace_font()
    atlas = build_atlas(font_path, args.font_size, grid.cell_w, grid.cell_h)
    digits = DigitField(grid.rows, grid.cols, seed=42).for_frame(0)

    first_frame = next(frame_iter(args.path))
    rendered = render_frame(first_frame, grid, atlas, digits, mode=args.mode)

    cv2.imwrite(args.output, rendered)
    print(f"Rendered frame saved to: {os.path.abspath(args.output)}")
    print(f"  Source dimensions : {video_info.width}x{video_info.height}")
    print(f"  Grid dimensions   : {grid.cols}x{grid.rows} characters")
    print(f"  Output dimensions : {grid.out_w}x{grid.out_h} pixels")
    print(f"  Render mode       : {args.mode}")


if __name__ == "__main__":
    main()
