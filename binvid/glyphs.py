from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def build_atlas(
    font_path: str,
    font_size: int,
    cell_w: int,
    cell_h: int,
) -> np.ndarray:
    """Build a pre-rendered glyph atlas for binary digits '0' and '1'.

    This function should be called ONCE at startup; do not call PIL drawing
    functions inside the per-frame loop.

    Parameters:
      font_path: Path to the TrueType / OpenType font file.
      font_size: Font size in pixels/points.
      cell_w: Character cell pixel width.
      cell_h: Character cell pixel height.

    Returns:
      A uint8 numpy array of shape (2, cell_h, cell_w), where:
        - atlas[0] is the glyph for "0"
        - atlas[1] is the glyph for "1"
        - values range from 0 (black background) to 255 (white foreground).
    """
    if not os.path.isfile(font_path):
        raise FileNotFoundError(f"Font file not found: '{font_path}'")
    if font_size <= 0:
        raise ValueError(f"font_size must be positive, got {font_size}.")
    if cell_w <= 0 or cell_h <= 0:
        raise ValueError(f"cell_w and cell_h must be positive, got ({cell_w}, {cell_h}).")

    font = ImageFont.truetype(font_path, font_size)
    atlas = np.zeros((2, cell_h, cell_w), dtype=np.uint8)

    for idx, char in enumerate(("0", "1")):
        img = Image.new("L", (cell_w, cell_h), 0)
        draw = ImageDraw.Draw(img)

        bbox = font.getbbox(char)
        if bbox is not None and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            left, top, right, bottom = bbox
            glyph_w = right - left
            glyph_h = bottom - top
            draw_x = round((cell_w - glyph_w) / 2.0 - left)
            draw_y = round((cell_h - glyph_h) / 2.0 - top)
        else:
            draw_x, draw_y = 0, 0

        draw.text((draw_x, draw_y), char, font=font, fill=255)
        atlas[idx] = np.array(img, dtype=np.uint8)

    return atlas


def preview_atlas(
    atlas: np.ndarray,
    scale: int = 8,
    output_path: str = "atlas_preview.png",
) -> str:
    """Upscale both glyphs using nearest-neighbour interpolation and save side-by-side.

    Parameters:
      atlas: uint8 numpy array of shape (2, cell_h, cell_w).
      scale: integer nearest-neighbour upsampling factor.
      output_path: file destination path for the preview PNG.

    Returns:
      Absolute path to the saved preview image.
    """
    if atlas.ndim != 3 or atlas.shape[0] < 2:
        raise ValueError(f"Expected atlas with shape (2, cell_h, cell_w), got {atlas.shape}.")
    if scale < 1:
        raise ValueError(f"scale must be at least 1, got {scale}.")

    # Combine glyph 0 and glyph 1 side by side
    combined = np.hstack([atlas[0], atlas[1]])

    # Upscale with nearest-neighbor
    upscaled = np.repeat(np.repeat(combined, scale, axis=0), scale, axis=1)

    img = Image.fromarray(upscaled, mode="L")
    img.save(output_path)
    return os.path.abspath(output_path)


def main() -> None:
    """Build glyph atlas using system monospace font and save a preview image."""
    from binvid.environment import find_monospace_font

    font_path = find_monospace_font()
    print(f"Building glyph atlas using font: {font_path}")

    # Standard cell dimensions matching default RenderConfig
    cell_w, cell_h = 8, 14
    font_size = 12

    atlas = build_atlas(font_path, font_size, cell_w, cell_h)
    preview_file = preview_atlas(atlas, scale=8, output_path="atlas_preview.png")

    print(f"Atlas generated successfully:")
    print(f"  Shape : {atlas.shape}")
    print(f"  Dtype : {atlas.dtype}")
    print(f"  Range : [{atlas.min()}, {atlas.max()}]")
    print(f"Preview image saved to: {preview_file}")


if __name__ == "__main__":
    main()
